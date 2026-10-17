"""Sequential, fail-closed frozen sweep execution using existing billed stages."""
from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path

import torch

from .checkpoint_plan import checkpoint_storage_plan
from .checkpoint_storage import require_checkpoint_space
from .config import Config, ModelConfig, TrainConfig
from .evaluate import evaluate_checkpoint, compare_evaluations
from .reporting import summarize_runs
from .search import plan_sweep
from .sweep_retention import intermediate_payload, retention_forecast, reject_symlink_chain, unlink_payload
from .train import train_run


def _sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def _sources():
    package = Path(__file__).parent
    return {str(path.relative_to(package)): _sha(path)
            for path in sorted(package.rglob('*.py'))}


def _cfg(value):
    value = copy.deepcopy(value)
    return Config(model=ModelConfig(**value['model']), train=TrainConfig(**value['train']), raw=value)


def _snapshot(path):
    path = Path(path)
    files = {}
    for item in sorted(path.rglob('*')):
        if item.is_symlink():
            raise ValueError('stage artifacts must not be symlinks')
        if item.is_file():
            files[str(item.relative_to(path))] = {'bytes': item.stat().st_size, 'sha256': _sha(item)}
    if not files:
        raise ValueError('stage did not publish artifacts')
    return files


def _atomic(path, value):
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _checkpoint(path, cfg, expected_tokens, *, manifest_hash, cooldown_steps=None, parent=None):
    """Fresh stage semantic gate, mmap storage instead of materializing a model."""
    path = Path(path)
    manifest = json.loads((path / 'manifest.json').read_text())
    if manifest.get('format_version') != 1 or manifest.get('data_manifest_hash') != manifest_hash:
        raise ValueError('checkpoint schema/data identity mismatch')
    state = path / 'state.pt'
    expected = manifest['files']['state.pt']
    if state.stat().st_size != expected['bytes'] or _sha(state) != expected['sha256']:
        raise ValueError('checkpoint checksum mismatch')
    payload = torch.load(state, map_location='cpu', weights_only=False, mmap=True)
    if (payload['config'] != cfg or payload['progress']['tokens'] != expected_tokens or
            payload.get('manifest_hash') != manifest_hash or payload.get('parent') != parent or manifest.get('parent') != parent):
        raise ValueError('stage checkpoint config/token endpoint mismatch')
    progress, extra = payload['progress'], payload.get('extra') or {}
    if cooldown_steps is None:
        if extra.get('cooldown_steps') is not None:
            raise ValueError('parent/branch checkpoint is already cooled')
    elif (extra.get('cooldown_steps') != cooldown_steps or
          extra.get('cooldown_end_tokens') != expected_tokens or
          progress.get('scheduler_step') != cooldown_steps or
          progress.get('cooldown_end_tokens') != expected_tokens):
        raise ValueError('cooldown checkpoint horizon/endpoint mismatch')
    return {'path': str(path), 'tokens': expected_tokens,
            'manifest_sha256': _sha(path / 'manifest.json'), 'state_sha256': expected['sha256']}


def run_sweep(base_cfg, candidates, out_dir, *, max_seconds, evaluation_max_seconds,
              device='cuda', cooldown=True, max_prediction_tokens=1_000_000,
              bootstrap_samples=1000, reference_candidate=None, prune_completed_intermediates=False,
              canary_steps=0, min_tokens_per_second=None, canary_warmup_steps=12):
    """Execute fresh full-rung parents, exact-endpoint cooldown forks and validation.

    max_seconds bounds EACH training segment; evaluation_max_seconds bounds each
    evaluation. Failed/interrupted stages require explicit external recovery; no
    charged partial stage is retried. Explicit opt-in pruning removes only
    completed intermediate checkpoint payloads after successful evaluation.
    Completed stages are
    reused only after streaming hashes of every frozen artifact match. Stage APIs
    own their billing; orchestration/hash/comparison/summary overhead is not separately billed.
    """
    from .throughput_canary import ThroughputCanary
    if type(canary_steps) is not int or canary_steps < 0:
        raise ValueError('canary_steps must be a nonnegative integer')
    canary_policy = None
    if canary_steps:
        gate = ThroughputCanary(canary_steps, min_tokens_per_second, canary_warmup_steps)
        canary_policy = {'measured_steps': canary_steps, 'min_tokens_per_second': min_tokens_per_second,
                         'warmup_steps': canary_warmup_steps,
                         'scope': 'first measured parent updates after explicitly excluded warmup; sum targets / sum step seconds; no extra training trajectory'}
    elif min_tokens_per_second is not None or canary_warmup_steps != 12:
        raise ValueError('canary threshold/warmup requires positive canary_steps')
    if type(prune_completed_intermediates) is not bool:
        raise ValueError('prune_completed_intermediates must be Boolean')
    for value in (max_seconds, evaluation_max_seconds):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError('stage time caps must be finite positive numbers')
    if type(cooldown) is not bool or type(max_prediction_tokens) is not int or max_prediction_tokens <= 0:
        raise ValueError('cooldown must be bool and prediction budget positive integer')
    if type(bootstrap_samples) is not int or not 0 <= bootstrap_samples <= 100_000:
        raise ValueError('bootstrap_samples must be in [0,100000]')
    context = base_cfg.raw.get('data', {}).get('context', 512)
    if type(context) is not int or context <= 0:
        raise ValueError('evaluation context must be positive integer')
    device = str(torch.device(device))
    frozen = plan_sweep(base_cfg, candidates)
    if any(run['name'] in ('summary', 'sweep.json', 'sweep.json.tmp', 'comparisons') for run in frozen['runs']):
        raise ValueError('candidate name is reserved for sweep orchestration')
    reference_candidate = frozen['runs'][0]['name'] if reference_candidate is None else reference_candidate
    if not isinstance(reference_candidate, str) or reference_candidate not in {run['name'] for run in frozen['runs']}:
        raise ValueError('reference_candidate must name a declared candidate')
    flags = [run['config'].get('project', {}).get('dry_run', True) for run in frozen['runs']]
    if any(type(flag) is not bool for flag in flags) or len(set(flags)) != 1:
        raise ValueError('all candidates must share an explicit Boolean dry_run policy')
    dry = flags[0]
    out = Path(out_dir).resolve()
    trajectories = []
    validation_hashes = set()
    evaluation_seeds = set()
    storage = 0
    for run in frozen['runs']:
        cfg = _cfg(run['config'])
        data = cfg.raw.get('data', {})
        if cooldown and cfg.train.schedule != 'constant_with_separate_cooldown':
            raise ValueError('separate sweep cooldown requires constant_with_separate_cooldown; use --no-cooldown for cosine')
        identities = {}
        for label, value in [('training', data.get('manifest')),
                ('validation', cfg.raw.get('evaluation', {}).get('validation_manifest', data.get('manifest')))]:
            path = Path(value).resolve() if value is not None else None
            if path is not None and path.is_dir():
                path /= 'manifest.json'
            if path is None or not path.is_file():
                if not dry:
                    raise ValueError(f'{label} manifest must exist before execution')
                identities[label] = {'path': str(path) if path else None, 'sha256': None}
            else:
                identities[label] = {'path': str(path), 'sha256': _sha(path)}
        validation_hashes.add(identities['validation']['sha256'])
        seed = cfg.raw.get('evaluation', {}).get('seed', 0)
        if type(seed) is not int or seed < 0:
            raise ValueError('evaluation seed must be a nonnegative integer')
        evaluation_seeds.add(seed)
        limit = cfg.train.rung_total_tokens
        batch = cfg.train.global_prediction_tokens_per_update
        if canary_policy and (limit + batch - 1) // batch < gate.required_updates:
            raise ValueError('parent horizon cannot complete the requested throughput canary')
        threshold = 9 * limit // 10
        branch = ((threshold + batch - 1) // batch) * batch
        if cooldown and not 0 < branch < limit:
            raise ValueError('rung has no proper optimizer-boundary 90% cooldown branch')
        horizon = (limit - branch + batch - 1) // batch if cooldown else None
        parent_plan = checkpoint_storage_plan(cfg)
        cooled_plan = checkpoint_storage_plan(cfg, start_tokens=branch, token_limit=limit,
                         max_steps=horizon, cooldown=True) if cooldown else None
        storage += parent_plan['required_additional_bytes'] + (cooled_plan['required_additional_bytes'] if cooled_plan else 0)
        rate = cfg.raw.get('project', {}).get('gpu_hourly_rate_usd')
        margin = cfg.raw.get('project', {}).get('cost_margin_fraction', .1)
        if isinstance(margin, bool) or not isinstance(margin, (int, float)) or not math.isfinite(margin) or margin < 0:
            raise ValueError('cost margin must be finite nonnegative')
        cap_cost = float(rate) * ((2 if cooldown else 1) * max_seconds + evaluation_max_seconds) / 3600 if rate is not None else None
        trajectories.append({'paths': {'parent': str(out / run['name'] / 'parent'), 'cooldown': str(out / run['name'] / 'cooldown') if cooldown else None, 'evaluation': str(out / run['name'] / 'evaluation')}, 'runtime_cap_based_usd': cap_cost, 'reservation_cap_based_usd': cap_cost * (1 + margin) if cap_cost is not None else None, 'name': run['name'], 'config': run['config'], 'config_sha256': run['config_sha256'],
            'data': identities, 'tokens': limit, 'parent_steps': (limit + batch - 1) // batch,
            'branch_threshold': threshold, 'branch_tokens': branch if cooldown else None,
            'cooldown_steps': horizon, 'cooldown_end_tokens': limit if cooldown else None,
            'parent_storage_plan': parent_plan, 'cooldown_storage_plan': cooled_plan})
    if len(evaluation_seeds) != 1:
        raise ValueError('sweep candidates require one identical evaluation seed')
    if len(validation_hashes) != 1:
        raise ValueError('sweep candidates require one identical frozen validation manifest')
    identity = {'version': 1, 'out_dir': str(out), 'trajectories': trajectories,
        'controls': {'throughput_canary': canary_policy, 'max_seconds': max_seconds, 'evaluation_max_seconds': evaluation_max_seconds,
                     'device': device, 'cooldown': cooldown, 'max_prediction_tokens': max_prediction_tokens,
                     'bootstrap_samples': bootstrap_samples, 'evaluation_context': context, 'evaluation_seed': next(iter(evaluation_seeds)), 'comparison_seed': next(iter(evaluation_seeds)), 'reference_candidate': reference_candidate, 'prune_completed_intermediates': prune_completed_intermediates},
        'runtime_source_sha256': _sources()}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    forecast = retention_forecast(trajectories) if prune_completed_intermediates else None
    if forecast is not None:
        storage = forecast['required_additional_bytes']
    planned = {'retention_forecast': forecast, 'retention_scope': ('final, any configured initial, and all metadata retained; completed milestone/recovery payloads no longer resumable' if prune_completed_intermediates else 'all completed stage artifacts retained'), 'kind': 'frozen_sweep_execution', 'dry_run': dry, 'identity': identity, 'identity_sha256': digest,
        'required_additional_bytes': storage,
        'comparison_pairs': [{'reference': reference_candidate, 'candidate': run['name']} for run in trajectories if run['name'] != reference_candidate],
        'runtime_cap_based_usd': sum(r['runtime_cap_based_usd'] for r in trajectories) if all(r['runtime_cap_based_usd'] is not None for r in trajectories) else None,
        'reservation_cap_based_usd': sum(r['reservation_cap_based_usd'] for r in trajectories) if all(r['reservation_cap_based_usd'] is not None for r in trajectories) else None,
        'runtime_cap_cost_scope': 'rate times declared per-stage time caps, not guaranteed upper bound: steps/checkpoints may overrun; ledger actuals authoritative',
        'scope': 'full configured schedule plus separately billed optional cooldown/evaluation; constant parents permit continuation, cosine horizon is fixed; no final test',
        'training_exposure_scope': 'cooldown repeats the suffix from the 90% branch in addition to the full parent trajectory',
        'cost_scope': 'cap-based estimates include all declared stages; each stage independently admits/bills actual elapsed time; orchestration/hash/comparison/summary overhead unbilled'}
    if dry:
        return planned
    out.mkdir(parents=True, exist_ok=True)
    with (out / '.sweep.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('another sweep invocation holds the output lock') from exc
        journal_path = out / 'sweep.json'
        if journal_path.exists():
            journal = json.loads(journal_path.read_text())
            if journal.get('identity_sha256') != digest or journal.get('identity') != identity:
                raise ValueError('frozen sweep identity mismatch')
            if any(stage['status'] != 'complete' for stage in journal['stages'].values()):
                raise RuntimeError('interrupted/failed sweep stage requires explicit recovery; automatic retry forbidden')
        else:
            if any(path.name != '.sweep.lock' for path in out.iterdir()):
                raise FileExistsError('sweep output contains unjournaled artifacts')
            journal = {**planned, 'stages': {}, 'prune_operations': {}, 'status': 'pending'}
            _atomic(journal_path, journal)
        def bill(ledger, directory):
            value = json.loads(Path(ledger).read_text())['runs'][str(Path(directory).resolve())]
            if value.get('status') != 'complete' or value.get('reserved_usd') != 0:
                raise ValueError('stage billing is not terminal complete')
            return {'ledger': str(ledger), 'run_id': str(Path(directory).resolve()), 'spent_usd': value['spent_usd']}
        def sources_unchanged():
            if _sources() != identity['runtime_source_sha256']:
                raise ValueError('frozen runtime sources changed during sweep')
        def inputs(run):
            sources_unchanged()
            for item in run['data'].values():
                if _sha(item['path']) != item['sha256']:
                    raise ValueError('frozen data manifest changed before stage')
        def prune_files(run):
            files = {}
            for phase in ('parent', 'cooldown'):
                key = run['name'] + '/' + phase
                if key not in journal['stages']:
                    continue
                for relative, info in journal['stages'][key]['artifacts'].items():
                    if intermediate_payload(relative):
                        files[key + '/' + relative] = {'stage': key, 'artifact': relative, **info}
            return files
        def checked_operations():
            operations = journal.get('prune_operations', {})
            if not isinstance(operations, dict) or (operations and not prune_completed_intermediates):
                raise ValueError('unauthorized prune journal')
            runs = {run['name']: run for run in trajectories}
            for name, operation in operations.items():
                if (name not in runs or operation.get('status') not in ('planned', 'complete')
                        or operation.get('files') != prune_files(runs[name])):
                    raise ValueError('prune journal does not match authorized original payloads')
                keys = [name + '/parent', name + '/evaluation']
                if runs[name]['cooldown_storage_plan'] is not None:
                    keys.append(name + '/cooldown')
                if any(journal['stages'].get(key, {}).get('status') != 'complete' for key in keys):
                    raise ValueError('pruning requires completed candidate evaluation/training')
            return operations
        def verify_stage(key, directory, ledger=None):
            record = journal['stages'][key]
            reject_symlink_chain(out, directory)
            actual = _snapshot(directory)
            expected = dict(record['artifacts'])
            for operation in checked_operations().values():
                for item in operation['files'].values():
                    if item['stage'] == key:
                        if operation['status'] == 'complete' and item['artifact'] in actual:
                            raise ValueError('completed prune payload unexpectedly reappeared')
                        if item['artifact'] not in actual:
                            expected.pop(item['artifact'])
            if actual != expected:
                raise ValueError(f'completed stage artifact hash mismatch: {key}')
            if json.loads((Path(directory) / '.sweep-stage.json').read_text()) != record['result']:
                raise ValueError('cached stage result differs from frozen artifact')
            if ledger is not None and bill(ledger, directory) != record['billing']:
                raise ValueError('completed stage billing identity changed')
            return record['result']
        def prune_candidate_locked(run):
            name = run['name']
            operations = journal.setdefault('prune_operations', {})
            checked_operations()
            # Validate every retained/slated candidate file and every bill before
            # the first unlink, including recovery of a partially deleted plan.
            keys = [name + '/parent', name + '/evaluation']
            if run['cooldown_storage_plan'] is not None:
                keys.append(name + '/cooldown')
            ledger = run['config']['project']['spend_ledger']
            for key in keys:
                verify_stage(key, out / key, ledger)
            slated_directories = {str((out / relative).parent) for relative in prune_files(run)}
            own_runs = {str(out / key) for key in keys}
            billing = json.loads(Path(ledger).read_text())
            for run_id, entry in billing['runs'].items():
                reference = entry.get('parent_checkpoint')
                if (run_id not in own_runs and reference is not None
                        and str(Path(reference).resolve()) in slated_directories
                        and (entry.get('status') != 'complete' or entry.get('reserved_usd') != 0)):
                    raise ValueError('external noncomplete/reserved run pins a slated checkpoint payload')
            if name not in operations:
                files = prune_files(run)
                operations[name] = {'status': 'planned', 'files': files,
                                    'original_payload_bytes': sum(item['bytes'] for item in files.values()),
                                    'scope': 'only completed direct milestone/recovery state.pt payloads; manifests/provenance preserved'}
                _atomic(journal_path, journal)  # Durable authorization BEFORE unlink.
            operation = operations[name]
            if operation['status'] == 'complete':
                return
            for relative, item in operation['files'].items():
                unlink_payload(out, relative, item)
            operation['status'] = 'complete'
            _atomic(journal_path, journal)
        def prune_candidate(run):
            # Same lock used by SpendLedger.begin: no concurrent cooperating
            # admission can pin a checkpoint between pin check and unlink.
            ledger = Path(run['config']['project']['spend_ledger'])
            with Path(str(ledger) + '.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                prune_candidate_locked(run)
        # Incomplete charged stages still fail closed above. Planned pruning is
        # separately recoverable without relaunching any charged stage.
        checked_operations()
        for run in trajectories:
            if journal.get('prune_operations', {}).get(run['name'], {}).get('status') == 'planned':
                prune_candidate(run)
        def stage(key, directory, callback, ledger=None):
            directory = Path(directory)
            if key in journal['stages']:
                return verify_stage(key, directory, ledger)
            if directory.exists():
                raise FileExistsError(f'unjournaled stage output: {directory}')
            journal['stages'][key] = {'status': 'running'}
            journal['status'] = 'running'
            _atomic(journal_path, journal)
            try:
                result = callback()
                sources_unchanged()
                billing = bill(ledger, directory) if ledger is not None else None
                _atomic(directory / '.sweep-stage.json', result)
                artifacts = _snapshot(directory)
                journal['stages'][key] = {'status': 'complete', 'artifacts': artifacts, 'result': result, 'billing': billing}
                _atomic(journal_path, journal)
                return result
            except BaseException as exc:
                journal['stages'][key] = {'status': 'failed', 'error': f'{type(exc).__name__}: {exc}'}
                journal['status'] = 'failed'
                _atomic(journal_path, journal)
                raise
        if journal['status'] != 'complete':
            pending_storage = sum(run['parent_storage_plan']['required_additional_bytes'] * (run['name'] + '/parent' not in journal['stages']) +
                (run['cooldown_storage_plan']['required_additional_bytes'] if run['cooldown_storage_plan'] and run['name'] + '/cooldown' not in journal['stages'] else 0) for run in trajectories)
            if prune_completed_intermediates:
                pending_forecast = retention_forecast(trajectories, journal['stages'])
                pending_storage = pending_forecast['required_additional_bytes']
                journal['pending_retention_forecast'] = pending_forecast
            journal['disk_admission'] = require_checkpoint_space(out, pending_storage)
            _atomic(journal_path, journal)
        training_dirs, evaluation_reports = [], []
        for run in trajectories:
            name, cfg = run['name'], _cfg(run['config'])
            base = out / name
            parent_dir = base / 'parent'
            def parent():
                inputs(run)
                observer = (ThroughputCanary(canary_steps, min_tokens_per_second, canary_warmup_steps)
                            if canary_policy else None)
                result = train_run(cfg, parent_dir, max_steps=run['parent_steps'], max_seconds=max_seconds, device=device,
                                   **({'_throughput_canary': observer} if observer is not None else {}))
                if Path(result['checkpoint']).resolve() != (parent_dir / 'final').resolve():
                    raise ValueError('parent returned checkpoint outside expected final directory')
                final = _checkpoint(result['checkpoint'], run['config'], run['tokens'], manifest_hash=run['data']['training']['sha256'])
                branch = None
                if cooldown:
                    branch = _checkpoint(parent_dir / f'milestone-{run["branch_threshold"]:012d}',
                                         run['config'], run['branch_tokens'], manifest_hash=run['data']['training']['sha256'])
                return {'checkpoint': final, 'branch': branch,
                        **({'throughput_canary': observer.report()} if observer is not None else {})}
            parent_result = stage(name + '/parent', parent_dir, parent, cfg.raw['project']['spend_ledger'])
            training_dirs.append(parent_dir)
            selected_info = parent_result['checkpoint']
            selected = selected_info['path']
            if cooldown:
                cooled_dir = base / 'cooldown'
                def cooled():
                    inputs(run)
                    result = train_run(cfg, cooled_dir, max_steps=run['cooldown_steps'], max_seconds=max_seconds,
                        device=device, resume=parent_result['branch']['path'], cooldown_steps=run['cooldown_steps'],
                        cooldown_end_tokens=run['tokens'])
                    if Path(result['checkpoint']).resolve() != (cooled_dir / 'final').resolve():
                        raise ValueError('cooldown returned checkpoint outside expected final directory')
                    return _checkpoint(result['checkpoint'], run['config'], run['tokens'], cooldown_steps=run['cooldown_steps'], manifest_hash=run['data']['training']['sha256'],
                        parent={'path': parent_result['branch']['path'], 'manifest_sha256': parent_result['branch']['manifest_sha256']})
                selected_info = stage(name + '/cooldown', cooled_dir, cooled, cfg.raw['project']['spend_ledger'])
                selected = selected_info['path']
                training_dirs.append(cooled_dir)
            evaluation_dir = base / 'evaluation'
            def evaluate():
                inputs(run)
                report = evaluate_checkpoint(cfg, selected, evaluation_dir, context=context,
                    max_prediction_tokens=max_prediction_tokens, max_seconds=evaluation_max_seconds,
                    device=device, bootstrap_samples=bootstrap_samples)
                if (report.get('kind') != 'held_out_validation' or not report['frozen_state']['unchanged'] or
                        report['validation_manifest_sha256'] != run['data']['validation']['sha256'] or
                        report['checkpoint_manifest_sha256'] != selected_info['manifest_sha256'] or
                        report['checkpoint_state_sha256'] != selected_info['state_sha256']):
                    raise ValueError('evaluation protocol/state mismatch')
                return {key: report[key] for key in ('checkpoint_manifest_sha256', 'checkpoint_state_sha256',
                    'validation_manifest_sha256', 'target_plan_sha256', 'available_prefix_plan_sha256', 'prediction_tokens', 'ce')}
            report = stage(name + '/evaluation', evaluation_dir, evaluate, cfg.raw['project']['spend_ledger'])
            evaluation_reports.append({'name': name, **report})
            if prune_completed_intermediates:
                prune_candidate(run)
        signatures = {(r['target_plan_sha256'], r['available_prefix_plan_sha256'], r['prediction_tokens']) for r in evaluation_reports}
        if len(signatures) != 1:
            raise ValueError('candidate validation targets/prefixes differ; comparison rejected')
        comparisons = []
        reference_report = out / reference_candidate / 'evaluation' / 'evaluation.json'
        for run in trajectories:
            if run['name'] == reference_candidate:
                continue
            comparison_dir = out / 'comparisons' / run['name']
            def compare():
                report = compare_evaluations(reference_report,
                    out / run['name'] / 'evaluation' / 'evaluation.json', comparison_dir,
                    bootstrap_samples=bootstrap_samples, seed=identity['controls']['comparison_seed'])
                return {'reference_candidate': reference_candidate, 'candidate': run['name'],
                        'path': str(comparison_dir / 'comparison.json'),
                        'delta_ce_right_minus_left': report['delta_ce_right_minus_left'],
                        'delta_ce_95_ci': report['uncertainty']['delta_ce_95_ci'],
                        'uncertainty': report['uncertainty'],
                        'training_comparison': report['training_comparison']}
            comparisons.append(stage('comparison/' + run['name'], comparison_dir, compare))
        summary_dir = out / 'summary'
        def summarize():
            result = summarize_runs(training_dirs, summary_dir)
            _atomic(summary_dir / 'validation.json', {'evaluations': evaluation_reports, 'reference_candidate': reference_candidate,
                'comparisons': comparisons, 'automatic_ranking_eligible': False,
                'scope': 'validation only; one configured seed per candidate; no HPO correction or automatic promotion'})
            return result
        stage('summary', summary_dir, summarize)
        journal['status'] = 'complete'
        _atomic(journal_path, journal)
        return journal
