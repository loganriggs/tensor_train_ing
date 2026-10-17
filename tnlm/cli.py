"""Run as python -m tnlm.cli COMMAND. Training defaults to configuration dry-run."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch

from .checkpoint_plan import checkpoint_storage_plan
from .config import Config
from .model import TensorLM
from .train import full_config, plan_search, train_run, profile_run, summarize_runs


def _checkpoint_disk_check(storage_plan, run_dir=None):
    """Advisory fresh-run projection against the destination filesystem now.

    Validation never creates the destination or reserves space. Without an
    explicit run directory, report the current filesystem rather than assuming
    that the spend ledger and future checkpoints share a filesystem.
    """
    from .checkpoint_storage import RESERVE_BYTES
    destination = Path(run_dir).absolute() if run_dir is not None else Path.cwd()
    info = {'destination': str(destination),
            'destination_source': 'run_dir' if run_dir is not None else 'current_working_directory',
            'required_additional_bytes': storage_plan['required_additional_bytes'],
            'reserve_bytes': RESERVE_BYTES,
            'free_bytes': None, 'filesystem_probe_path': None,
            'scope': 'advisory fresh serial run projection; no reservation; existing artifacts and concurrent writes excluded'}
    try:
        probe = destination
        while not probe.exists():
            parent = probe.parent
            if parent == probe:
                raise FileNotFoundError(str(probe))
            probe = parent
        info['filesystem_probe_path'] = str(probe)
        info['free_bytes'] = shutil.disk_usage(probe).free
    except OSError as error:
        info.update(status='unavailable', error=str(error))
        return info, ['Checkpoint disk availability could not be checked: ' + str(error)]
    required = info['required_additional_bytes'] + RESERVE_BYTES
    if info['free_bytes'] < required:
        info['status'] = 'insufficient'
        return info, [f"Checkpoint storage projection needs {info['required_additional_bytes']} bytes "
                      f"plus {RESERVE_BYTES} bytes reserve, but only {info['free_bytes']} bytes are free "
                      f"on {probe}. Adjust retention or choose another run directory before launch."]
    info['status'] = 'ok'
    return info, []


def validate_config(cfg, device=None, run_dir=None):
    # Meta construction counts parameters without allocating a 100M model.
    class ShapeOnlyLM(TensorLM):
        def reset_parameters(self):
            pass  # counting only: no data-dependent spectral initialization
    with torch.device('meta'):
        model = ShapeOnlyLM(cfg.model)
    counts = model.param_counts()
    context = cfg.raw.get('data', {}).get('context', 512)
    batch = cfg.raw.get('data', {}).get('micro_batch_size', 1)
    audit = model.operator_audit()
    probe_device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    backend = cfg.model.attention_backend
    if backend == 'auto':
        backend = 'triton' if probe_device.type == 'cuda' else 'reference'
    if cfg.model.attention == 'softmax':
        backend = 'flex' if backend == 'triton' else 'softmax'
    from .shapes import memory_plan
    gpu_gb = (torch.cuda.get_device_properties(probe_device).total_memory / 1e9
              if probe_device.type == 'cuda' and torch.cuda.is_available() else 80.0)
    capacity = memory_plan(cfg.model, gpus=1, micro_tokens_per_gpu=batch * context, gpu_gb=gpu_gb)
    # Each polynomial attention branch has degree <=5, followed by degree <=2
    # MLP. Residual sums do not increase this upper bound. Dynamic norms prevent
    # a global polynomial claim; bounds say nothing about tractable contraction.
    degree_bound = 10 ** cfg.model.layers if audit['class'] == 'P' else None
    loss_rows = batch * context
    if cfg.model.loss_chunk_tokens > 0:
        loss_rows = min(loss_rows, cfg.model.loss_chunk_tokens)
    storage_plan = checkpoint_storage_plan(cfg)
    disk_check, warnings = _checkpoint_disk_check(storage_plan, run_dir)
    return {'parameters': counts, 'operator_audit': audit,
            'memory_plan': capacity,
            'checkpoint_storage_plan': storage_plan,
            'checkpoint_disk_check': disk_check, 'warnings': warnings,
            'memory_plan_assumptions': {'gpu_capacity_GB': gpu_gb, 'state_bytes_per_parameter': 16,
                'scope': 'approximate FSDP/AdamW capacity model with measured activation constants and fixed loss-workspace assumptions; not measured peak memory or proof of fit',
                'capacity_source': 'selected CUDA device' if probe_device.type == 'cuda' and torch.cuda.is_available() else 'hypothetical 80 GB GPU'},
            'global_polynomial_degree_upper_bound': degree_bound,
            'tensor_analysis_scope': 'local low-rank factors and explicit normalizers; no guarantee of efficient global contraction',
            'estimated_parameter_optimizer_gradient_bytes': counts['total'] * 16,
            'estimated_one_fp32_attention_matrix_bytes': batch * cfg.model.heads * context**2 * 4,
            'estimated_one_fp32_training_logit_chunk_bytes': loss_rows * cfg.model.vocab_size * 4,
            'scaling': {'default_training_world_size': 1,
                        'distributed_training': {'entrypoint': 'train/cooldown --distributed under torchrun',
                            'optimizer': cfg.train.optimizer, 'supported_optimizers': ['adamw', 'muon'],
                            'compile_supported': False,
                            'validated_transport': 'CPU/gloo', 'recovery_world_size': 'must match checkpoint',
                            'initialization': 'fresh runs stream canonical CPU components; lagged calibration uses a replicated prefix with collective checks; resume allocates local shards; steady-state estimates exclude initialization scratch'},
                        'attention_backend': cfg.model.attention_backend,
                        'resolved_attention_backend': backend,
                        'device': str(probe_device),
                        'cuda_available': torch.cuda.is_available(),
                        'activation_checkpointing': cfg.model.activation_checkpointing,
                        'loss_chunk_tokens': cfg.model.loss_chunk_tokens,
                        'training_materializes_full_logits': cfg.model.loss_chunk_tokens == 0,
                        'inference_materializes_full_logits': True},
            'memory_note': 'individual workspace estimates, not peak memory; exclude activations, simultaneous intermediates, compiler workspace; attention matrix estimate describes dense reference',
            'weight_decay_policy': 'hidden linear-map weights only; embeddings/output vocabulary/gains/biases/head scalars excluded',
            'lagged_collection_policy': 'all nonpadding tokens; once per update; no compile/recompute',
            'config': full_config(cfg)}


def _run_analysis(cfg, args):
    """Meter the Claude-owned diagnostic implementation without changing it."""
    import time
    from .train import SpendLedger, evaluation_validator
    project = cfg.raw.get('project', {})
    data = cfg.raw.get('data', {})
    context = args.context if args.context is not None else data.get('context', 512)
    rw_layer_window = getattr(args, 'rw_layer_window', None)
    if rw_layer_window is not None and (type(rw_layer_window) is not int or rw_layer_window < 0):
        raise ValueError('rw_layer_window must be a nonnegative integer')
    if args.batches <= 0 or context <= 0 or args.max_seconds <= 0:
        raise ValueError('analysis requires positive batches, context and runtime cap')
    if project.get('dry_run', True):
        return {'dry_run': True, 'command': 'analyze', 'checkpoint': args.checkpoint,
                'context': context, 'batches': args.batches, 'max_seconds': args.max_seconds,
                'rw_layer_window': rw_layer_window,
                'note': 'normalization/weight diagnostics; use evaluate for matched-target quality comparisons'}
    try:
        from .analysis import analyze
    except ModuleNotFoundError as error:
        if error.name == 'tnlm.analysis':
            raise RuntimeError('Claude-owned tnlm.analysis is unavailable; no analysis was launched') from error
        raise
    from .checkpoint import load_checkpoint
    from .data import (PackedSampler, manifest_hash, batch_hash, audit_validation_overlap,
                       validate_tokenizer_compatibility)
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(f'analysis output already exists: {out}')
    device = torch.device(args.device)
    rate = project.get('gpu_hourly_rate_usd')
    if rate is None or (device.type == 'cuda' and rate <= 0):
        raise ValueError('declare actual gpu_hourly_rate_usd (explicit 0 allowed for CPU)')
    ledger = SpendLedger(project.get('spend_ledger', 'runs/spend.json'),
                         project.get('total_budget_usd', 1000), project.get('reserve_usd', 150),
                         stage_budgets=project.get('stage_budgets'))
    run_id = str(out.resolve())
    ledger.begin(run_id, rate, args.max_seconds, args.run_budget_usd,
                 parent_checkpoint=args.checkpoint,
                 cost_margin_fraction=project.get('cost_margin_fraction', .1), stage=project.get('stage', 'A'))
    started, status = time.monotonic(), 'failed'
    try:
        evaluation = cfg.raw.get('evaluation', {})
        manifest = evaluation.get('validation_manifest', data.get('manifest'))
        tokenizer_compatibility = validate_tokenizer_compatibility(
            data['manifest'], manifest, allow_test_only=device.type == 'cpu')
        sampler = PackedSampler(manifest, split='val', context=context,
                                batch_size=data.get('micro_batch_size', 1), seed=int(evaluation.get('seed', 0)))
        model = TensorLM(cfg.model).to(device)
        payload = load_checkpoint(args.checkpoint, model, expected_manifest_hash=manifest_hash(data['manifest']),
                        config_validator=evaluation_validator(cfg), restore_rng=False, map_location='cpu')
        checkpoint_code = payload['code']
        checkpoint_progress = payload['progress']
        from .evaluate import _training_descriptor
        training_descriptor = _training_descriptor(payload)
        del payload  # release unused full optimizer/RNG state before diagnostics
        model.eval()
        def state_fingerprint():
            digest = hashlib.sha256()
            # Include nonpersistent statistic accumulators, which state_dict omits.
            for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
                digest.update(name.encode())
                digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
                digest.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
            return digest.hexdigest()
        before_state = state_fingerprint()
        def lifecycle():
            modules = {name: (module.training, getattr(module, 'collecting', None),
                id(getattr(module, '_valid', None)),
                getattr(module, 'denom_mode', None), id(getattr(module, 'denom_saved', None)),
                id(getattr(module, 'denom_const', None)), getattr(module, 'ablate', None),
                tuple(module._forward_hooks), tuple(module._forward_pre_hooks), tuple(module._backward_hooks))
                for name, module in model.named_modules()}
            gradients = {name: parameter.requires_grad for name, parameter in model.named_parameters()}
            return modules, gradients
        before_modes = lifecycle()
        sampler_initial = sampler.state_dict()
        batches, batch_mappings = [], []
        for _ in range(args.batches):
            if time.monotonic() - started >= args.max_seconds:
                raise TimeoutError('analysis runtime cap during probe setup')
            batches.append(tuple(t.to(device) for t in sampler.next_batch()))
            batch_mappings.append(sampler.batch_mappings())
        probe_hashes = [batch_hash(batch) for batch in batches]
        probe_plan = {'version': 1, 'validation_manifest_sha256': manifest_hash(manifest),
            'split': 'val', 'context': context, 'sampler_initial': sampler_initial,
            'batch_mappings': batch_mappings, 'batch_hashes': probe_hashes}
        probe_plan_sha256 = hashlib.sha256(json.dumps(
            probe_plan, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        selected_documents = {item['document_id'] for mapping in batch_mappings
            for row in mapping for item in row['ranges']
            if item['packed_token_start'] < row['input_length']}
        overlap_audit = audit_validation_overlap(data['manifest'], manifest,
                                                document_ids=selected_documents)
        if overlap_audit['has_overlap']:
            raise ValueError('analysis validation has exact whole-document content overlap')
        if time.monotonic() - started >= args.max_seconds:
            raise TimeoutError('analysis runtime cap before diagnostics')
        probe_seed = int(evaluation.get('seed', 0))
        if probe_seed < 0:
            raise ValueError('analysis probe seed must be nonnegative')
        cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
        with torch.random.fork_rng(devices=cuda_devices):
            torch.random.default_generator.manual_seed(probe_seed)
            if cuda_devices:
                torch.cuda.default_generators[cuda_devices[0]].manual_seed(probe_seed)
            analysis_options = {} if rw_layer_window is None else {'rw_layer_window': rw_layer_window}
            result = analyze(model, batches, str(out), **analysis_options)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        if [batch_hash(batch) for batch in batches] != probe_hashes:
            raise RuntimeError('analysis changed probe batch inputs or targets')
        after_state = state_fingerprint()
        after_modes = lifecycle()
        if before_state != after_state or before_modes != after_modes:
            raise RuntimeError('analysis changed model state, modes, collection flags, gradient flags or hooks')
        if not isinstance(result, dict):
            raise TypeError('analyze must return a JSON-serializable dictionary')
        result['training_descriptor'] = training_descriptor
        provenance = {'checkpoint': str(Path(args.checkpoint).resolve()),
            'training_descriptor': training_descriptor,
            'checkpoint_progress': checkpoint_progress,
            'checkpoint_code': checkpoint_code,
            'checkpoint_manifest_sha256': hashlib.sha256((Path(args.checkpoint) / 'manifest.json').read_bytes()).hexdigest(),
            'validation_manifest_sha256': manifest_hash(manifest),
            'split': 'val', 'batches': args.batches, 'context': context,
            'rw_layer_window': rw_layer_window,
            'probe_seed': probe_seed,
            'runtime_source_sha256': {str(path.relative_to(Path(__file__).parent)):
                hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(Path(__file__).parent.rglob('*.py'))},
            'batch_hashes': probe_hashes,
            'probe_plan': probe_plan, 'probe_plan_sha256': probe_plan_sha256,
            'data_overlap_audit': overlap_audit,
            'tokenizer_compatibility': tokenizer_compatibility,
            'frozen_state': {'before': before_state, 'after': after_state, 'unchanged': True},
            'sampler': sampler.state_dict(), 'runtime_cap_seconds': args.max_seconds,
            'cost_scope': 'probe setup and diagnostic execution; opaque diagnostic call cannot be preempted mid-operation'}
        provenance['critical_model_source_matches'] = {name:
            provenance['runtime_source_sha256'].get(name) == checkpoint_code.get('source_sha256', {}).get(f'tnlm/{name}')
            for name in ('model.py', 'attention.py', 'normalization.py', 'config.py')}
        provenance['critical_model_sources_match_checkpoint'] = all(provenance['critical_model_source_matches'].values())
        result['evaluation_provenance'] = provenance
        json.dumps(result, allow_nan=False)
        out.mkdir(parents=True, exist_ok=True)
        (out / 'invocation.json').write_text(json.dumps(provenance, indent=2))
        status = 'complete'
    finally:
        ledger.finish(run_id, time.monotonic() - started, status)
    result['billing'] = json.loads(ledger.path.read_text())['runs'][run_id]
    (out / 'analysis-invocation-result.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    return result



def _distributed_run_from_cli(cfg, args):
    # NCCL / set_device can initialize CUDA before the trainer is entered.
    from .runtime import deterministic_execution, deterministic_requested
    enabled = deterministic_requested(cfg)
    if cfg.raw.get('project', {}).get('dry_run', True):
        return _distributed_run_with_policy(cfg, args)
    with deterministic_execution(enabled, device=args.device):
        return _distributed_run_with_policy(cfg, args)


def _distributed_run_with_policy(cfg, args):
    """Initialize a bounded torchrun group; only its coordinator prints results."""
    import math
    import os
    from datetime import timedelta
    import torch.distributed as dist
    timeout = args.collective_timeout_seconds
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('collective timeout must be finite and positive')
    context_fork = getattr(args, 'context_fork', False)
    if type(context_fork) is not bool or (context_fork and args.resume is None):
        raise ValueError('context_fork requires a Boolean flag and explicit parent checkpoint')
    if cfg.raw.get('project', {}).get('dry_run', True):
        result = plan_search(cfg)
        result.update(distributed=True, collective_timeout_seconds=timeout,
                      context_fork=context_fork, new_context=cfg.raw.get('data', {}).get('context', 512),
                      parent=str(args.resume) if args.resume is not None else None,
                      parent_validation='pending' if args.resume is not None else None,
                      launch='torchrun --standalone --nproc-per-node N -m tnlm.cli train --distributed ...')
        return result if int(os.environ.get('RANK', '0')) == 0 else None
    owned_group = not dist.is_initialized()
    device = torch.device(args.device)
    if owned_group:
        if any(name not in os.environ for name in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'MASTER_ADDR', 'MASTER_PORT')):
            raise ValueError('--distributed requires a torchrun launch with rank/rendezvous environment')
        if device.type == 'cuda':
            if device.index is not None:
                raise ValueError('use --device cuda so torchrun LOCAL_RANK selects each GPU')
            device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
            torch.cuda.set_device(device)
        elif device.type != 'cpu':
            raise ValueError('distributed training supports CPU/gloo or CUDA/NCCL')
        dist.init_process_group('nccl' if device.type == 'cuda' else 'gloo',
                                timeout=timedelta(seconds=timeout))
    try:
        from .distributed_train import distributed_train_run
        rank = dist.get_rank()
        result = distributed_train_run(cfg, args.run_dir, max_steps=args.max_steps,
            max_seconds=args.max_seconds, device=str(device), resume=args.resume,
            context_fork=context_fork,
            cooldown_steps=getattr(args, 'cooldown_steps', None),
            cooldown_end_tokens=getattr(args, 'cooldown_end_tokens', None), run_budget_usd=args.run_budget_usd,
            export_max_state_bytes=getattr(args, 'export_max_state_bytes', None))
        return result if rank == 0 else None
    finally:
        if owned_group:
            dist.destroy_process_group()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('validate_config', 'plan_search'):
        p = sub.add_parser(name)
        p.add_argument('--config', required=True)
        if name == 'plan_search':
            p.add_argument('--candidates', help='YAML/JSON list of named configuration overrides; planning only')
        if name == 'validate_config':
            p.add_argument('--device', help='resolve auto attention for this device without allocating weights')
            p.add_argument('--run-dir', help='future checkpoint destination for advisory free-disk check; defaults to current working directory')
    p = sub.add_parser('sweep-config', help='export one recorded sweep candidate as YAML without training')
    p.add_argument('--sweep', required=True, help='directory containing sweep.json')
    p.add_argument('--candidate', required=True, help='exact recorded candidate name')
    p.add_argument('--out', help='new YAML file; omitted prints YAML to stdout')
    p = sub.add_parser('sweep')
    p.add_argument('--config', required=True)
    p.add_argument('--candidates', required=True, help='YAML/JSON list of named configuration overrides')
    p.add_argument('--out', required=True)
    p.add_argument('--max-seconds', type=float, required=True, help='runtime bound per training segment')
    p.add_argument('--evaluation-max-seconds', type=float, required=True)
    p.add_argument('--max-prediction-tokens', type=int, default=1_000_000)
    p.add_argument('--bootstrap-samples', type=int, default=1000)
    p.add_argument('--device', default='cuda')
    p.add_argument('--no-cooldown', action='store_true', help='evaluate uncooled endpoints explicitly')
    p.add_argument('--reference-candidate', help='predeclared comparison reference; defaults to first candidate')
    p.add_argument('--prune-completed-intermediates', action='store_true',
                   help='after successful evaluation, remove intermediate checkpoint payloads; retain initial/final checkpoints and provenance')
    p.add_argument('--canary-steps', type=int, default=0, help='measured parent updates for optional throughput gate')
    p.add_argument('--min-tok-s', '--expect-tok-s', dest='min_tok_s', type=float,
                   help='minimum scored targets/s; expect-tok-s is an alias for this lower threshold')
    p.add_argument('--canary-warmup-steps', type=int, default=12, help='initial parent updates excluded from canary timing')
    p.add_argument('--snapshot-sources', nargs='?', const='', metavar='DIR',
                   help='fresh sweep only: freeze sources/config/grid then launch isolated Python; default DIR is OUT.sources; dry runs create nothing')
    p = sub.add_parser('reconcile', help='inspect or explicitly reconcile one terminal runtime ledger entry')
    p.add_argument('--ledger', required=True)
    p.add_argument('--run-id', required=True, help='exact ledger run identity')
    p.add_argument('--metrics', action='append', default=[], help='explicit JSON/JSONL elapsed-time evidence; never proves termination')
    p.add_argument('--apply', action='store_true', help='write operator-attested reconciliation; default only inspects')
    p.add_argument('--expected-row-sha256')
    p.add_argument('--operator')
    p.add_argument('--reason')
    p.add_argument('--all-workers-stopped', action='store_true', help='attest this identified job and all ranks/workers have stopped')
    p.add_argument('--observed-elapsed-seconds', type=float)
    totals = p.add_mutually_exclusive_group()
    totals.add_argument('--total-elapsed-seconds', type=float)
    totals.add_argument('--total-cost-usd', type=float)
    p = sub.add_parser('prepare_data')
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--source-jsonl')
    source.add_argument('--hf-dataset')
    p.add_argument('--hf-config')
    p.add_argument('--hf-data-files', nargs='+')
    p.add_argument('--hf-shard-checksums', help='JSON file mapping pinned shard paths to SHA256')
    p.add_argument('--hf-local-shards', help='JSON file mapping pinned shard paths to existing local Parquet files')
    p.add_argument('--revision', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--max-tokens', type=int, default=50_000_000)
    p.add_argument('--max-documents', type=int, default=100_000)
    p = sub.add_parser('prepare_mechanisms')
    p.add_argument('--out', required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--train-examples', type=int, default=32, help='examples per task')
    p.add_argument('--validation-examples', type=int, default=16, help='examples per task')
    p = sub.add_parser('evaluate_mechanisms')
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--task-manifest', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--max-examples', type=int, default=32)
    p.add_argument('--max-prompt-tokens', type=int, help='explicit probe context override; default configured training context')
    p.add_argument('--max-seconds', type=float, required=True)
    p.add_argument('--component', action='append', help='branch removal, e.g. 0:mlp; default first attention and MLP branches')
    p.add_argument('--run-budget-usd', type=float)
    p.add_argument('--device', default='cpu')
    for name in ('train', 'cooldown', 'branch'):
        p = sub.add_parser(name)
        p.add_argument('--config', required=True)
        p.add_argument('--run-dir', required=True)
        p.add_argument('--max-steps', type=int, required=True)
        p.add_argument('--max-seconds', type=float, required=True)
        p.add_argument('--run-budget-usd', type=float)
        p.add_argument('--device', default='cuda')
        p.add_argument('--resume', required=name in ('cooldown', 'branch'))
        p.add_argument('--context-fork', action='store_true', help='explicitly branch a resume checkpoint into a larger configured data.context')
        p.add_argument('--distributed', action='store_true', help='run coordinated FSDP training under torchrun')
        p.add_argument('--export-max-state-bytes', type=int, help='optionally export distributed weights for evaluation, capped at this full CPU state size')
        p.add_argument('--collective-timeout-seconds', type=float, default=120., help='distributed process-group timeout')
        if name == 'branch':
            p.add_argument('--batch-tokens', type=int, required=True, help='new global prediction targets per optimizer update')
            p.add_argument('--lr-scale', type=float, required=True, help='explicit multiplier for every optimizer group learning rate')
        if name == 'cooldown':
            p.add_argument('--cooldown-steps', type=int, required=True)
            p.add_argument('--cooldown-end-tokens', type=int, help='exact cumulative token endpoint; final update may be partial')
    p = sub.add_parser('profile')
    p.add_argument('--config', required=True)
    p.add_argument('--run-dir', required=True)
    p.add_argument('--steps', type=int, default=10)
    p.add_argument('--warmup-steps', type=int, default=2)
    p.add_argument('--kernels', action='store_true', help='rank CUDA kernel/device-operation time over post-warmup training updates')
    p.add_argument('--trace-steps', type=int, default=0,
                   help='export Chrome CPU/CUDA timelines for the first N complete post-warmup updates (0 disables, maximum 100)')
    p.add_argument('--max-seconds', type=float, required=True)
    p.add_argument('--run-budget-usd', type=float)
    p.add_argument('--device', default='cuda')
    p = sub.add_parser('analyze')
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--batches', type=int, default=2)
    p.add_argument('--rw-layer-window', type=int, metavar='K',
                   help='restrict read/write readers to at most K layers ahead (K >= 0; unembedding always included); omitted means all pairs')
    p.add_argument('--context', type=int, help='evaluation context override; checkpoint training config remains fixed')
    p.add_argument('--max-seconds', type=float, default=300)
    p.add_argument('--run-budget-usd', type=float)
    p.add_argument('--device', default='cpu')
    p = sub.add_parser('evaluate')
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--context', type=int)
    p.add_argument('--max-prediction-tokens', type=int, default=1_000_000)
    p.add_argument('--max-seconds', type=float, required=True)
    p.add_argument('--bootstrap-samples', type=int, default=1000)
    p.add_argument('--run-budget-usd', type=float)
    p.add_argument('--device', default='cpu')
    p = sub.add_parser('record_spend')
    p.add_argument('--config', required=True)
    p.add_argument('--charge-id', required=True)
    p.add_argument('--usd', type=float, required=True)
    p.add_argument('--stage', required=True, choices=('A', 'B', 'C', 'D', 'Reserve'))
    p.add_argument('--description', required=True)
    p = sub.add_parser('compare_evaluations')
    comparison_input = p.add_mutually_exclusive_group(required=True)
    comparison_input.add_argument('--left')
    comparison_input.add_argument('--pairs', help='YAML/JSON list of name, left, right report paths; relative paths resolve from this file')
    p.add_argument('--right')
    p.add_argument('--out', required=True)
    p.add_argument('--bootstrap-samples', type=int, default=1000)
    p.add_argument('--seed', type=int, default=0)
    p = sub.add_parser('compare_seed_pairs')
    p.add_argument('--pairs', required=True, help='YAML/JSON list of seed, left, right evaluation-report paths; relative paths resolve from this file')
    p.add_argument('--out', required=True)
    p.add_argument('--allow-differences', nargs='*', default=[], help='explicit training-descriptor dimensions allowed to differ between conditions')
    p = sub.add_parser('summarize')
    p.add_argument('--run-dirs', nargs='+', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--max-plot-points', type=int, default=2000)
    p = sub.add_parser('check_math')
    p.add_argument('--verbose', action='store_true')
    args = parser.parse_args(argv)
    if args.command == 'sweep-config':
        from .sweep_config import export_sweep_config
        exported = export_sweep_config(args.sweep, args.candidate)
        if args.out is None:
            print(exported, end='')
        else:
            with Path(args.out).open('x') as stream:
                stream.write(exported)
        return 0
    if args.command == 'compare_evaluations':
        if args.pairs is not None and args.right is not None:
            parser.error('--pairs cannot be combined with --right')
        if args.left is not None and args.right is None:
            parser.error('--left requires --right')
    if args.command == 'reconcile':
        from .budget import SpendLedger
        ledger = SpendLedger(args.ledger)
        if args.apply:
            if not (args.expected_row_sha256 and args.operator and args.reason and args.all_workers_stopped):
                parser.error('--apply requires --expected-row-sha256, --operator, --reason and --all-workers-stopped')
            result = ledger.reconcile_terminal(args.run_id,
                expected_row_sha256=args.expected_row_sha256, operator=args.operator, reason=args.reason,
                terminal_attestation={'run_id': args.run_id, 'all_workers_stopped': True},
                observed_elapsed_seconds=args.observed_elapsed_seconds,
                total_elapsed_seconds=args.total_elapsed_seconds, total_cost_usd=args.total_cost_usd,
                evidence_paths=args.metrics)
        else:
            if any(value is not None for value in (args.expected_row_sha256, args.operator, args.reason,
                    args.observed_elapsed_seconds, args.total_elapsed_seconds, args.total_cost_usd)) or args.all_workers_stopped:
                parser.error('reconciliation controls require --apply; inspection only accepts ledger, run-id and metrics')
            result = ledger.inspect_reconciliation(args.run_id, evidence_paths=args.metrics)
    elif args.command == 'prepare_data':
        from .data import prepare_data
        kwargs = {}
        if args.hf_local_shards:
            kwargs['hf_local_shards'] = json.loads(Path(args.hf_local_shards).read_text())
        if args.hf_shard_checksums:
            kwargs['hf_shard_checksums'] = json.loads(Path(args.hf_shard_checksums).read_text())
        result = {'manifest': str(prepare_data(args.out, source_jsonl=args.source_jsonl,
                    hf_dataset=args.hf_dataset, hf_config=args.hf_config,
                    hf_data_files=args.hf_data_files, revision=args.revision,
                    max_tokens=args.max_tokens, max_documents=args.max_documents, **kwargs))}
    elif args.command == 'prepare_mechanisms':
        from .data import prepare_mechanism_tasks
        result = {'manifest': str(prepare_mechanism_tasks(args.out, seed=args.seed,
            train_examples=args.train_examples, validation_examples=args.validation_examples))}
    elif args.command == 'compare_evaluations':
        if args.pairs is not None:
            import yaml
            from .comparison_batch import compare_evaluation_pairs
            pair_file = Path(args.pairs).resolve()
            pairs = yaml.safe_load(pair_file.read_text())
            if isinstance(pairs, list):
                for pair in pairs:
                    if isinstance(pair, dict):
                        for side in ('left', 'right'):
                            if isinstance(pair.get(side), str):
                                pair[side] = str((pair_file.parent / pair[side]).resolve())
            result = compare_evaluation_pairs(pairs, args.out,
                        bootstrap_samples=args.bootstrap_samples, seed=args.seed)
        else:
            from .train import compare_evaluations
            result = compare_evaluations(args.left, args.right, args.out,
                        bootstrap_samples=args.bootstrap_samples, seed=args.seed)
    elif args.command == 'compare_seed_pairs':
        import yaml
        from .replication import compare_seed_pairs
        pair_file = Path(args.pairs).resolve()
        pairs = yaml.safe_load(pair_file.read_text())
        if isinstance(pairs, list):
            for pair in pairs:
                if isinstance(pair, dict):
                    for side in ('left', 'right'):
                        if isinstance(pair.get(side), str):
                            pair[side] = str((pair_file.parent / pair[side]).resolve())
        result = compare_seed_pairs(pairs, args.out, allowed_between_dimensions=args.allow_differences)
    elif args.command == 'summarize':
        result = summarize_runs(args.run_dirs, args.out, max_plot_points=args.max_plot_points)
    elif args.command == 'check_math':
        import pytest
        return pytest.main([str(Path(__file__).resolve().parents[1] / 'tests' / 'test_math.py'),
                            '-v' if args.verbose else '-q'])
    else:
        cfg = Config.from_yaml(args.config)
        if args.command == 'record_spend':
            from .train import SpendLedger
            project = cfg.raw.get('project', {})
            if project.get('dry_run', True):
                result = {'dry_run': True, 'charge_id': args.charge_id, 'usd': args.usd, 'stage': args.stage}
            else:
                ledger = SpendLedger(project.get('spend_ledger', 'runs/spend.json'),
                            project.get('total_budget_usd', 1000), project.get('reserve_usd', 150),
                            stage_budgets=project.get('stage_budgets'))
                result = ledger.record_charge(args.charge_id, args.usd, stage=args.stage, description=args.description)
        elif args.command == 'validate_config':
            result = validate_config(cfg, device=args.device, run_dir=args.run_dir)
        elif args.command == 'plan_search':
            if args.candidates:
                import yaml
                from .search import plan_sweep
                result = plan_sweep(cfg, yaml.safe_load(Path(args.candidates).read_text()))
            else:
                result = plan_search(cfg)
        elif args.command == 'sweep':
            import yaml
            from .sweep import run_sweep
            candidates = yaml.safe_load(Path(args.candidates).read_text())
            if args.canary_steps or args.min_tok_s is not None or args.canary_warmup_steps != 12:
                from .throughput_canary import ThroughputCanary
                ThroughputCanary(args.canary_steps, args.min_tok_s, args.canary_warmup_steps)
            snapshot_preview = None
            if args.snapshot_sources is not None:
                from .search import plan_sweep
                planned = plan_sweep(cfg, candidates)
                policies = [run['config'].get('project', {}).get('dry_run', True) for run in planned['runs']]
                if any(type(value) is not bool for value in policies) or len(set(policies)) != 1:
                    raise ValueError('all candidates must share an explicit Boolean dry_run policy')
                snapshot = Path(args.snapshot_sources or (str(args.out) + '.sources')).absolute()
                snapshot_preview = {'path': str(snapshot), 'created': False,
                                    'scope': 'fresh source/config/grid snapshot; dependencies and data remain external'}
                if not policies[0]:
                    from .source_snapshot import create_snapshot, run_snapshot
                    if Path(args.out).exists() or Path(args.out).is_symlink():
                        raise FileExistsError('automatic source snapshot requires a fresh sweep output; reuse the existing snapshot explicitly for re-entry')
                    if (snapshot.resolve().is_relative_to(Path(args.out).resolve()) or
                            Path(args.out).resolve().is_relative_to(snapshot.resolve())):
                        raise ValueError('source snapshot and sweep output directories must not overlap')
                    source_root = Path(__file__).resolve().parents[1]
                    config_path, candidates_path = Path(args.config).resolve(), Path(args.candidates).resolve()
                    create_snapshot(source_root, snapshot, inputs=[config_path, candidates_path], workdir=Path.cwd())
                    child = ['sweep', '--config', str(config_path), '--candidates', str(candidates_path),
                             '--out', str(Path(args.out).absolute()), '--max-seconds', str(args.max_seconds),
                             '--evaluation-max-seconds', str(args.evaluation_max_seconds),
                             '--max-prediction-tokens', str(args.max_prediction_tokens),
                             '--bootstrap-samples', str(args.bootstrap_samples), '--device', args.device]
                    if args.no_cooldown:
                        child.append('--no-cooldown')
                    if args.reference_candidate is not None:
                        child.extend(['--reference-candidate', args.reference_candidate])
                    if args.prune_completed_intermediates:
                        child.append('--prune-completed-intermediates')
                    if args.canary_steps or args.min_tok_s is not None or args.canary_warmup_steps != 12:
                        child.extend(['--canary-steps', str(args.canary_steps),
                                      '--canary-warmup-steps', str(args.canary_warmup_steps)])
                        if args.min_tok_s is not None:
                            child.extend(['--min-tok-s', str(args.min_tok_s)])
                    return run_snapshot(snapshot, child, workdir=Path.cwd())
            result = run_sweep(cfg, candidates, args.out,
                max_seconds=args.max_seconds, evaluation_max_seconds=args.evaluation_max_seconds,
                device=args.device, cooldown=not args.no_cooldown,
                max_prediction_tokens=args.max_prediction_tokens, bootstrap_samples=args.bootstrap_samples,
                reference_candidate=args.reference_candidate,
                prune_completed_intermediates=args.prune_completed_intermediates,
                **({'canary_steps': args.canary_steps, 'min_tokens_per_second': args.min_tok_s,
                    'canary_warmup_steps': args.canary_warmup_steps}
                   if args.canary_steps or args.min_tok_s is not None or args.canary_warmup_steps != 12 else {}))
            if snapshot_preview is not None:
                result['source_snapshot'] = snapshot_preview
        elif args.command == 'evaluate_mechanisms':
            from .train import evaluate_mechanisms
            result = evaluate_mechanisms(cfg, args.checkpoint, args.task_manifest, args.out,
                max_examples=args.max_examples, max_prompt_tokens=args.max_prompt_tokens,
                max_seconds=args.max_seconds, device=args.device,
                components=tuple(args.component) if args.component else ('0:attn', '0:mlp'),
                run_budget_usd=args.run_budget_usd)
        elif args.command == 'evaluate':
            from .train import evaluate_checkpoint
            result = evaluate_checkpoint(cfg, args.checkpoint, args.out, context=args.context,
                        max_prediction_tokens=args.max_prediction_tokens, max_seconds=args.max_seconds,
                        device=args.device, bootstrap_samples=args.bootstrap_samples,
                        run_budget_usd=args.run_budget_usd)
        elif args.command == 'profile':
            result = profile_run(cfg, args.run_dir, steps=args.steps, warmup_steps=args.warmup_steps,
                                 max_seconds=args.max_seconds, device=args.device,
                                 run_budget_usd=args.run_budget_usd, kernels=args.kernels,
                                 **({'trace_steps': args.trace_steps} if args.trace_steps != 0 else {}))
        elif args.command == 'branch' and args.distributed:
            raise ValueError('batch branch currently supports serial training only')
        elif args.command in ('train', 'cooldown') and args.distributed:
            result = _distributed_run_from_cli(cfg, args)
        elif args.command in ('train', 'cooldown', 'branch'):
            if args.export_max_state_bytes is not None:
                raise ValueError('--export-max-state-bytes is for --distributed; ordinary checkpoints already support evaluation')
            result = train_run(cfg, args.run_dir, max_steps=args.max_steps,
                               max_seconds=args.max_seconds, device=args.device,
                               resume=args.resume, run_budget_usd=args.run_budget_usd,
                               context_fork=args.context_fork,
                               cooldown_steps=getattr(args, 'cooldown_steps', None),
                               cooldown_end_tokens=getattr(args, 'cooldown_end_tokens', None),
                               **({'batch_tokens': args.batch_tokens, 'lr_scale': args.lr_scale} if args.command == 'branch' else {}))
        else:
            result = _run_analysis(cfg, args)
    if result is not None:
        print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
