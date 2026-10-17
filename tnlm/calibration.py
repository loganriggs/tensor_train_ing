"""Training-only frozen-prefix calibration of lagged normalization."""
from __future__ import annotations

import time

import torch

from .normalization import lagged_modules


def _check_lagged_divisor(norm):
    divisor = norm.denom(norm.weight)
    if not bool(torch.isfinite(divisor).all()) or not bool((divisor > 0).all()):
        raise FloatingPointError('lagged statistic has a nonfinite/nonpositive compute divisor')


def calibrate_lagged(model, cfg, manifest, device, *, deadline, distributed=False, collective_control=False):
    """Calibrate sites in order, replaying one frozen training-only prefix.

    collective_control=True runs the SAME prefix on every FSDP rank, agreeing
    inputs before forward and coordinating local validation/deadline failures.
    Statistics must stay distributed=False: replicated compute is not new data.
    Requires the public root reset_iter_state API after an intentional early stop.
    Unexpected process/collective failures require job relaunch; local validation
    phases cannot recover a broken process group. No activation recomputation.

    Local mode retains the serial numeric procedure. Both modes restore original
    per-module train/eval flags; errors also restore all small norm buffers and
    collector flags/masks. Parameters are never cloned, gathered, or updated.
    """
    from .data import PackedSampler, batch_hash
    if collective_control:
        if not torch.distributed.is_initialized():
            raise ValueError('collective calibration requires an initialized process group')
        from .distributed_checkpoint import _phase as phase, _agree as agree
    else:
        def phase(label, callback):
            return callback()
        def agree(label, value):
            return None

    def control_preflight():
        if not collective_control:
            return 1
        from torch.distributed.fsdp import FSDPModule
        if distributed:
            raise ValueError('replicated-prefix collective calibration requires distributed=False statistics')
        if not torch.distributed.is_initialized() or not isinstance(model, FSDPModule):
            raise ValueError('collective calibration requires an initialized group and FSDP root')
        if not callable(getattr(model, 'reset_iter_state', None)):
            raise ValueError('collective calibration requires public FSDP root reset_iter_state')
        if any(parameter.grad is not None for parameter in model.parameters()):
            raise ValueError('collective calibration requires cleared parameter gradients')
        return torch.distributed.get_world_size()
    world = phase('calibration control preflight', control_preflight)
    norms = lagged_modules(model)
    names = {id(module): name for name, module in model.named_modules()}
    modes = [(module, module.training) for module in model.modules()]
    snapshots = [(norm, {name: value.detach().clone() for name, value in norm.named_buffers()},
                  norm.collecting, norm._valid) for norm in norms]
    handles = []
    success = False
    class SiteReached(Exception):
        pass
    def stop_at_site(module, inputs, output):
        raise SiteReached()
    def deadline_check():
        if time.monotonic() >= deadline:
            raise TimeoutError('runtime cap during initial calibration/verification')
    def setup():
        budget = int(cfg.raw.get('data', {}).get('calibration_tokens', 100_000))
        if budget <= 0 or not norms:
            raise ValueError('lagged calibration requires positive tokens and lagged norm sites')
        if any(norm.collecting or norm._valid is not None or bool(norm.acc_sum.ne(0)) or bool(norm.acc_count.ne(0)) for norm in norms):
            raise ValueError('calibration requires no pending statistic collection/masks')
        sampler = PackedSampler(manifest, split='calibration', batch_size=1,
                                context=cfg.raw.get('data', {}).get('context', 512), seed=cfg.train.seed)
        return budget, sampler
    def prepare_batch(sampler, remaining):
        batch = sampler.next_batch()
        idx, docs, _ = (tensor.to(device) for tensor in batch)
        if (idx < 0).any() or (idx >= cfg.model.vocab_size).any():
            raise ValueError('calibration token outside real vocabulary')
        valid = docs >= 0
        positions = valid.flatten().nonzero().flatten()
        valid.flatten()[positions[remaining:]] = False
        n = int(valid.sum())
        identity = {'batch_hash': batch_hash(batch), 'valid_mask_hash': batch_hash((valid,)),
                    'selected_tokens': n, 'input_tokens': int((docs >= 0).sum()), 'cursor': sampler.state_dict()}
        return idx, docs, valid, identity
    def remove_handle(handle):
        handle.remove()
        handles.remove(handle)

    try:
        budget, sampler = phase('calibration setup', setup)
        initial_cursor = sampler.state_dict()
        agree('calibration prefix/site identity', {'manifest_hash': sampler.manifest_hash, 'cursor': initial_cursor,
              'budget': budget, 'context': sampler.context, 'rows': sampler.epoch_row_count,
              'sites': [names[id(norm)] for norm in norms], 'distributed_statistics': distributed})
        phase('calibration model eval mode', model.eval)
        prefix_tokens, hashes, site_stats, exposures = None, [], [], 0
        with torch.no_grad():
            for site_index, norm in enumerate(norms):
                phase('calibration prefix reset', lambda: sampler.load_state_dict(initial_cursor))
                seen, site_hashes = 0, []
                def install_hook():
                    handle = norm.register_forward_hook(stop_at_site)
                    handles.append(handle)  # retain even if another rank fails setup
                    return handle
                handle = phase('calibration site hook installation', install_hook)
                try:
                    for row_index in range(sampler.epoch_row_count):
                        if seen >= budget:
                            break
                        phase('calibration deadline', deadline_check)
                        idx, docs, valid, identity = phase('calibration batch validation', lambda: prepare_batch(sampler, budget - seen))
                        phase('calibration pre-forward deadline', deadline_check)
                        agree('calibration batch identity', {'site': site_index, 'row': row_index, **identity})
                        n = identity['selected_tokens']
                        if not n:  # all ranks agreed this is empty; no forward
                            continue
                        def site_forward():
                            norm.collecting = True
                            norm.set_valid_mask(valid)
                            reached = False
                            try:
                                with torch.autocast(device.type, dtype=torch.bfloat16,
                                                    enabled=cfg.train.bf16_compute and device.type == 'cuda'):
                                    model(idx, doc_ids=docs)
                            except SiteReached:
                                reached = True
                            finally:
                                norm.collecting = False
                                norm.set_valid_mask(None)
                                if collective_control:
                                    model.reset_iter_state()
                            if not reached:
                                raise RuntimeError('calibration site was not executed')
                        phase('calibration stopped site forward', site_forward)
                        seen += n
                        exposures += identity['input_tokens']
                        site_hashes.append(identity['batch_hash'])
                finally:
                    phase('calibration site hook removal', lambda: remove_handle(handle))
                def commit_site():
                    if seen == 0:
                        raise ValueError('calibration prefix contains no valid tokens')
                    norm.commit(distributed=distributed)
                    _check_lagged_divisor(norm)
                    return {'site': names[id(norm)], 'tokens': seen, 'mean_sq': float(norm.mean_sq)}
                site = phase('calibration site commit/divisor', commit_site)
                agree('calibration committed site', site)
                if prefix_tokens is None:
                    prefix_tokens, hashes = seen, site_hashes
                def replay_check():
                    if prefix_tokens != seen or hashes != site_hashes:
                        raise RuntimeError('calibration prefix replay changed between sites')
                phase('calibration site replay validation', replay_check)
                site_stats.append(site)
            phase('calibration verification prefix reset', lambda: sampler.load_state_dict(initial_cursor))
            verified, verification_hashes = 0, []
            for row_index in range(sampler.epoch_row_count):
                if verified >= budget:
                    break
                phase('calibration verification deadline', deadline_check)
                idx, docs, valid, identity = phase('calibration verification batch validation', lambda: prepare_batch(sampler, budget - verified))
                phase('calibration verification pre-forward deadline', deadline_check)
                agree('calibration verification batch identity', {'row': row_index, **identity})
                if not identity['selected_tokens']:
                    continue
                def verify_forward():
                    try:
                        with torch.autocast(device.type, dtype=torch.bfloat16,
                                            enabled=cfg.train.bf16_compute and device.type == 'cuda'):
                            return model(idx, doc_ids=docs)
                    finally:
                        if collective_control:
                            model.reset_iter_state()
                logits = phase('calibration verification forward', verify_forward)
                def verify_output():
                    if not torch.isfinite(logits).all():
                        raise FloatingPointError('nonfinite logits after calibration; fixed-scale candidate rejected')
                phase('calibration verification output validation', verify_output)
                del logits
                exposures += identity['input_tokens']
                verified += identity['selected_tokens']
                verification_hashes.append(identity['batch_hash'])
            def final_replay_check():
                if verified != prefix_tokens or verification_hashes != hashes:
                    raise RuntimeError('calibration verification prefix changed')
            phase('calibration final replay validation', final_replay_check)
        result = {'split': 'calibration', 'tokens': prefix_tokens or 0, 'requested_tokens': budget,
                  'forward_token_exposures': exposures, 'passes': len(norms), 'sites': site_stats,
                  'verification_passes': 1, 'verified_prefix_tokens': verified, 'verification_logits_finite': True,
                  'batch_hashes': hashes, 'sampler': sampler.state_dict(),
                  'prefix_replication_world_size': world,
                  'aggregate_forward_token_exposures': exposures * world,
                  'policy': 'sequential sites, frozen upstream constants, repeated identical prefix, no optimizer updates',
                  'scope': 'initialization statistics on recorded prefix; replicated compute is not new data; no guarantee of stability under distribution/context shifts'}
        agree('calibration complete report', result)
        success = True
        return result
    finally:
        # Cleanup is local even after a failed collective; never initiate fresh
        # communication in this finalizer. Small norm buffers alone are restored.
        for handle in handles:
            handle.remove()
        for norm, buffers, collecting, valid in snapshots:
            norm.collecting = collecting
            norm.set_valid_mask(valid)
            if not success:
                with torch.no_grad():
                    current = dict(norm.named_buffers())
                    for name, value in buffers.items():
                        current[name].copy_(value)
        for module, training in modes:
            module.training = training
