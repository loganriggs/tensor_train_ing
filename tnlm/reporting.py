"""Bounded summaries and standalone plots of training run segments."""
from __future__ import annotations

import json
import math
from pathlib import Path


def summarize_runs(run_dirs, out_dir, *, max_plot_points=2000):
    """Export cost/quality metadata and training-loss curves for measured runs.

    Curves are segment-local for elapsed time/cost; global scored-token positions
    include inherited training. The table keeps parent identity explicit. Training
    CE is not held-out quality, and P/N audits do not establish interpretability.
    """
    import csv
    if max_plot_points < 2:
        raise ValueError('max_plot_points must be at least two')
    table, curves = [], []
    for directory in run_dirs:
        directory = Path(directory).resolve()
        result = json.loads((directory / 'result.json').read_text())
        ledger = json.loads(Path(result['spend_ledger']).read_text())
        billing = ledger['runs'][result['run_id']]
        if billing['status'] != 'complete':
            raise ValueError(f'cannot summarize unfinished run: {directory}')
        metrics = directory / 'metrics.jsonl'
        count = 0
        if metrics.exists():
            with metrics.open() as f:
                count = sum(1 for _ in f)
        stride = max(1, math.ceil(count / max_plot_points))
        points, first, last = [], None, None
        segment_tokens, total_step_seconds = 0, 0.0
        if metrics.exists():
            with metrics.open() as f:
                for index, line in enumerate(f):
                    record = json.loads(line)
                    if first is None:
                        first = record
                    last = record
                    segment_tokens += record['prediction_tokens']
                    total_step_seconds += record['seconds']
                    if index % stride == 0 or index == count - 1:
                        points.append({key: record[key] for key in
                            ('step', 'tokens', 'loss', 'run_elapsed_seconds', 'run_billed_usd')})
        progress = result['progress']
        descriptor = result.get('training_descriptor', {})
        if not isinstance(descriptor, dict) or descriptor.get('source') != 'checkpoint_payload' or descriptor.get('version') != 1:
            descriptor = {}
        table.append({'run': str(directory), 'operator_class': result['operator_audit']['class'],
            'normalization': result['model_config']['normalization'],
            'optimizer': result.get('train_config', {}).get('optimizer'),
            'configured_lr': result.get('train_config', {}).get('lr'),
            'seed': descriptor.get('seed'),
            'width': descriptor.get('width'), 'depth': descriptor.get('depth'),
            'schedule': descriptor.get('schedule', 'unknown'),
            'cooldown_status': descriptor.get('cooldown_status', 'unknown'),
            'completion_status': descriptor.get('completion_status', 'unknown'),
            'segment_target_prediction_tokens': descriptor.get('segment_target_prediction_tokens'),
            'cooldown_steps': descriptor.get('cooldown_steps'),
            'aux_adamw_lr': result.get('train_config', {}).get('aux_adamw_lr'),
            'attention': result['model_config']['attention'],
            'parameters': result['parameters']['total'], 'context': result['data']['context'],
            'context_history': descriptor.get('context_history'),
            'context_history_status': descriptor.get('context_history_status', 'unknown'),
            'data_manifest_sha256': result['data']['manifest_sha256'],
            'global_prediction_tokens': progress['tokens'], 'run_prediction_tokens': segment_tokens,
            'trajectory_calibration_tokens': progress.get('calibration', {}).get('tokens', 0),
            'run_calibration_tokens': 0 if result.get('parent') else progress.get('calibration', {}).get('tokens', 0),
            'parent_checkpoint': result.get('parent'), 'final_training_ce': last['loss'] if last else None,
            'run_billed_usd': billing['spent_usd'], 'run_elapsed_seconds': billing['elapsed_seconds'],
            'step_average_prediction_tokens_per_second': segment_tokens / total_step_seconds if total_step_seconds else None,
            'cuda_peak_allocated_bytes': result['cuda_peak_allocated_bytes'],
            'cuda_peak_reserved_bytes': result['cuda_peak_reserved_bytes'],
            'device': result['runtime']['device'], 'compile_enabled': result['compile_enabled'],
            'warmup_dominates_completed_segment': result['warmup_dominates_completed_segment']})
        curves.append({'run': str(directory), 'label': directory.name, 'points': points,
                       'original_metric_points': count})
    if not table:
        raise ValueError('at least one completed run is required')
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=False)
    bundle = {'runs': table, 'curves': curves,
              'metric': 'training cross entropy; not held-out model quality',
              'context_scope': 'context is the current segment; context_history records known earlier segments; missing legacy history is unknown',
              'schedule_scope': 'cooldown and completion labels use checkpoint-equivalent training evidence; legacy results remain unknown; mixed schedules/horizons are not automatically comparable',
              'automatic_ranking_eligible': False,
              'step_timing_scope': 'all recorded steps including initial compile; use profile for declared warmup exclusion',
              'cost_scope': 'per run segment, includes setup/compilation/checkpoints; parent cost not silently combined',
              'interpretation_scope': 'operator classification only; use analyze for measured norm sensitivity and weight diagnostics'}
    (out / 'comparison.json').write_text(json.dumps(bundle, indent=2, allow_nan=False))
    with (out / 'comparison.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for field, label, filename in (
        ('tokens', 'Global scored prediction tokens', 'loss-vs-tokens.svg'),
        ('run_elapsed_seconds', 'Run elapsed seconds (includes setup)', 'loss-vs-time.svg'),
        ('run_billed_usd', 'Run billed USD', 'loss-vs-cost.svg')):
        fig, ax = plt.subplots(figsize=(6, 4))
        for curve in curves:
            points = curve['points']
            if points:
                ax.plot([p[field] for p in points], [p['loss'] for p in points], label=curve['label'])
        ax.set(xlabel=label, ylabel='Training cross entropy')
        ax.grid(alpha=.25)
        if any(c['points'] for c in curves):
            ax.legend()
        fig.tight_layout()
        fig.savefig(out / filename)
        plt.close(fig)
    return {'output': str(out), 'runs': len(table), 'artifacts': sorted(p.name for p in out.iterdir()),
            'metric': bundle['metric'], 'cost_scope': bundle['cost_scope']}

