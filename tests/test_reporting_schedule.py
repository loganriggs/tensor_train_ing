import json

from tnlm.reporting import summarize_runs
from tnlm.train import train_run


def test_summary_distinguishes_partial_parent_and_complete_cooldown(config, tmp_path):
    parent_dir = tmp_path / 'parent'
    cooled_dir = tmp_path / 'cooled'
    parent = train_run(config, parent_dir, max_steps=2, max_seconds=60, device='cpu')
    train_run(config, cooled_dir, max_steps=2, max_seconds=60, device='cpu',
              resume=parent['checkpoint'], cooldown_steps=2, cooldown_end_tokens=25)
    out = tmp_path / 'summary'
    summarize_runs([parent_dir, cooled_dir], out)
    summary = json.loads((out / 'comparison.json').read_text())
    before, after = summary['runs']
    assert (before['cooldown_status'], before['completion_status']) == ('uncooled', 'partial')
    assert before['segment_target_prediction_tokens'] == 100
    assert (after['cooldown_status'], after['completion_status']) == ('cooled', 'complete')
    assert after['segment_target_prediction_tokens'] == 25
    assert after['run_prediction_tokens'] == 11  # parent tokens are not counted twice
    assert after['global_prediction_tokens'] == 25
    assert not summary['automatic_ranking_eligible']
    # Legacy artifact is valid input but lacks authoritative schedule evidence.
    result_path = parent_dir / 'result.json'
    result = json.loads(result_path.read_text())
    result.pop('training_descriptor')
    result_path.write_text(json.dumps(result))
    legacy = tmp_path / 'legacy-summary'
    summarize_runs([parent_dir], legacy)
    row = json.loads((legacy / 'comparison.json').read_text())['runs'][0]
    assert row['cooldown_status'] == row['completion_status'] == 'unknown'
    assert row['segment_target_prediction_tokens'] is None
