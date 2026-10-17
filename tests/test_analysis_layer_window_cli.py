import json
from pathlib import Path
import pytest
from tnlm.cli import _run_analysis, main
from test_analysis_probe_provenance import _fixture, _diagnostic

@pytest.mark.parametrize('window', [0, 4])
def test_window_forwarded_and_recorded(tmp_path, monkeypatch, window):
    cfg, args, _, _ = _fixture(tmp_path)
    args.rw_layer_window = window
    calls = []
    def analyze(model, batches, out, *, rw_layer_window):
        calls.append(rw_layer_window)
        return {}
    _diagnostic(monkeypatch, analyze)
    result = _run_analysis(cfg, args)
    assert calls == [window]
    assert result['evaluation_provenance']['rw_layer_window'] == window
    assert json.loads((Path(args.out)/'invocation.json').read_text())['rw_layer_window'] == window

@pytest.mark.parametrize('window', [None, 0, 4])
def test_cli_dry_run_reports_window(tmp_path, capsys, window):
    config = tmp_path/'config.yaml';config.write_text('project:\n  dry_run: true\n')
    argv = ['analyze', '--config', str(config), '--checkpoint', 'absent', '--out', str(tmp_path/'out')]
    if window is not None:argv += ['--rw-layer-window', str(window)]
    main(argv)
    assert json.loads(capsys.readouterr().out)['rw_layer_window'] == window
    assert not (tmp_path/'out').exists()


def test_negative_window_rejected_before_work(tmp_path):
    config = tmp_path/'config.yaml';config.write_text('project:\n  dry_run: true\n')
    with pytest.raises(ValueError, match='nonnegative integer'):
        main(['analyze', '--config', str(config), '--checkpoint', 'absent', '--out', str(tmp_path/'out'),
              '--rw-layer-window', '-1'])
    assert not (tmp_path/'out').exists()
