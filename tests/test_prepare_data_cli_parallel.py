import json
import subprocess
import sys


def test_cli_parallel_matches_serial(tmp_path):
    source = tmp_path / 'source.jsonl'
    source.write_text(''.join(json.dumps({'id': str(i), 'text': 'Hello world!\r\n🙂' * i}) + '\n' for i in range(30)))
    outputs = [tmp_path / 'serial', tmp_path / 'parallel']
    for workers, output in zip((1, 2), outputs):
        subprocess.run([sys.executable, '-m', 'tnlm.cli', 'prepare_data', '--source-jsonl', str(source),
                        '--revision', 'fixture-v1', '--out', str(output), '--workers', str(workers),
                        '--tokenization-batch-bytes', '64'], check=True, capture_output=True, timeout=60)
    assert {p.relative_to(outputs[0]): p.read_bytes() for p in outputs[0].rglob('*') if p.is_file()} == {p.relative_to(outputs[1]): p.read_bytes() for p in outputs[1].rglob('*') if p.is_file()}
