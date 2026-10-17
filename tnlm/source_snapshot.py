"""Freeze and run tnlm sources without changing the shared development package."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def _record(payload):
    return {'sha256': hashlib.sha256(payload).hexdigest(), 'bytes': len(payload)}


def _source_files(root):
    package = root / 'tnlm'
    if package.is_symlink() or not package.is_dir():
        raise ValueError('source must contain a regular tnlm directory')
    result = {}
    for path in sorted(package.rglob('*')):
        if path.is_symlink():
            raise ValueError(f'source symlink forbidden: {path}')
        if '__pycache__' in path.parts or path.suffix == '.pyc':
            continue
        if path.is_file():
            result[path.relative_to(root).as_posix()] = path.read_bytes()
    if 'tnlm/cli.py' not in result or 'tnlm/__init__.py' not in result:
        raise ValueError('source requires tnlm/cli.py and tnlm/__init__.py')
    return result


def create_snapshot(source_root, destination, inputs=(), *, workdir=None):
    root = Path(source_root).resolve()
    destination = Path(destination).absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    if destination.resolve().is_relative_to(root / 'tnlm'):
        raise ValueError('snapshot cannot be inside the source package')
    working_directory = Path(workdir).resolve() if workdir is not None else root
    if not working_directory.is_dir() or working_directory.is_relative_to(destination.resolve()):
        raise ValueError('working directory must exist outside the snapshot')
    payloads = _source_files(root)
    input_paths, input_payloads = {}, {}
    for index, value in enumerate(inputs):
        path = Path(value)
        if not path.is_absolute():
            path = root / path
        if path.is_symlink() or not path.is_file():
            raise ValueError(f'input must be a regular file: {path}')
        path = path.resolve()
        if str(path) in input_paths:
            raise ValueError(f'duplicate input: {path}')
        relative = f'inputs/{index}-{path.name}'
        input_paths[str(path)] = relative
        input_payloads[str(path)] = path.read_bytes()
        payloads[relative] = input_payloads[str(path)]
    manifest = {'version': 1, 'source_root': str(root), 'working_directory': str(working_directory), 'input_paths': input_paths,
                'files': {name: _record(payload) for name, payload in payloads.items()}}
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir()  # exclusive claim; manifest is the completion marker
    try:
        for name, payload in payloads.items():
            path = destination / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
        # Recheck full inventory and bytes after copying to catch concurrent edits.
        if _source_files(root) != {k: v for k, v in payloads.items() if k.startswith('tnlm/')}:
            raise RuntimeError('source changed while creating snapshot')
        if any(Path(name).is_symlink() or Path(name).read_bytes() != payload
               for name, payload in input_payloads.items()):
            raise RuntimeError('input changed while creating snapshot')
        temporary = destination / '.manifest.json.tmp'
        with temporary.open('x') as stream:
            json.dump(manifest, stream, sort_keys=True, indent=2)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        temporary.rename(destination / 'manifest.json')
        for path in sorted(destination.rglob('*'), reverse=True):
            path.chmod(0o555 if path.is_dir() else 0o444)
        destination.chmod(0o555)
        for directory in [p for p in destination.rglob('*') if p.is_dir()] + [destination, destination.parent]:
            fd = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        verify_snapshot(destination)
    except BaseException:
        destination.chmod(0o755)
        for path in destination.rglob('*'):
            if path.is_dir() and not path.is_symlink():
                path.chmod(0o755)
        shutil.rmtree(destination)
        raise
    return manifest


def verify_snapshot(snapshot):
    root = Path(snapshot)
    if root.is_symlink():
        raise ValueError('snapshot root must not be a symlink')
    root = root.resolve()
    if (root / 'manifest.json').is_symlink():
        raise ValueError('snapshot manifest must not be a symlink')
    manifest = json.loads((root / 'manifest.json').read_text())
    if (manifest.get('version') != 1 or not isinstance(manifest.get('files'), dict)
            or not isinstance(manifest.get('input_paths'), dict)
            or not isinstance(manifest.get('source_root'), str)):
        raise ValueError('invalid snapshot manifest')
    if 'working_directory' in manifest and (not isinstance(manifest['working_directory'], str)
            or not Path(manifest['working_directory']).is_absolute()):
        raise ValueError('snapshot working directory must be an absolute path')
    expected = manifest['files']
    actual = {}
    for path in root.rglob('*'):
        if path.is_symlink():
            raise ValueError('snapshot symlink forbidden')
        if path.is_file() and path != root / 'manifest.json':
            actual[path.relative_to(root).as_posix()] = _record(path.read_bytes())
    if actual != expected or not {'tnlm/__init__.py', 'tnlm/cli.py'} <= actual.keys():
        raise ValueError('snapshot file inventory or checksum mismatch')
    for original, relative in manifest['input_paths'].items():
        if (not Path(original).is_absolute() or relative not in actual
                or not relative.startswith('inputs/')):
            raise ValueError('invalid snapshot input mapping')
    return manifest


def run_snapshot(snapshot, args, *, workdir=None):
    manifest = verify_snapshot(snapshot)
    root = Path(snapshot).resolve()
    cwd = Path(workdir if workdir is not None else manifest.get('working_directory', manifest['source_root'])).resolve()
    if cwd.is_relative_to(root):
        raise ValueError('working directory must be outside the read-only snapshot')
    arguments = list(args)
    if not arguments:
        raise ValueError('supply a tnlm CLI command')
    captured = manifest['input_paths']
    snapshot_inputs = {str(root / name) for name in captured.values()}
    def pinned(value):
        path = Path(value)
        path = (cwd / path).resolve() if not path.is_absolute() else path.resolve()
        if str(path) in captured:
            return str(root / captured[str(path)])
        if str(path) in snapshot_inputs:
            return str(path)
        raise ValueError(f'config/candidates input was not captured: {value}')
    index = 0
    while index < len(arguments):
        item = arguments[index]
        flag, equals, value = item.partition('=')
        if (flag.startswith('--') and flag not in ('--config', '--candidates')
                and any(name.startswith(flag) for name in ('--config', '--candidates'))):
            raise ValueError('spell --config and --candidates in full to pin their inputs')
        if flag in ('--config', '--candidates'):
            if equals:
                arguments[index] = flag + '=' + pinned(value)
            else:
                if index + 1 == len(arguments):
                    raise ValueError(f'missing value for {flag}')
                index += 1
                arguments[index] = pinned(arguments[index])
        index += 1
    # -I ignores PYTHONPATH and the current directory; explicit package insertion
    # takes precedence over editable installations. -B prevents cache writes.
    bootstrap = ('import runpy,sys; sys.path.insert(0,sys.argv.pop(1)); '
                 'sys.argv[0]="tnlm"; runpy.run_module("tnlm.cli",run_name="__main__")')
    return subprocess.run([sys.executable, '-I', '-B', '-c', bootstrap,
                           str(root), *arguments], cwd=cwd, check=False).returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    create = commands.add_parser('create')
    create.add_argument('--source', required=True)
    create.add_argument('--out', required=True)
    create.add_argument('--input', action='append', default=[])
    create.add_argument('--workdir', help='record the working directory for later runs; defaults to source root')
    verify = commands.add_parser('verify')
    verify.add_argument('snapshot')
    run = commands.add_parser('run')
    run.add_argument('--snapshot', required=True)
    run.add_argument('--workdir')
    run.add_argument('args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command == 'create':
        print(json.dumps(create_snapshot(args.source, args.out, args.input, workdir=args.workdir), indent=2))
    elif args.command == 'verify':
        manifest = verify_snapshot(args.snapshot)
        print(json.dumps({'verified': True, 'files': len(manifest['files'])}))
    else:
        sys.exit(run_snapshot(args.snapshot, args.args[1:] if args.args[:1] == ['--'] else args.args,
                              workdir=args.workdir))


if __name__ == '__main__':
    main()
