"""Advisory sequential sweep retention arithmetic and narrow payload selection."""
from pathlib import PurePosixPath


def intermediate_payload(name):
    path = PurePosixPath(name)
    return (not path.is_absolute() and len(path.parts) == 2 and path.parts[1] == 'state.pt'
            and any(path.parts[0].startswith(prefix) and len(path.parts[0]) > len(prefix)
                    for prefix in ('milestone-', 'recovery-')))


def retention_forecast(trajectories, completed_stages=()):
    """Additional bytes only; never credit future reclamation of existing files."""
    complete = set(completed_stages)
    retained = peak = 0
    candidates = []
    for run in trajectories:
        active = permanent = 0
        for phase, key in [('parent', 'parent_storage_plan'), ('cooldown', 'cooldown_storage_plan')]:
            plan = run[key]
            if plan is not None and run['name'] + '/' + phase not in complete:
                active += plan['required_additional_bytes']
                counts = plan.get('count_breakdown', {'initial': 1, 'final': 1})
                permanent += (counts['initial'] + counts['final']) * plan['per_checkpoint_estimate_bytes']
        candidate_peak = retained + active
        peak = max(peak, candidate_peak)
        candidates.append({'name': run['name'], 'new_prior_retained_bytes': retained,
                           'new_active_full_peak_bytes': active, 'new_candidate_peak_bytes': candidate_peak,
                           'new_retained_initial_final_bytes': permanent})
        retained += permanent
    return {'required_additional_bytes': peak, 'new_final_retained_bytes': retained,
            'candidates': candidates,
            'scope': 'advisory max of cumulative newly retained initial/final estimates plus active candidate full peaks; already-completed stages excluded; no existing reclamation credit; non-checkpoint metadata allowance/exclusions inherit checkpoint plans'}


def reject_symlink_chain(root, path):
    """Reject redirects at the stage root and each ancestor below anchored out."""
    from pathlib import Path
    root, path = Path(root), Path(path)
    relative = path.relative_to(root)
    current = root
    if current.is_symlink():
        raise ValueError('sweep root is a symlink')
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise ValueError('prune/verification path contains a symlink')


def unlink_payload(root, relative, expected):
    """Hash/unlink through directory FDs with no-follow traversal; fsync deletion.

    The sweep and ledger locks coordinate cooperating processes. This is not a
    guarantee against privileged hostile concurrent inode/name replacement.
    """
    import hashlib
    import os
    import stat
    parts = PurePosixPath(relative).parts
    if PurePosixPath(relative).is_absolute() or not parts or any(part in ('.', '..') for part in parts):
        raise ValueError('invalid prune path')
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    directory = os.open(root, flags)
    try:
        for part in parts[:-1]:
            next_directory = os.open(part, flags, dir_fd=directory)
            os.close(directory)
            directory = next_directory
        try:
            descriptor = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory)
        except FileNotFoundError:
            return False  # Durable planned operation may have already unlinked it.
        with os.fdopen(descriptor, 'rb') as stream:
            info = os.fstat(stream.fileno())
            digest = hashlib.sha256()
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(chunk)
        if not stat.S_ISREG(info.st_mode) or info.st_size != expected['bytes'] or digest.hexdigest() != expected['sha256']:
            raise ValueError('slated prune payload changed before unlink')
        current = os.stat(parts[-1], dir_fd=directory, follow_symlinks=False)
        if (current.st_dev, current.st_ino) != (info.st_dev, info.st_ino):
            raise ValueError('prune payload inode changed before unlink')
        os.unlink(parts[-1], dir_fd=directory)
        os.fsync(directory)
        return True
    finally:
        os.close(directory)
