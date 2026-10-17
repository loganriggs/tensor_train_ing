"""Export recorded sweep configurations without loading models or current defaults."""
import hashlib
import json
from pathlib import Path

import yaml


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def export_sweep_config(sweep, candidate):
    """Return directly usable YAML with provenance comments, preserving saved values."""
    path = Path(sweep) / 'sweep.json'
    raw = path.read_bytes()
    journal = json.loads(raw)
    identity = journal.get('identity')
    if not isinstance(identity, dict) or identity.get('version') != 1:
        raise ValueError('unsupported or missing sweep identity')
    if _digest(identity) != journal.get('identity_sha256'):
        raise ValueError('sweep identity checksum mismatch')
    trajectories = identity.get('trajectories')
    if not isinstance(trajectories, list) or not all(isinstance(row, dict) for row in trajectories):
        raise ValueError('invalid sweep trajectories')
    selected = [row for row in trajectories if row.get('name') == candidate]
    if len(selected) != 1:
        raise ValueError('candidate must match exactly one recorded trajectory')
    row = selected[0]
    config = row.get('config')
    if not isinstance(config, dict) or not all(isinstance(config.get(k), dict) for k in ('model', 'train')):
        raise ValueError('invalid recorded candidate configuration')
    if _digest(config) != row.get('config_sha256'):
        raise ValueError('candidate config checksum mismatch')
    provenance = {'source': str(path.resolve()), 'source_sha256': hashlib.sha256(raw).hexdigest(),
                  'candidate': candidate, 'identity_sha256': journal['identity_sha256'],
                  'config_sha256': row['config_sha256']}
    return ('# Exact recorded sweep config; current schema defaults are not injected.\n'
            '# This export does not certify checkpoint completion.\n'
            '# provenance: ' + json.dumps(provenance, sort_keys=True) + '\n'
            + yaml.safe_dump(config, sort_keys=False))
