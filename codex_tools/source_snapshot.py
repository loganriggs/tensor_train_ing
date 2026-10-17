"""Compatibility entrypoint; authoritative implementation lives in tnlm."""
from pathlib import Path
import sys

# Preserve direct-script usage from any working directory, including python -I.
if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tnlm.source_snapshot import main
    raise SystemExit(main())
else:
    from tnlm import source_snapshot as _implementation
    # A module alias preserves private-helper monkeypatches used by callers and
    # tests, rather than copying function globals into a divergent shim module.
    sys.modules[__name__] = _implementation
