"""Add project root to sys.path for scripts run as `python scripts/...`."""
import sys
from pathlib import Path


def project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "config.py").is_file():
            return parent
    raise RuntimeError("Could not find project root (config.py)")


def setup_path() -> Path:
    root = project_root()
    root_s = str(root)
    if root_s not in sys.path:
        sys.path.insert(0, root_s)
    return root
