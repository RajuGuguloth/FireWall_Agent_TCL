"""Smoke tests — run: python -m pytest tests/ -q"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_config_paths_exist():
    import config as cfg
    assert cfg.N_FEATURES == 17
    assert cfg.WINDOW_SIZE == 20
    assert "data" in cfg.RAW_DIR
    assert cfg.DATASET_CSV.endswith("combined_dataset_v5_final.csv")


def test_cascade_runtime_loads():
    from src.inference.cascade_r18 import CascadeRuntime
    rt = CascadeRuntime.load()
    assert "BENIGN" in rt.classes
    assert rt.gate is not None


def test_api_import():
    import api.main  # noqa: F401
