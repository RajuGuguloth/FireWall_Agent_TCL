# Legacy code — do not use for R18 production

| Path | Why legacy |
|------|------------|
| `legacy/main.py` | Mock Tier-1 demo |
| `legacy/decision/` | Old router, WINDOW_SIZE=64 |
| `legacy/tier1/`, `legacy/tier2/` | R17 trainers |
| `legacy/features/` | Old feature extractors |
| `legacy/src_v3/` | Pre-R18 src tree |
| `legacy/export_onnx_r17.py` | 5-class ONNX |
| `scripts/etl/` | v4/v5 data prep only |

**Production:** `uvicorn api.main:app` — see [docs/CODEBASE.md](../docs/CODEBASE.md).
