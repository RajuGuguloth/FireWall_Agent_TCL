# Repository layout (R18)

Standard ML-service layout used in industry teams.

```
firewall_ml_project/
├── README.md                 # Quick start
├── config.py                 # Single source of truth (paths, features, thresholds)
├── Dockerfile / docker-compose.yml
├── requirements.txt          # Training environment
├── requirements_api.txt      # API / inference only
│
├── api/                      # HTTP service (FastAPI)
│   ├── main.py
│   ├── alert_store.py
│   ├── tier_metrics.py
│   └── static/               # SOC dashboard
│
├── src/                      # Importable production code
│   ├── inference/
│   │   └── cascade_r18.py    # Tier-1 → Tier-2 ONNX → Tier-3 (shared by API + eval)
│   └── models/
│       └── cnn_gru_v6.py     # CNN-GRU architecture
│
├── scripts/                  # CLI entry points (run from repo root)
│   ├── data/                 # Dataset preparation
│   ├── training/             # Model training
│   ├── export/               # ONNX + Tier-3 export
│   ├── eval/                 # Offline evaluation
│   ├── benchmark/            # Latency benchmarks
│   ├── analysis/             # Plots, thesis figures
│   └── etl/                  # Legacy data pipelines (v4/v5)
│
├── data/
│   ├── raw/                  # CSV datasets, PCAP extracts
│   ├── splits/               # v6_sequences .npy splits
│   └── alerts.db             # SOC audit trail (runtime)
│
├── models/                   # Trained artifacts (.pkl, .pth, .onnx)
├── results/                  # Metrics JSON, benchmark outputs
├── docs/                     # Thesis, interview prep, agent prompts
├── tests/                    # Smoke / unit tests
└── legacy/                   # Old R17/R4 code — do not use in production
```

## Rules

1. **Never duplicate** features, paths, or thresholds — use `config.py`.
2. **Never change cascade logic** outside `src/inference/cascade_r18.py`.
3. **Datasets** live under `data/raw/`; processed splits under `data/splits/`.
4. **Run scripts** from repo root: `python scripts/training/train_cnn_gru_v6.py`

## Production pipeline

```bash
python scripts/data/prepare_v6_sequences.py
python scripts/training/train_tier1_gate_v6.py
python scripts/training/train_cnn_gru_v6.py
python scripts/export/export_tier2_classifier_onnx.py
python scripts/export/export_tier2_embedding_onnx.py
python scripts/export/export_tier3_oneclass.py
python scripts/eval/measure_cascade_flow.py
python scripts/benchmark/benchmark_latency.py
uvicorn api.main:app --host 127.0.0.1 --port 8000
```
