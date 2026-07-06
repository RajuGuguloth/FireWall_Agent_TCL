# Hybrid-Sentinel — NDN AI Firewall (R18)

**3-tier ML intrusion detection** for network packet windows: **ALLOW / FLAG / BLOCK** with per-tier explainability.

This README is the **project map**. Start here to find files and run the system on a new machine.

---

## Table of contents

1. [What this project does](#what-this-project-does)
2. [Company submission checklist](#company-submission-checklist)
3. [Prerequisites](#prerequisites)
4. [First-time setup](#first-time-setup)
5. [Two ways to run](#two-ways-to-run)
6. [Folder map — where everything lives](#folder-map--where-everything-lives)
7. [Key files quick reference](#key-files-quick-reference)
8. [Full training pipeline](#full-training-pipeline)
9. [Run the API & dashboard](#run-the-api--dashboard)
10. [Docker](#docker)
11. [Verify it works](#verify-it-works)
12. [Results & artifacts](#results--artifacts)
13. [How to package for submission](#how-to-package-for-submission)
14. [Troubleshooting](#troubleshooting)
15. [More documentation](#more-documentation)

---

## What this project does

```
Packet window (20 × 17 features)
        │
        ▼
  Tier-1  Random Forest gate     → fast ALLOW if clearly benign
        │
        ▼
  Tier-2  CNN-GRU (ONNX)         → 6-class attack typing + confidence
        │
        ▼
  Tier-3  Mahalanobis one-class  → zero-day FLAG on novel embeddings
        │
        ▼
  ALLOW / FLAG / BLOCK  +  tier_trace audit log
```

**Attack classes:** BENIGN, BRUTE_FORCE, DDOS_HTTP_FLOOD, SLOW_HTTP, PORT_SCAN, DNS_TUNNELING

---

## Company submission checklist

This repository is prepared for the final internship archival request. Use the table below to verify that each requested item is present before submitting the package.

| Company requirement | Where it is covered |
|---------------------|---------------------|
| Final source code | Full repository, packaged by `scripts/package_submission.sh` |
| Well documented report | `docs/FINAL_SUBMISSION_REPORT.md`, `docs/IEEE_REPORT.md`, `Hybrid_Sentinel__A_Three_Tier_Confidence_GatedAI_Firewall_for_NDN_Edge_Deployment.pdf` |
| Training approach | [Full training pipeline](#full-training-pipeline), `docs/DESIGN_AND_IMPLEMENTATION.md` |
| Performance metrics | `results/submission_metrics_r18.json`, `results/r18_tier_metrics.json`, `results/r18_latency_benchmark.json` |
| Evaluation results | `scripts/eval/measure_cascade_flow.py`, `api/tier_metrics.py`, `results/submission_figures/` |
| Overall setup documentation | [First-time setup](#first-time-setup), [Run the API & dashboard](#run-the-api--dashboard), [Docker](#docker) |
| Design notes and implementation details | `docs/DESIGN_AND_IMPLEMENTATION.md`, `docs/CODEBASE.md`, `src/inference/cascade_r18.py` |
| Runnable demo / dashboard | FastAPI app in `api/main.py`; open `http://127.0.0.1:8000/` after server start |

For the exact archive steps, see [How to package for submission](#how-to-package-for-submission) and `docs/SUBMISSION_PACKAGE.md`.

---

## Prerequisites

| Requirement | Notes |
|-------------|--------|
| **Python 3.11** | Tested on 3.11.x |
| **OS** | macOS or Linux (Windows may work with WSL) |
| **RAM** | 8 GB minimum; 16 GB recommended for training |
| **Disk** | ~2 GB for venv + models + data splits |
| **Git** | To clone the repository |

You also need the **dataset and model artifacts** (see [Required artifacts](#required-artifacts-before-running-the-api)).

---

## First-time setup

Clone the repo and open a terminal **in the project root** (`firewall_ml_project/`).

```bash
# 1. Clone (replace with your repo URL)
git clone <your-repo-url> firewall_ml_project
cd firewall_ml_project

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt       # training + eval (full project)
# OR for API only:
# pip install -r requirements_api.txt
```

> **Important:** All commands below are run from the **repository root**, not from inside `scripts/`.

---

## Two ways to run

| Goal | You need | Commands |
|------|----------|----------|
| **A. Demo / API only** | Pre-trained `models/` + `data/splits/v6_sequences/` | [Run the API](#run-the-api--dashboard) |
| **B. Train everything** | Raw CSV in `data/raw/` | [Full training pipeline](#full-training-pipeline-from-scratch) |

---

## Folder map — where everything lives

```
firewall_ml_project/
│
├── README.md                 ← YOU ARE HERE (project map + run guide)
├── config.py                 ← Paths, features, thresholds (edit nothing else for paths)
├── requirements.txt          ← Python deps for training
├── requirements_api.txt      ← Python deps for API only (lighter)
├── Dockerfile
├── docker-compose.yml
│
├── api/                      ← PRODUCTION HTTP SERVICE
│   ├── main.py               ← FastAPI app (start with uvicorn)
│   ├── alert_store.py        ← SQLite alert persistence
│   ├── tier_metrics.py       ← Offline metrics for dashboard
│   └── static/
│       ├── dashboard.html    ← SOC dashboard UI
│       └── flow3d.js         ← 3D attack-flow visualization
│
├── src/                      ← CORE PYTHON LIBRARY (imported by API & scripts)
│   ├── inference/
│   │   └── cascade_r18.py    ← Single cascade implementation (Tier 1→2→3)
│   └── models/
│       └── cnn_gru_v6.py       ← CNN-GRU neural network architecture
│
├── scripts/                  ← RUNNABLE CLI SCRIPTS (run from repo root)
│   ├── data/
│   │   └── prepare_v6_sequences.py    ← Build R18 train/val/test .npy splits
│   ├── training/
│   │   ├── train_tier1_gate_v6.py     ← Tier-1 BENIGN vs ATTACK gate
│   │   ├── train_cnn_gru_v6.py        ← Tier-2 6-class CNN-GRU
│   │   └── train_tier3_gnn_v6.py      ← Tier-3 GNN (research only)
│   ├── export/
│   │   ├── export_tier2_classifier_onnx.py
│   │   ├── export_tier2_embedding_onnx.py
│   │   └── export_tier3_oneclass.py
│   ├── eval/
│   │   ├── measure_cascade_flow.py    ← Ground-truth 3-tier eval (use for metrics)
│   │   └── eval_pipeline_v6.py        ← Tier-1+2 only (quick check)
│   ├── benchmark/
│   │   └── benchmark_latency.py
│   ├── analysis/                      ← Thesis plots & figures
│   └── etl/                           ← Old v4/v5 data scripts (legacy)
│
├── data/
│   ├── raw/
│   │   └── combined_dataset_v5_final.csv   ← Main training CSV (required for training)
│   ├── splits/
│   │   └── v6_sequences/             ← X_train.npy, y_test.npy, etc.
│   └── alerts.db                     ← Created at runtime by API
│
├── models/                           ← TRAINED ARTIFACTS (required for API)
│   ├── tier1_gate_v6.pkl
│   ├── tier2_cnn_gru_v1_r18.pth
│   ├── tier2_r18_temperature.json
│   ├── tier3_oneclass_v6.pkl
│   ├── serialized/
│   │   ├── v6_scaler.pkl
│   │   └── v6_encoder.pkl
│   └── onnx/
│       ├── tier2_cnn_gru_r18.onnx
│       └── tier2_embedding_r18.onnx
│
├── results/                          ← Metrics & benchmark JSON outputs
│   ├── r18_tier_metrics.json
│   └── r18_latency_benchmark.json
│
├── docs/                             ← Documentation
│   ├── CODEBASE.md                   ← Architecture rules for developers
│   ├── DESIGN_AND_IMPLEMENTATION.md  ← Design notes & implementation details
│   ├── FINAL_SUBMISSION_REPORT.md    ← Final handover report
│   ├── SUBMISSION_PACKAGE.md         ← Packaging checklist
│   ├── IEEE_REPORT.md                ← IEEE report notes
│   └── thesis.pdf                    ← Thesis/report PDF
│
├── tests/
│   └── test_smoke.py                 ← Basic import / load tests
│
└── legacy/                           ← OLD R17/R4 CODE — do not use for R18
    ├── README.md
    ├── tier1/, tier2/, decision/
    └── src_v3/
```

### Folders you can ignore (research / old work)

| Folder | Purpose |
|--------|---------|
| `legacy/` | Old trainers and broken R17 pipeline |
| `scripts/etl/` | v4/v5 data preparation only |
| `data/splits/v4_*`, `v5_*` | Old sequence splits |
| `attack_lab/`, `attack_captures/` | PCAP capture lab (optional) |
| `deployment/` | Cloud deploy stubs (optional) |

---

## Key files quick reference

| What you need | File path |
|---------------|-----------|
| All paths & thresholds | `config.py` |
| Cascade inference logic | `src/inference/cascade_r18.py` |
| API entry point | `api/main.py` |
| Dashboard UI | `api/static/dashboard.html` |
| Raw dataset | `data/raw/combined_dataset_v5_final.csv` |
| Test sequences | `data/splits/v6_sequences/X_test.npy` |
| Tier-1 model | `models/tier1_gate_v6.pkl` |
| Tier-2 PyTorch weights | `models/tier2_cnn_gru_v1_r18.pth` |
| Tier-2 ONNX (production) | `models/onnx/tier2_cnn_gru_r18.onnx` |
| Tier-3 detector | `models/tier3_oneclass_v6.pkl` |
| Full metrics JSON | `results/r18_tier_metrics.json` |

---

## Full training pipeline

Use this when you have the raw CSV and want to rebuild the R18 split, retrain the models, export ONNX, and regenerate metrics on a new machine.

### Step 0 — Place required training artifacts

Place the raw CSV here (filename must match `config.py`):

```
data/raw/combined_dataset_v5_final.csv          # raw source CSV, if policy allows sharing
```

If only the API demo is required, the raw CSV is not needed; the pre-trained models and `data/splits/v6_sequences/X_test.npy` are enough.

### Step 1 — Install & activate

```bash
cd firewall_ml_project
source .venv/bin/activate
pip install -r requirements.txt
```

### Step 2 — Run pipeline (in order)

```bash
# Build the group-disjoint R18 sequence split and shared scaler/encoder
python scripts/data/prepare_v6_sequences.py

# Train models from the prepared v6 sequence split
python scripts/training/train_tier1_gate_v6.py
python scripts/training/train_cnn_gru_v6.py

# Export for production inference
python scripts/export/export_tier2_classifier_onnx.py
python scripts/export/export_tier2_embedding_onnx.py
python scripts/export/export_tier3_oneclass.py

# Evaluate & benchmark
python scripts/eval/measure_cascade_flow.py
python scripts/benchmark/benchmark_latency.py
python scripts/analysis/generate_submission_metrics_r18.py
```

Expected outputs land in `models/`, `data/splits/v6_sequences/`, and `results/`.

---

## Run the API & dashboard

### Required artifacts before running the API

The API will **not start** unless these files exist:

```
models/serialized/v6_scaler.pkl
models/serialized/v6_encoder.pkl
models/tier1_gate_v6.pkl
models/tier2_cnn_gru_v1_r18.pth
models/tier2_r18_temperature.json
models/onnx/tier2_cnn_gru_r18.onnx
models/onnx/tier2_embedding_r18.onnx    # required if tier3_oneclass exists
models/tier3_oneclass_v6.pkl          # optional but recommended
data/splits/v6_sequences/X_test.npy   # for /demo/traffic only
```

### Start the server

```bash
cd firewall_ml_project
source .venv/bin/activate
pip install -r requirements_api.txt

uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Leave this terminal running while you view the dashboard.

### Open in browser

| URL | What it is |
|-----|------------|
| http://127.0.0.1:8000/ | SOC dashboard (live session, metrics, 3D flow, alerts) |
| http://127.0.0.1:8000/dashboard | Same dashboard route, useful if `/` is cached |
| http://127.0.0.1:8000/docs | Swagger API docs |
| http://127.0.0.1:8000/health | Health check JSON |

### How reviewers can see live dashboard activity

1. Start the server with the `uvicorn` command above.
2. Open `http://127.0.0.1:8000/` in Chrome/Edge/Safari.
3. In a second terminal, fire demo traffic:

```bash
curl -X POST "http://127.0.0.1:8000/demo/traffic?n=20"
```

4. Refresh or watch the dashboard. The alert timeline, counts, tier traces, latency, and attack labels are populated from the API and SQLite alert store.
5. Open `http://127.0.0.1:8000/alerts?limit=20` to inspect the raw alert records.

### Quick API tests (new terminal)

```bash
# Health
curl http://127.0.0.1:8000/health

# Offline tier metrics (cached JSON)
curl http://127.0.0.1:8000/metrics/tiers

# Fire real test sequences through the cascade
curl -X POST "http://127.0.0.1:8000/demo/traffic?n=5"

# Session stats
curl http://127.0.0.1:8000/stats
```

### Predict on custom traffic

Send a JSON window of **20 packets × 17 features** per request:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sequence": [[...17 floats...], ...20 packets...]}'
```

Response includes `action`, `label`, `tier_trace`, and `latency_ms`.

---

## Docker

Requires `models/` and `data/` on the host (they are gitignored — copy or train first).

```bash
cd firewall_ml_project
docker compose up --build
```

API will be at http://127.0.0.1:8000/

---

## Verify it works

> Requires trained `models/` artifacts (see [Required artifacts](#required-artifacts-before-running-the-api)).

```bash
source .venv/bin/activate

# Smoke tests (imports + model load)
python -c "
from tests.test_smoke import test_config_paths_exist, test_cascade_runtime_loads, test_api_import
test_config_paths_exist(); test_cascade_runtime_loads(); test_api_import()
print('OK')
"

# Or with pytest
pip install pytest
python -m pytest tests/ -q
```

---

## Results & artifacts

Held-out test set: **14,219 sequences** (814 benign, 13,405 attack)

| Metric | Value |
|--------|-------|
| Tier-2 macro-F1 | 0.9849 |
| Full cascade attack detection | 100% |
| Benign false-positive rate | 1.11% |
| Full cascade latency (CPU) | 4.19 ms |
| Full cascade throughput | 238 sequences/s |

Detailed metrics: `results/r18_tier_metrics.json`  
Latency benchmark: `results/r18_latency_benchmark.json`

---

## How to package for submission

Run this from the repository root after confirming the API and tests work:

```bash
bash scripts/package_submission.sh
```

Expected archives:

```text
submission/firewall_ml_project_R18_source_<date>.zip
submission/firewall_ml_project_R18_models_<date>.zip
```

Submit both archives if the reviewer needs to run the dashboard. The source archive contains code, docs, reports, and result figures. The models archive contains the trained R18 artifacts required by the API. If company policy allows data sharing, also provide `data/splits/v6_sequences/` so reviewers can run `/demo/traffic` and recompute metrics without retraining.

Before sending, follow `docs/SUBMISSION_PACKAGE.md`.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: config` | Run commands from **repo root**, not inside `scripts/` |
| `Missing Tier-2 ONNX` on API start | Run `python scripts/export/export_tier2_classifier_onnx.py` |
| `combined_dataset_v5_final.csv` not found | Place CSV in `data/raw/` (see `config.py` → `DATASET_CSV`) |
| Dashboard shows no latency | Hard-refresh browser (`Cmd+Shift+R`); run new `/demo/traffic` |
| Port 8000 in use | `lsof -ti :8000 \| xargs kill -9` then restart uvicorn |
| Tier-1 slow (~14 ms) | Old model with `n_jobs=-1`; re-run `train_tier1_gate_v6.py` |

---

## More documentation

| Document | Contents |
|----------|----------|
| [docs/IEEE_REPORT.md](docs/IEEE_REPORT.md) | IEEE figure guide (repo paths) |
| **`Hybrid_Sentinel_IEEE_Overleaf.zip`** | **Self-contained Overleaf package** |
| [docs/DESIGN_AND_IMPLEMENTATION.md](docs/DESIGN_AND_IMPLEMENTATION.md) | Design notes & implementation details |
| [docs/SUBMISSION_PACKAGE.md](docs/SUBMISSION_PACKAGE.md) | How to zip and submit final deliverables |
| [docs/CODEBASE.md](docs/CODEBASE.md) | Developer rules, production vs legacy |
| [docs/INTERVIEW_PREP.md](docs/INTERVIEW_PREP.md) | Interview talking points |
| [legacy/README.md](legacy/README.md) | What not to use |
| [docs/thesis.pdf](docs/thesis.pdf) | Full thesis document |

---

## Project rules (for contributors)

1. **One config** — paths and features only in `config.py`
2. **One cascade** — inference logic only in `src/inference/cascade_r18.py`
3. **Datasets** in `data/raw/`; splits in `data/splits/`
4. **Do not use** `legacy/` for R18 production work
