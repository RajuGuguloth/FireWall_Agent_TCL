# Internship Final Submission — Packaging Guide

Use this checklist to respond to the TCL/IITM archival request.

---

## What to submit

| # | Item | Location in repo |
|---|------|------------------|
| 1 | **Final source code** | Full repo (see zip script below) |
| 2 | **Final report** | `docs/FINAL_SUBMISSION_REPORT.md` |
| 3 | **Training approach** | Report §4 + `docs/DESIGN_AND_IMPLEMENTATION.md` §3–5 |
| 4 | **Performance metrics** | `results/submission_metrics_r18.json`, `results/r18_tier_metrics.json`, `results/r18_latency_benchmark.json` |
| 5 | **Evaluation results** | Report §5–6; run `measure_cascade_flow.py` for fresh log |
| 6 | **Setup documentation** | `README.md`, `docs/CODEBASE.md` |
| 7 | **Design & implementation** | `docs/DESIGN_AND_IMPLEMENTATION.md` |
| 8 | **Dashboard demo** | `api/main.py`, `api/static/dashboard.html`, root README dashboard section |
| 9 | **IEEE/Overleaf report package** | `Hybrid_Sentinel_IEEE_Overleaf.zip` |
| 10 | **Compiled IEEE PDF** | `Hybrid_Sentinel__A_Three_Tier_Confidence_GatedAI_Firewall_for_NDN_Edge_Deployment.pdf` |

---

## Recommended email reply (template)

```
Subject: Final Submission — Hybrid-Sentinel AI Firewall (R18)

Dear [Name],

Please find attached the final submission package for the AI Firewall project.

Contents:
1. Source code archive (firewall_ml_project_R18_source.zip)
2. Trained model bundle (firewall_ml_project_R18_models.zip) — R18 artifacts only
3. Documentation:
   - FINAL_SUBMISSION_REPORT.md (report, metrics, training, evaluation)
   - DESIGN_AND_IMPLEMENTATION.md (design notes)
   - README.md (setup and reproduction)
4. Results: submission_metrics_r18.json, r18_tier_metrics.json, r18_latency_benchmark.json
5. IEEE/Overleaf package: Hybrid_Sentinel_IEEE_Overleaf.zip
6. Compiled report PDF: Hybrid_Sentinel__A_Three_Tier_Confidence_GatedAI_Firewall_for_NDN_Edge_Deployment.pdf

Production version: Round 18 (R18)
Key results: 100% attack detection, 1.11% benign FPR, 4.19ms cascade latency on held-out test (n=14,219).

Dashboard demo: run `uvicorn api.main:app --host 127.0.0.1 --port 8000` and open http://127.0.0.1:8000/.
Happy to walk through the API + dashboard if needed.

Best regards,
Raju Guguloth
```

---

## Build submission archives

From repository root:

```bash
# 1. Source code (no huge legacy models, no raw PCAP/CSV)
bash scripts/package_submission.sh

# 2. Optional: refresh metrics before packaging
source .venv/bin/activate
python scripts/eval/measure_cascade_flow.py | tee results/cascade_flow_final.txt
python api/tier_metrics.py   # or: curl http://127.0.0.1:8000/metrics/tiers?refresh=true
```

Outputs:

- `submission/firewall_ml_project_R18_source.zip` — code + docs + results JSON  
- `submission/firewall_ml_project_R18_models.zip` — R18 models only (~2 MB)

---

## What goes in each zip

### Source zip (required)

- `api/`, `src/`, `scripts/`, `tests/`, `config.py`
- `requirements.txt`, `requirements_api.txt`
- `Dockerfile`, `docker-compose.yml`
- `README.md`, `docs/` (all markdown + thesis.pdf)
- `Hybrid_Sentinel_IEEE_Overleaf.zip` (self-contained IEEE report package)
- `Hybrid_Sentinel__A_Three_Tier_Confidence_GatedAI_Firewall_for_NDN_Edge_Deployment.pdf` (compiled report)
- `results/r18_*.json`, `results/cascade_flow_final.txt` (if generated)
- `legacy/` (optional — for historical reference)

### Excluded from source zip

- `.venv/`, `__pycache__/`, `.git/`
- `tier1_rf_v3.pkl`, `tier1_rf_v4.pkl` (6+ GB each)
- `*.pcap`, `*.csv` (patent/size — note in README)
- `data/splits/*.npy` (large — separate or regenerate via pipeline)

### Models zip (required for runnable demo)

- `models/tier1_gate_v6.pkl`
- `models/tier2_cnn_gru_v1_r18.pth`
- `models/tier2_r18_temperature.json`
- `models/tier3_oneclass_v6.pkl`
- `models/serialized/v6_scaler.pkl`, `v6_encoder.pkl`
- `models/onnx/tier2_cnn_gru_r18.onnx`, `tier2_embedding_r18.onnx`

### Data zip (optional — if reviewers must run without retraining)

- `data/splits/v6_sequences/` (6 `.npy` files, ~130 MB)
- `data/raw/combined_dataset_v5_final.csv` (if policy allows)

---

## Pre-submission checklist

- [ ] Fill in name/email in `docs/FINAL_SUBMISSION_REPORT.md` §11
- [ ] Run `python -m pytest tests/ -q` — all pass
- [ ] Run `measure_cascade_flow.py` — attack detection 100%, benign FPR around 1.11%
- [ ] Confirm API starts: `uvicorn api.main:app --port 8000`
- [ ] Open dashboard http://127.0.0.1:8000/ — metrics load
- [ ] Trigger demo traffic: `curl -X POST "http://127.0.0.1:8000/demo/traffic?n=20"`
- [ ] Build zips with `scripts/package_submission.sh`
- [ ] Upload to GitLab / share drive / email per TCL instructions
- [ ] Push final commit to `tc-ndn-iitm` GitLab if required

---

## If they only want GitLab (no zip)

Ensure these paths are committed or uploaded to the project wiki:

1. `docs/FINAL_SUBMISSION_REPORT.md`
2. `docs/DESIGN_AND_IMPLEMENTATION.md`
3. `README.md`
4. Link to `results/r18_tier_metrics.json` in repo or wiki attachment

Note: `models/`, `data/`, `results/` may be gitignored — use GitLab **Release assets** or LFS for binaries.

---

*Last updated: July 2026*
