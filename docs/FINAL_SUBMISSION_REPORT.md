# Hybrid-Sentinel NDN AI Firewall — Final Submission Report

**Project:** tc-ndn-iitm — AI-Driven Next-Generation Firewall  
**Production version:** Round 18 (R18)  
**Author:** Raju Guguloth  
**Institution:** IIT Madras (M.Tech / TC collaboration)  
**Submission date:** July 2026  
**Repository:** `firewall_ml_project`

---

## 1. Executive summary

**Hybrid-Sentinel** is a 3-tier machine-learning intrusion detection system for NDN-style network traffic. It inspects windows of **20 packets × 17 behavioral features** and returns **ALLOW**, **FLAG**, or **BLOCK** with a per-tier audit trace.

On a held-out test set of **14,219 sequences** (814 benign, 13,405 attack), the full R18 cascade achieves:

| Metric | Result |
|--------|--------|
| End-to-end attack detection | **100%** |
| Benign false-positive rate | **0.86%** (7 / 814 flagged; 0 blocked) |
| Tier-2 macro-F1 (6 classes) | **0.9851** |
| Full cascade latency (CPU) | **4.19 ms** average |
| Tier-1 model size | **156 KB** (vs 6.76 GB legacy) |

Deliverables include: training pipeline, ONNX export, FastAPI service, SOC dashboard, SQLite alert store, Docker support, and offline evaluation scripts.

---

## 2. Problem statement

### 2.1 Research context (TCL GitLab wiki)

The project follows the **hybrid AI firewall** architecture:

1. **Phase 1 — Fast filter:** cheap gate on most traffic  
2. **Phase 2 — Deep AI:** ONNX classifier on suspicious traffic only  
3. **NDN motivation:** content-centric networks benefit from inspection at the forwarding edge  

### 2.2 Technical problem solved in R18

Build a cascade that:

- Minimizes benign false positives  
- Maximizes attack detection  
- Classifies attack **type** where possible  
- Flags **zero-day / novel** patterns without predicting a class name  
- Runs in **low single-digit milliseconds** on CPU  
- Provides **explainable** per-tier decisions for SOC audit  

### 2.3 Why Round 18 replaced Round 17

| R17 flaw | R18 fix |
|----------|---------|
| BENIGN stripped from Tier-2 training → **100% benign FPR** | 6 classes including BENIGN |
| Validation set = test set (metric inflation) | 70/15/15 group-disjoint train/val/test |
| Tier-1 6.76 GB model, feature drift | Lean 156 KB gate, shared `config.py` |
| Fragmented cascade logic | Single `src/inference/cascade_r18.py` |

---

## 3. System architecture

```
Packet window (20 × 17 features, scaled)
        │
        ▼
  Tier-1  Random Forest gate     P(BENIGN) ≥ 0.90 → ALLOW
        │                        else → escalate
        ▼
  Tier-2  CNN-GRU (ONNX)         6-class + confidence thresholds
        │                        BLOCK / FLAG / ALLOW
        ▼
  Tier-3  Mahalanobis one-class  on 128-D embedding → FLAG if novel
        │
        ▼
  ALLOW / FLAG / BLOCK  +  tier_trace
```

### Attack classes

| Class | Lab source |
|-------|------------|
| BENIGN | Baseline HTTP/DNS/HTTPS PCAP |
| BRUTE_FORCE | SSH/login brute-force PCAP |
| DDOS_HTTP_FLOOD | HTTP flood PCAP |
| SLOW_HTTP | Slow-rate HTTP PCAP |
| PORT_SCAN | SYN/connect scan PCAP |
| DNS_TUNNELING | DNS tunnel PCAP |
| ANOMALY | Tier-3 output only (not trained class) |

---

## 4. Training approach

### 4.1 Data pipeline

| Step | Script | Output |
|------|--------|--------|
| PCAP → 17 features | `scripts/etl/extract_v5_features.py` | `data/raw/combined_dataset_v5_final.csv` |
| Sequences + split | `scripts/data/prepare_v6_sequences.py` | `data/splits/v6_sequences/*.npy` |
| Scaler + encoder | (same script) | `models/serialized/v6_scaler.pkl`, `v6_encoder.pkl` |

**Split policy:** Group-disjoint 70/15/15 by pseudo-flow blocks (1000 packets same label). Scaler fit on **train only**. No train/val/test group overlap (asserted in script).

### 4.2 Tier-1 — BENIGN vs ATTACK gate

| Item | Detail |
|------|--------|
| Script | `scripts/training/train_tier1_gate_v6.py` |
| Model | RandomForest (150 trees, max_depth=16) |
| Input | Window summary: mean(17) + std(17) + last packet(17) = 51 dims |
| Output | `models/tier1_gate_v6.pkl` |
| Inference | `n_jobs=1` for low single-sample latency |

### 4.3 Tier-2 — 6-class CNN-GRU

| Item | Detail |
|------|--------|
| Script | `scripts/training/train_cnn_gru_v6.py` |
| Architecture | Conv1D(17→64) → GRU(64→128, 2 layers) → Linear(128→6) |
| Loss | Focal Loss (γ=2), class weights (BENIGN=2.0) |
| Calibration | Temperature scaling on validation set |
| Output | `models/tier2_cnn_gru_v1_r18.pth`, `tier2_r18_temperature.json` |
| Production | `scripts/export/export_tier2_classifier_onnx.py` → ONNX |

### 4.4 Tier-3 — Zero-day anomaly

| Item | Detail |
|------|--------|
| Script | `scripts/export/export_tier3_oneclass.py` |
| Method | Mahalanobis distance on Tier-2 embeddings (benign train distribution) |
| Output | `models/tier3_oneclass_v6.pkl` |
| Role | FLAG only — does **not** predict attack type |

### 4.5 Thresholds (`config.py`)

| Parameter | Value | Effect |
|-----------|-------|--------|
| `GATE_THRESHOLD` | 0.90 | Tier-1 fast ALLOW |
| `BLOCK_THRESHOLD` | 0.95 | Tier-2 auto-block |
| `FLAG_THRESHOLD` | 0.80 | Tier-2 human review |

---

## 5. Performance metrics

*Source: `results/r18_tier_metrics.json` (generated 2026-06-17)*

### 5.1 Tier-1 gate (test n=14,219)

| Metric | Value |
|--------|-------|
| Accuracy | 1.0 |
| Benign FPR | 0.0% |
| Attack recall | 100% |
| Model size | ~156 KB |

### 5.2 Tier-2 CNN-GRU (test n=14,219)

| Metric | Value |
|--------|-------|
| Macro-F1 | **0.9851** |
| Accuracy | 0.9807 |

**Per-class F1:**

| Class | F1 | Recall |
|-------|-----|--------|
| BENIGN | 1.000 | 1.000 |
| BRUTE_FORCE | 0.954 | 0.914 |
| DDOS_HTTP_FLOOD | 0.957 | 0.997 |
| DNS_TUNNELING | 1.000 | 1.000 |
| PORT_SCAN | 1.000 | 1.000 |
| SLOW_HTTP | 1.000 | 1.000 |

*Weakest class: BRUTE_FORCE recall ~91% — overlaps with SLOW_HTTP in early packets.*

### 5.3 Tier-3 one-class

| Metric | Value |
|--------|-------|
| ROC-AUC | 1.0 (lab embeddings) |
| Benign false alarm | 0.86% |
| Attack detection | 100% |

### 5.4 Full cascade (authoritative eval)

| Metric | Value |
|--------|-------|
| Attack detection | **100%** |
| Benign FPR | **0.86%** |
| Tier-2 workload | 94.3% of sequences |
| Tier-3 workload | 13.2% (full eval); ~7.4% (API runtime path) |

**Eval command:** `python scripts/eval/measure_cascade_flow.py`

### 5.5 Latency (CPU, n=2000 runs)

*Source: `results/r18_latency_benchmark.json`*

| Component | Avg (ms) | p99 (ms) |
|-----------|----------|----------|
| Tier-1 gate | 3.66 | 9.31 |
| Tier-2 ONNX | 0.18 | 0.29 |
| Tier-3 one-class | 0.21 | 0.36 |
| **Full cascade** | **4.19** | 6.49 |

API `/predict` adds ~2–3 ms (JSON + SQLite). See `results/r18_api_latency_check.json`.

---

## 6. Evaluation methodology

1. **Held-out test:** `data/splits/v6_sequences/X_test.npy` — never used for training or model selection.  
2. **Per-tier metrics:** `api/tier_metrics.py` → cached `results/r18_tier_metrics.json`.  
3. **Cascade flow:** `scripts/eval/measure_cascade_flow.py` — packet counts at each stage.  
4. **Partial eval:** `scripts/eval/eval_pipeline_v6.py` — Tier-1+2 only; **not** final production metric.  
5. **Live API:** `/demo/traffic` replays test sequences; session counters are runtime-only (not offline F1).

---

## 7. Setup and reproduction

See **README.md** (repository root) for full instructions.

**Quick start (API):**

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements_api.txt
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

**Full training pipeline:**

```bash
pip install -r requirements.txt
python scripts/data/prepare_v6_sequences.py
python scripts/training/train_tier1_gate_v6.py
python scripts/training/train_cnn_gru_v6.py
python scripts/export/export_tier2_classifier_onnx.py
python scripts/export/export_tier2_embedding_onnx.py
python scripts/export/export_tier3_oneclass.py
python scripts/eval/measure_cascade_flow.py
python scripts/benchmark/benchmark_latency.py
```

**Docker:** `docker compose up --build` (requires `models/` and `data/` on host).

---

## 8. Scope, limitations, and future work

### In scope (R18)

- Hybrid 3-tier cascade on lab PCAP traffic  
- 6-class behavioral detection + zero-day FLAG  
- FastAPI + dashboard + audit logging  
- ONNX production path  

### Out of scope / future (tc-ndn-iitm program)

| Item | Owner / status |
|------|----------------|
| Hybrid bare-metal + cloud topology | Tushar — GitLab issue #1 |
| Malware Zoo (50 samples in NDN Data) | Jyotirmaya — in progress |
| Native NDN Interest/Data packet fields | Future feature engineering |
| Packet BERT payload tokens | Wiki long-term; R18 uses behavioral features |
| UNSW / external PCAP validation | Via `extract_v5_features.py`, not raw CSV import |

### Known limitations

- Validated on **in-silico Docker lab** PCAPs, not live NFD deployment  
- ~80–90% enterprise traffic is TLS — payload-dependent attacks need edge decryption  
- Tier-3 ROC-AUC 1.0 is on lab embeddings; needs OOD validation with Malware Zoo  
- BRUTE_FORCE recall weakest at ~91%  

---

## 9. Repository structure (production)

```
api/              FastAPI + SOC dashboard
src/inference/    cascade_r18.py (single cascade)
src/models/       CNN-GRU architecture
scripts/          data, training, export, eval, benchmark
config.py         paths, features, thresholds
docs/             this report, design notes, CODEBASE.md
tests/            smoke tests
legacy/           R17 and older — do not use
```

---

## 10. References and related documents

| Document | Path |
|----------|------|
| Setup guide | `README.md` |
| Architecture rules | `docs/CODEBASE.md` |
| Design & implementation | `docs/DESIGN_AND_IMPLEMENTATION.md` |
| Submission packaging | `docs/SUBMISSION_PACKAGE.md` |
| Metrics JSON | `results/r18_tier_metrics.json` |
| Latency JSON | `results/r18_latency_benchmark.json` |
| Thesis | `docs/thesis.pdf` |
| TCL wiki | Next-Generation Firewall (Snort vs Packet BERT vs NDN) |

---

## 11. Sign-off

This report documents the **Round 18 production deliverable** of the Hybrid-Sentinel AI Firewall developed during the IIT Madras / TCL collaboration. All metrics cited are from the **v6 held-out test set** unless labeled as API runtime measurements.

**Prepared by:** Raju Guguloth  
**Contact:** [your email]  
**GitLab:** tc-ndn-iitm / firewall_ml_project
