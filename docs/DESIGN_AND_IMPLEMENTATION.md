# Design Notes & Implementation Details — Hybrid-Sentinel R18

This document archives **design decisions** and **implementation specifics** for TCL/IITM records.

---

## 1. Design principles

| Principle | Implementation |
|-----------|----------------|
| Single contract | All tiers import paths, features, thresholds from `config.py` |
| Single cascade | All inference logic in `src/inference/cascade_r18.py` |
| Fail closed on attacks | Tier-1 escalates uncertain traffic; Tier-2 blocks high-confidence attacks |
| Fail open on benign (with FLAG) | Medium confidence → FLAG, not BLOCK |
| Explainability | Every API response includes `tier_trace` JSON |
| No test leakage | Group-disjoint splits; scaler on train only |

---

## 2. Feature contract (17 dimensions)

Defined in `config.py` — **do not redefine elsewhere**.

```
packet_length, has_tcp, has_udp, has_icmp,
payload_length, payload_entropy,
is_ack, is_rst, is_fin, is_psh,
is_high_port_src, ip_ttl, ip_proto,
dst_port, tcp_flags,
flow_total_bytes, flow_mean_pkt_len
```

**Extraction:** Scapy from PCAP (`scripts/etl/extract_v5_features.py`).  
**Window:** 20 packets, stride 10.  
**Scaling:** `StandardScaler` fit on train; clip to ±5.0.

**Design note:** Behavioral features work on encrypted traffic (no payload decode) but cannot see shellcode byte patterns (Packet BERT path in wiki).

---

## 3. Tier-1 gate design

### Why Random Forest?

- Fast CPU inference on summarized windows  
- Interpretable feature importance  
- No GPU required at edge  

### Input engineering

```python
summary = concat(mean(window, axis=packets), std(window), last_packet)
# Shape: (51,) per window
```

### Why not legacy `tier1_rf_v4.pkl` (6.76 GB)?

- Trained on different column set (flow CSV, not 17-feature windows)  
- Serialization bloat; misaligned with Tier-2  
- R18 gate: 150 trees, 156 KB, same schema as Tier-2  

### `n_jobs` fix (July 2026)

Model saved with `n_jobs=-1` caused ~14 ms single-sample inference. Fixed: `rf.n_jobs = 1` before `joblib.dump` and on API load → ~3.7 ms.

---

## 4. Tier-2 CNN-GRU design

### Architecture (`src/models/cnn_gru_v6.py`)

```
Input: (batch, 20, 17)
  → Conv1d(17→64, k=3) + BatchNorm + ReLU
  → GRU(64→128, 2 layers, dropout=0.3)
  → Last hidden state → Dropout → Linear(128→6)
```

### Why CNN + GRU (not Transformer)?

- Dataset ~93K train sequences — Transformers need more data  
- O(n) GRU vs O(n²) attention — better latency  
- CNN captures per-packet local patterns; GRU temporal order  

### Why Focal Loss?

Class imbalance (attacks >> benign in raw rows). Focal loss down-weights easy examples. BENIGN weight 2.0 to protect rare benign windows.

### Temperature calibration

Softmax logits divided by learned temperature T≈0.77 on validation set before confidence thresholds. Reduces overconfident BLOCK decisions.

### ONNX export

- Classifier: `models/onnx/tier2_cnn_gru_r18.onnx`  
- Embedding head: `models/onnx/tier2_embedding_r18.onnx` (for Tier-3)  
- Runtime: `onnxruntime` in API (no PyTorch at inference for Tier-2)

---

## 5. Tier-3 one-class design

### Role

**Anomaly detector only** — outputs FLAG with label `ANOMALY`, not an attack class name.

### Algorithm

1. Extract 128-D embedding from Tier-2 GRU hidden state (benign train samples)  
2. Fit mean μ and inverse covariance Σ⁻¹  
3. Score = (x − μ)ᵀ Σ⁻¹ (x − μ)  
4. If score > threshold → FLAG  

### API vs full eval difference

| Path | When Tier-3 runs |
|------|------------------|
| **API** (`cascade_r18.py`) | Tier-2 ALLOW leak where pred ≠ BENIGN (~7.4%) |
| **Full eval** (`measure_cascade_flow.py`) | All ALLOW candidates incl. Tier-1 fast-path (~13.2%) |

Document this in meetings — both are correct for their context.

---

## 6. Cascade decision logic

Pseudocode (matches `cascade_r18.py`):

```
scale(window)
p_benign = tier1.predict_proba(summary(window))

if p_benign >= 0.90:
    candidate = ALLOW
else:
    probs = softmax(tier2_onnx(window) / T)
    if argmax == BENIGN:
        candidate = ALLOW
    elif max_prob > 0.95:
        return BLOCK
    elif max_prob >= 0.80:
        return FLAG
    else:
        candidate = ALLOW  # leak

if tier3_enabled and candidate == ALLOW and tier2_pred != BENIGN:
    if mahalanobis(embed(window)) > threshold:
        return FLAG, label=ANOMALY

return candidate
```

---

## 7. API implementation (`api/main.py`)

| Component | Purpose |
|-----------|---------|
| `CascadeRuntime.load()` | Load scaler, gate, ONNX, Tier-3 |
| `POST /predict` | Single window → action + tier_trace + latency_ms |
| `POST /predict/batch` | Batch inference |
| `POST /demo/traffic` | Replay X_test.npy samples |
| `GET /metrics/tiers` | Cached offline metrics |
| `GET /stats` | Live session counters |
| `alert_store.py` | SQLite WAL, stores latency_ms |

**Dashboard:** `api/static/dashboard.html` — live session, offline metrics, 3D flow (`flow3d.js`).

---

## 8. Data integrity (R18 vs R17)

### R17 bug — BENIGN stripped

`prepare_v5_sequences.py` line 41–42 filtered to attacks only. Any benign window at runtime forced into attack class.

### R17 bug — val = test

`train_cnn_gru_v4.py` used `X_test` as validation for early stopping.

### R18 fix

`prepare_v6_sequences.py`:

- 6 classes including BENIGN  
- 70/15/15 split with `GroupShuffleSplit`  
- `assert` no group overlap across train/val/test  
- Scaler fit on `X_train` only, then transform val/test  

---

## 9. File manifest (production artifacts)

| Artifact | Path |
|----------|------|
| Tier-1 gate | `models/tier1_gate_v6.pkl` |
| Tier-2 weights | `models/tier2_cnn_gru_v1_r18.pth` |
| Temperature | `models/tier2_r18_temperature.json` |
| Tier-2 ONNX | `models/onnx/tier2_cnn_gru_r18.onnx` |
| Embedding ONNX | `models/onnx/tier2_embedding_r18.onnx` |
| Tier-3 | `models/tier3_oneclass_v6.pkl` |
| Scaler | `models/serialized/v6_scaler.pkl` |
| Encoder | `models/serialized/v6_encoder.pkl` |
| Sequences | `data/splits/v6_sequences/*.npy` |
| Raw CSV | `data/raw/combined_dataset_v5_final.csv` |

**Exclude from submission zip:** `tier1_rf_v3.pkl`, `tier1_rf_v4.pkl` (~6 GB each), old R17 checkpoints unless requested.

---

## 10. Testing

```bash
python -m pytest tests/ -q
```

Smoke tests: config contract, `CascadeRuntime.load()`, API import.

---

## 11. Version history summary

| Round | Key change |
|-------|------------|
| v3/v4 | Per-flow RF; weak BRUTE_FORCE/SLOW_HTTP F1 |
| R16 | CNN-GRU introduced; 3 attack classes |
| R17 | 5 attack classes; **BENIGN removed**; val=test; broken cascade |
| **R18** | 6 classes; honest splits; lean gate; unified cascade; ONNX API |

---

## 12. Integration points (team program)

| Future input | Integration method |
|--------------|-------------------|
| Malware Zoo PCAPs | `extract_v5_features.py` → append to CSV → retrain or Tier-3 only |
| NFD topology | Deploy API container inline with forwarder tap |
| UNSW PCAP replay | Docker isolated capture → same extractor |
| Packet BERT | Optional Tier-2b expert; router by protocol header |

---

*Document version: R18 — July 2026*
