# Interview Prep — Hybrid-Sentinel

## Q: Does this work on real production traffic?

"The system is validated on lab-captured PCAP traffic across 5 attack types. In production, the main constraint is TLS — roughly 80–90% of enterprise traffic is TLS 1.3 encrypted. Our classifier operates on packet-level behavioral features (timing, size, flags, entropy), not payload content, so it remains partially effective on encrypted traffic for volume-based attacks like DDoS. For payload-dependent attacks like SlowHTTP, decryption at the network edge via a TLS inspection proxy would be required upstream of our classifier. This is a deployment architecture decision, not a model limitation — and it is the known next frontier for this type of system."

## Q: Why 3 tiers instead of one deep model?

"One deep model on every packet at high PPS is expensive on CPU. Tier-1 filters obvious benign traffic using a lightweight gate. Only suspicious flows escalate to the CNN-GRU ONNX model (0.30 ms avg on R18 benchmark). Tier-3 screens uncertain attack leaks for anomaly. This gives speed on easy traffic and depth on hard cases."

## Q: Why CNN-GRU and not a Transformer?

"Transformers need large datasets and have O(n²) attention cost. Our training set is ~93K sequences with ~14K held-out test. GRU is more data-efficient and processes sequences linearly. The CNN layer extracts per-packet feature patterns; the GRU reads how those patterns evolve over 20 packets. For our sequence length and dataset size, this architecture reaches 0.985 macro-F1 on held-out test."

## Q: What is your weakest result?

"BRUTE_FORCE recall at ~0.91 is the weakest class. The attack shares TCP connection patterns with SlowHTTP in early packets, causing occasional misclassification. We tried focal loss weight tuning (alpha 1.2 and 1.5) without reaching 0.95 — likely needs more brute-force training data, not weights alone. All other attack classes are at 0.97+ recall or 1.0."

## Q: Is this production-ready?

"Production-style — not production-deployed. We have FastAPI, ONNX export, per-decision audit logging to SQLite, Dockerfile, and a 3-tier cascade verified on 14,219 held-out test sequences. Shared inference lives in `cascade_r18.py` (single source, no drift). What remains is live traffic validation and TLS upstream integration. TRL 4 — proof of concept in a controlled environment."

## Q: How fast is it?

"Tier-2 ONNX inference averages **0.18 ms** per window (R18 benchmark, 2000 runs). Tier-3 anomaly check adds **0.21 ms**. Tier-1 gate averages **3.66 ms** (after fixing `n_jobs` inference overhead). Full cascade averages **4.19 ms** on CPU. This beats the 13 ms G-Scaler baseline by ~3.1×."

## Key numbers to memorize

- Test set: **14,219** sequences (814 benign, 13,405 attack)
- Macro F1: **0.9851** (Tier-2, held-out test)
- Attack detection: **100%** (full 3-tier cascade on test)
- Benign FPR: **0.86%** (7 of 814 benign flagged, 0 blocked)
- Tier-2 ONNX latency: **0.30 ms** avg
- Full cascade latency: **4.19 ms** avg
- Tier funnel: **5.7%** T1 allow → **94.3%** T2 → **13.2%** T3 screen
