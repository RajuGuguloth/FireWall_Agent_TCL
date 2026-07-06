"""
R18 latency benchmark — tier1_gate_v6 + ONNX CNN-GRU + Tier-3 one-class.
"""
import json
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import joblib
import numpy as np
import onnxruntime as ort

import config as cfg
from src.inference.cascade_r18 import gate_summary, load_gate, maha_score

N_PACKETS = 2000
G_SCALER_BASELINE_MS = 13.0
RESULTS_PATH = os.path.join(cfg.BASE_DIR, "results", "r18_latency_benchmark.json")


def stats(latencies_ms):
    a = np.array(latencies_ms)
    return {
        "min_ms": round(float(np.min(a)), 4),
        "avg_ms": round(float(np.mean(a)), 4),
        "p99_ms": round(float(np.percentile(a, 99)), 4),
        "max_ms": round(float(np.max(a)), 4),
    }


def main():
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)

    scaler = joblib.load(cfg.SCALER_PATH)
    gate = load_gate()
    gate_bi = list(gate.classes_).index(1)
    with open(cfg.TIER2_TEMP) as f:
        temperature = float(json.load(f)["temperature"])

    ort_cls = ort.InferenceSession(cfg.TIER2_ONNX)
    cls_in = ort_cls.get_inputs()[0].name
    embed_sess = ort.InferenceSession(cfg.TIER2_EMBED) if os.path.isfile(cfg.TIER2_EMBED) else None
    embed_in = embed_sess.get_inputs()[0].name if embed_sess else None
    t3 = joblib.load(cfg.TIER3_ONECLASS) if os.path.isfile(cfg.TIER3_ONECLASS) else None

    np.random.seed(42)
    raw = np.random.randn(N_PACKETS, cfg.WINDOW_SIZE, cfg.N_FEATURES).astype(np.float32)
    scaled = np.zeros_like(raw)
    for i in range(N_PACKETS):
        flat = scaler.transform(raw[i].reshape(-1, cfg.N_FEATURES))
        flat = np.clip(flat, -cfg.CLIP_VAL, cfg.CLIP_VAL)
        scaled[i] = flat.reshape(cfg.WINDOW_SIZE, cfg.N_FEATURES)

    results = {}

    # Tier-1 gate
    lat_t1 = []
    t0 = time.perf_counter()
    for i in range(N_PACKETS):
        t = time.perf_counter()
        _ = gate.predict_proba(gate_summary(scaled[i:i + 1]))[0, gate_bi]
        lat_t1.append((time.perf_counter() - t) * 1000)
    results["tier1_gate"] = {**stats(lat_t1), "pps": int(N_PACKETS / (time.perf_counter() - t0))}

    # Tier-2 ONNX
    lat_t2 = []
    t0 = time.perf_counter()
    for i in range(N_PACKETS):
        t = time.perf_counter()
        _ = ort_cls.run(None, {cls_in: scaled[i:i + 1]})[0]
        lat_t2.append((time.perf_counter() - t) * 1000)
    results["tier2_onnx"] = {**stats(lat_t2), "pps": int(N_PACKETS / (time.perf_counter() - t0))}

    # Tier-3 embed + Mahalanobis
    if embed_sess and t3:
        lat_t3 = []
        t0 = time.perf_counter()
        for i in range(N_PACKETS):
            t = time.perf_counter()
            emb = embed_sess.run(None, {embed_in: scaled[i:i + 1]})[0][0]
            _ = maha_score(emb, t3)
            lat_t3.append((time.perf_counter() - t) * 1000)
        results["tier3_oneclass"] = {**stats(lat_t3), "pps": int(N_PACKETS / (time.perf_counter() - t0))}
    else:
        results["tier3_oneclass"] = {"error": "missing embed or tier3_oneclass"}

    # Full cascade (matches api/main.py logic)
    lat_cascade = []
    t0 = time.perf_counter()
    for i in range(N_PACKETS):
        t = time.perf_counter()
        seq = scaled[i:i + 1]
        p_ben = gate.predict_proba(gate_summary(seq))[0, gate_bi]
        if p_ben >= cfg.GATE_THRESHOLD:
            pass
        else:
            logits = ort_cls.run(None, {cls_in: seq})[0][0]
            z = logits / temperature
            z -= z.max()
            probs = np.exp(z) / np.exp(z).sum()
            conf = float(probs.max())
            label_idx = int(probs.argmax())
            classes = list(joblib.load(cfg.ENCODER_PATH).classes_)
            label = classes[label_idx]
            if label != "BENIGN" and conf < cfg.BLOCK_THRESHOLD and embed_sess and t3:
                if label != "BENIGN" and conf < cfg.FLAG_THRESHOLD:
                    emb = embed_sess.run(None, {embed_in: seq})[0][0]
                    _ = maha_score(emb, t3)
        lat_cascade.append((time.perf_counter() - t) * 1000)
    results["full_cascade"] = {**stats(lat_cascade), "pps": int(N_PACKETS / (time.perf_counter() - t0))}

    cascade_avg = results["full_cascade"]["avg_ms"]
    beat = round(G_SCALER_BASELINE_MS / cascade_avg, 2) if cascade_avg > 0 else 0

    out = {
        "benchmark_date": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "round": 18,
        "n_packets": N_PACKETS,
        "window": cfg.WINDOW_SIZE,
        "features": cfg.N_FEATURES,
        "g_scaler_baseline_avg_ms": G_SCALER_BASELINE_MS,
        "beat_baseline_by_x": beat,
        "results": results,
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(out, f, indent=2)

    print("=" * 56)
    print("  R18 Latency Benchmark")
    print("=" * 56)
    for name, s in results.items():
        if "error" in s:
            print(f"  {name}: {s['error']}")
        else:
            print(f"  {name}: avg={s['avg_ms']}ms p99={s['p99_ms']}ms max={s['max_ms']}ms PPS={s['pps']}")
    print(f"  G-Scaler baseline: {G_SCALER_BASELINE_MS}ms avg")
    print(f"  Full cascade beats baseline by: {beat}x")
    print(f"  Saved → {RESULTS_PATH}")


if __name__ == "__main__":
    main()
