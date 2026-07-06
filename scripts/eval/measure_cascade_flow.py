"""
Cascade flow tracer — counts packets at every tier on the v6 TEST set.
Uses config.py paths and the serialized Tier-3 detector (same as production).
"""
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import joblib
import numpy as np
import torch

import config as cfg
from src.inference.cascade_r18 import gate_summary, load_gate
from src.models.cnn_gru_v6 import CNNGRUClassifier


def embed(model, X):
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 512):
            x = torch.FloatTensor(X[i:i + 512])
            x = model.relu(model.bn1(model.conv1(x.transpose(1, 2)))).transpose(1, 2)
            out.append(model.gru(x)[0][:, -1, :].numpy())
    return np.concatenate(out)


def main():
    le = joblib.load(cfg.ENCODER_PATH)
    names = list(le.classes_)
    B = names.index("BENIGN")

    Xte = np.load(f"{cfg.SEQ_DIR}/X_test.npy")
    yte = np.load(f"{cfg.SEQ_DIR}/y_test.npy")
    gate = load_gate()
    with open(cfg.TIER2_TEMP) as f:
        T = float(json.load(f)["temperature"])

    m = CNNGRUClassifier(num_classes=len(names))
    m.load_state_dict(torch.load(cfg.TIER2_PTH, map_location="cpu"))
    m.eval()

    t3 = joblib.load(cfg.TIER3_ONECLASS)
    t3_thr = float(t3["threshold"])

    benign = yte == B
    attack = ~benign
    N, nb, na = len(yte), int(benign.sum()), int(attack.sum())

    def row(lbl, mask):
        return (
            f"  {lbl:<28} {mask.sum():>7,}  | benign {(mask & benign).sum():>6,} "
            f"| attack {(mask & attack).sum():>6,}"
        )

    print("=" * 70)
    print("  CASCADE FLOW — clean v6 TEST set (config + tier3_oneclass_v6.pkl)")
    print("=" * 70)
    print(row("STAGE 0  INPUT", np.ones(N, bool)))

    pben = gate.predict_proba(gate_summary(Xte))[:, list(gate.classes_).index(1)]
    gate_allow = pben >= cfg.GATE_THRESHOLD
    esc = ~gate_allow
    print("\n-- TIER-1 (lean gate) --")
    print(row("ALLOWED (fast path)", gate_allow))
    print(row("ESCALATED -> Tier-2", esc))

    probs = np.zeros((N, len(names)), np.float32)
    with torch.no_grad():
        if esc.sum():
            probs[esc] = torch.softmax(m(torch.FloatTensor(Xte[esc])) / T, 1).numpy()
    pred = probs.argmax(1)
    conf = probs.max(1)
    t2_block = esc & (pred != B) & (conf > cfg.BLOCK_THRESHOLD)
    t2_flag = esc & (pred != B) & (conf >= cfg.FLAG_THRESHOLD) & (conf <= cfg.BLOCK_THRESHOLD)
    t2_allow = esc & ~t2_block & ~t2_flag
    print("\n-- TIER-2 (CNN-GRU, on escalated only) --")
    print(row("BLOCK (deny)", t2_block))
    print(row("FLAG (review)", t2_flag))
    print(row("ALLOWED (candidate)", t2_allow))

    allow_candidates = gate_allow | t2_allow
    E = embed(m, Xte)
    d = E - t3["mu"]
    scores = np.einsum("ij,jk,ik->i", d, t3["inv_cov"], d)
    anom = scores > t3_thr
    t3_flag = allow_candidates & anom
    t3_allow = allow_candidates & ~anom
    print("\n-- TIER-3 (one-class zero-day net, on ALLOW candidates) --")
    print(row("FLAG (novel/anomaly)", t3_flag))
    print(row("ALLOWED (final)", t3_allow))

    final_block = t2_block
    final_flag = t2_flag | t3_flag
    final_allow = t3_allow
    print("\n" + "=" * 70)
    print("  FINAL OUTCOME")
    print("=" * 70)
    print(row("ALLOW", final_allow))
    print(row("FLAG", final_flag))
    print(row("BLOCK", final_block))

    print("\n" + "=" * 70)
    print("  KEY RATES")
    print("=" * 70)
    b_denied = (final_block | final_flag) & benign
    a_denied = (final_block | final_flag) & attack
    print(f"  Benign false-positive rate : {100 * b_denied.sum() / nb:5.2f}%  ({b_denied.sum()}/{nb})")
    print(f"  Attack detection rate      : {100 * a_denied.sum() / na:5.2f}%  ({a_denied.sum()}/{na})")
    print(
        f"  Attacks leaked past gate   : {(esc & attack).sum()}/{na} escalated, "
        f"{(gate_allow & attack).sum()} allowed at gate"
    )
    print(f"  Tier-2 workload            : {esc.sum()}/{N} ({100 * esc.mean():.1f}%) sequences")
    print(f"  Tier-3 workload            : {allow_candidates.sum()}/{N} ({100 * allow_candidates.mean():.1f}%)")


if __name__ == "__main__":
    main()
