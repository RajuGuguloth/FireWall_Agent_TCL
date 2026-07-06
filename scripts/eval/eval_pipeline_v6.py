"""
Round 18 — Tier-1+2 cascade evaluation (no Tier-3).
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
from sklearn.metrics import f1_score

import config as cfg
from src.inference.cascade_r18 import gate_summary, load_gate
from src.models.cnn_gru_v6 import CNNGRUClassifier


def main():
    le = joblib.load(cfg.ENCODER_PATH)
    names = list(le.classes_)
    B = names.index("BENIGN")
    Xte = np.load(f"{cfg.SEQ_DIR}/X_test.npy")
    yte = np.load(f"{cfg.SEQ_DIR}/y_test.npy")
    gate = load_gate()
    with open(cfg.TIER2_TEMP) as f:
        T = float(json.load(f)["temperature"])

    model = CNNGRUClassifier(num_classes=len(names))
    model.load_state_dict(torch.load(cfg.TIER2_PTH, map_location="cpu"))
    model.eval()

    bidx = list(gate.classes_).index(1)
    pgate = gate.predict_proba(gate_summary(Xte))[:, bidx]
    fast_allow = pgate >= cfg.GATE_THRESHOLD
    esc = ~fast_allow

    probs = np.zeros((len(Xte), len(names)), dtype=np.float32)
    if esc.sum():
        with torch.no_grad():
            lg = model(torch.FloatTensor(Xte[esc])) / T
            probs[esc] = torch.softmax(lg, 1).numpy()
    t2pred = probs.argmax(1)
    t2conf = probs.max(1)

    action = np.array(["ALLOW"] * len(Xte), dtype=object)
    final_pred = np.full(len(Xte), B)
    for i in range(len(Xte)):
        if fast_allow[i]:
            continue
        if t2pred[i] == B:
            continue
        final_pred[i] = t2pred[i]
        if t2conf[i] > cfg.BLOCK_THRESHOLD:
            action[i] = "BLOCK"
        elif t2conf[i] >= cfg.FLAG_THRESHOLD:
            action[i] = "FLAG"
        else:
            action[i] = "ALLOW"

    benign = yte == B
    attack = ~benign
    b_blocked = (benign & (action != "ALLOW")).sum()
    fpr = 100 * b_blocked / benign.sum()
    a_caught = (attack & (action != "ALLOW")).sum()
    det = 100 * a_caught / attack.sum()

    print("=" * 60)
    print("  TIER-1+2 CASCADE — clean v6 TEST (Tier-3 excluded)")
    print("=" * 60)
    print(f"  Test sequences            : {len(Xte):,} (benign {benign.sum():,}, attack {attack.sum():,})")
    print(f"  Fast-path allowed by gate : {fast_allow.sum():,} ({100 * fast_allow.mean():.1f}%)")
    print("-" * 60)
    print(f"  BENIGN false-positive rate: {fpr:.2f}%   ({b_blocked}/{benign.sum()})")
    print(f"  ATTACK detection rate     : {det:.2f}%   ({a_caught}/{attack.sum()})")
    mf1 = f1_score(yte, final_pred, average="macro", zero_division=0)
    print(f"  Overall macro-F1 (6-class): {mf1:.4f}")
    caught = attack & (action != "ALLOW")
    if caught.sum():
        type_acc = 100 * (final_pred[caught] == yte[caught]).mean()
        print(f"  Attack-type accuracy (on caught): {type_acc:.2f}%")
    print("=" * 60)
    print("  For full 3-tier eval: python measure_cascade_flow.py")


if __name__ == "__main__":
    main()
