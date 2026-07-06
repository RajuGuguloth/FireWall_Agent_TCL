"""
Round 18 — Lean Tier-1 BENIGN-vs-ATTACK gate.
"""
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

import config as cfg
DATA = cfg.SEQ_DIR
ENC  = cfg.ENCODER_PATH
OUT  = cfg.TIER1_GATE


def summary(X):
    return np.concatenate([X.mean(1), X.std(1), X[:, -1, :]], axis=1)


def main():
    le = joblib.load(ENC); names = list(le.classes_); benign = names.index("BENIGN")
    Xtr = np.load(f"{DATA}/X_train.npy"); ytr = np.load(f"{DATA}/y_train.npy")
    Xte = np.load(f"{DATA}/X_test.npy");  yte = np.load(f"{DATA}/y_test.npy")

    Str, Ste = summary(Xtr), summary(Xte)
    btr = (ytr == benign).astype(int)          # 1 = BENIGN, 0 = ATTACK
    bte = (yte == benign).astype(int)

    rf = RandomForestClassifier(n_estimators=150, max_depth=16,
                                class_weight="balanced", n_jobs=-1, random_state=42)
    rf.fit(Str, btr)
    # IMPORTANT: persist inference-safe settings.
    # n_jobs=-1 can be slower for single-sample inference due to threadpool overhead.
    rf.n_jobs = 1
    joblib.dump(rf, OUT)

    pred = rf.predict(Ste)
    # confusion in BENIGN/ATTACK terms
    cm = confusion_matrix(bte, pred, labels=[1, 0])  # rows/cols: BENIGN, ATTACK
    b_ok, b_miss = cm[0, 0], cm[0, 1]   # benign->benign, benign->attack(FP)
    a_miss, a_ok = cm[1, 0], cm[1, 1]   # attack->benign(FN), attack->attack
    size_mb = os.path.getsize(OUT) / 1e6
    print("=" * 60); print("  Tier-1 gate (BENIGN vs ATTACK) — TEST"); print("=" * 60)
    print(f"  Model size               : {size_mb:.2f} MB  (was 6760 MB)")
    print(f"  Benign correctly allowed : {b_ok}/{b_ok + b_miss}")
    print(f"  Benign FALSE POSITIVE    : {b_miss}  -> {100*b_miss/(b_ok+b_miss):.2f}%")
    print(f"  Attack recall (caught)   : {100*a_ok/(a_ok+a_miss):.2f}%  (missed {a_miss})")


if __name__ == "__main__":
    main()
