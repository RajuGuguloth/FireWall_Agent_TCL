"""
Train the NDN proof-of-concept detector and report real metrics.

Mirrors the production pipeline conventions: StandardScaler + clip on the
per-packet features, then a window "summary" representation (mean, std, last
packet) identical in spirit to the Tier-1 gate (train_tier1_gate_v6.py). A
RandomForest multiclass classifier separates BENIGN / INTEREST_FLOODING /
CACHE_POLLUTION on NDN forwarder-state features.

Outputs (to models/ndn/ and results/ndn/):
    ndn_poc_rf.pkl            trained classifier
    ndn_poc_scaler.pkl        fitted scaler
    ndn_metrics.json          accuracy, macro-F1, per-class report
    ndn_confusion_matrix.png  confusion matrix figure

Usage:
    python -m ndn_poc.train_poc --data data/ndn
"""
from __future__ import annotations

import argparse
import json
import os

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

CLIP_VAL = 5.0


def summary(X: np.ndarray) -> np.ndarray:
    """(N, window, F) -> (N, 3F): mean, std, last packet. Matches Tier-1 gate."""
    return np.concatenate([X.mean(1), X.std(1), X[:, -1, :]], axis=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/ndn")
    ap.add_argument("--models_out", type=str, default="models/ndn")
    ap.add_argument("--results_out", type=str, default="results/ndn")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.models_out, exist_ok=True)
    os.makedirs(args.results_out, exist_ok=True)

    X = np.load(os.path.join(args.data, "ndn_windows.npy"))
    y = np.load(os.path.join(args.data, "ndn_labels.npy"))
    classes = sorted(np.unique(y).tolist())

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=args.seed
    )

    # scale per-packet features (fit on train packets only), then clip
    F = X.shape[2]
    scaler = StandardScaler().fit(Xtr.reshape(-1, F))

    def prep(A: np.ndarray) -> np.ndarray:
        flat = scaler.transform(A.reshape(-1, F))
        flat = np.clip(flat, -CLIP_VAL, CLIP_VAL)
        return summary(flat.reshape(A.shape))

    Str, Ste = prep(Xtr), prep(Xte)

    clf = RandomForestClassifier(
        n_estimators=200, max_depth=18, class_weight="balanced",
        n_jobs=-1, random_state=args.seed,
    )
    clf.fit(Str, ytr)
    clf.n_jobs = 1

    pred = clf.predict(Ste)
    acc = accuracy_score(yte, pred)
    macro_f1 = f1_score(yte, pred, average="macro")
    report = classification_report(yte, pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(yte, pred, labels=classes)

    joblib.dump(clf, os.path.join(args.models_out, "ndn_poc_rf.pkl"))
    joblib.dump(scaler, os.path.join(args.models_out, "ndn_poc_scaler.pkl"))

    # attack detection rate: any attack window not predicted BENIGN
    attack_mask = yte != "BENIGN"
    attack_detected = (pred[attack_mask] != "BENIGN").mean() if attack_mask.any() else 0.0
    benign_mask = yte == "BENIGN"
    benign_fp = (pred[benign_mask] != "BENIGN").mean() if benign_mask.any() else 0.0

    metrics = {
        "classes": classes,
        "n_train": int(len(ytr)),
        "n_test": int(len(yte)),
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "attack_detection_rate": round(float(attack_detected), 4),
        "benign_false_positive_rate": round(float(benign_fp), 4),
        "per_class": {
            c: {k: round(float(report[c][k]), 4) for k in ("precision", "recall", "f1-score")}
            for c in classes if c in report
        },
        "confusion_matrix": {"labels": classes, "matrix": cm.tolist()},
    }
    with open(os.path.join(args.results_out, "ndn_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    _plot_cm(cm, classes, os.path.join(args.results_out, "ndn_confusion_matrix.png"))

    print("=" * 64)
    print("  NDN proof-of-concept detector — HELD-OUT TEST")
    print("=" * 64)
    print(f"  classes                  : {classes}")
    print(f"  train / test windows     : {len(ytr)} / {len(yte)}")
    print(f"  accuracy                 : {acc:.4f}")
    print(f"  macro-F1                 : {macro_f1:.4f}")
    print(f"  attack detection rate    : {attack_detected:.4f}")
    print(f"  benign false-positive    : {benign_fp:.4f}")
    print("  per-class F1             :")
    for c in classes:
        if c in report:
            print(f"    {c:<18}: P={report[c]['precision']:.3f} "
                  f"R={report[c]['recall']:.3f} F1={report[c]['f1-score']:.3f}")
    print(f"  confusion matrix labels  : {classes}")
    for row_label, row in zip(classes, cm):
        print(f"    {row_label:<18}: {row.tolist()}")
    print(f"  saved metrics -> {args.results_out}/ndn_metrics.json")
    print(f"  saved figure  -> {args.results_out}/ndn_confusion_matrix.png")


def _plot_cm(cm: np.ndarray, classes: list, path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=30, ha="right", fontsize=8)
    ax.set_yticklabels(classes, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("NDN PoC — Confusion Matrix (held-out)")
    thresh = cm.max() / 2 if cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
