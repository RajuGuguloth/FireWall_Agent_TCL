#!/usr/bin/env python3
"""
Regenerate IEEE single-column confusion matrices with large, readable labels.

Designed for IEEEtran column width (~3.3–3.5 in). Outputs PDF+PNG to:
  results/ieee_figures/
  docs/r18/metrics/  (Track A)
  docs/ndn_poc/ieee/ (Track B)
  docs/ieee_overleaf_package/figures/
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.06,
    }
)

C_NAVY = "#1B2A4A"


def _short_label(name: str) -> str:
    mapping = {
        "BENIGN": "BENIGN",
        "BRUTE_FORCE": "BRUTE\nFORCE",
        "DDOS_HTTP_FLOOD": "DDOS\nHTTP",
        "DNS_TUNNELING": "DNS\nTUNNEL",
        "PORT_SCAN": "PORT\nSCAN",
        "SLOW_HTTP": "SLOW\nHTTP",
        "CACHE_POLLUTION": "CACHE\nPOLLUTION",
        "INTEREST_FLOODING": "INTEREST\nFLOOD",
    }
    return mapping.get(name, name.replace("_", "\n"))


def plot_clear_cm(
    cm: np.ndarray,
    labels: list[str],
    title: str,
    stem: str,
    out_dirs: list[Path],
    *,
    show_counts: bool = False,
    figsize: tuple[float, float] = (3.35, 3.45),
) -> None:
    """Single-column normalized confusion matrix with readable labels."""
    cm = np.asarray(cm, dtype=float)
    row = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row, where=row > 0)
    n = len(labels)

    # Extra bottom margin for wrapped / rotated x labels
    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0, aspect="equal")

    shorts = [_short_label(lb) for lb in labels]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))

    if n >= 6:
        # Rotate x labels to avoid collision in single column
        ax.set_xticklabels(shorts, fontsize=6.5, linespacing=0.9, rotation=35, ha="right")
        ax.set_yticklabels(shorts, fontsize=6.5, linespacing=0.9)
        cell_fs = 7.5
        count_mode = False
    else:
        ax.set_xticklabels(shorts, fontsize=7.5, linespacing=0.95)
        ax.set_yticklabels(shorts, fontsize=7.5, linespacing=0.95)
        cell_fs = 9
        count_mode = show_counts

    ax.set_xlabel("Predicted label", fontsize=9, labelpad=6)
    ax.set_ylabel("True label", fontsize=9, labelpad=4)
    ax.set_title(title, fontsize=10, pad=8, color=C_NAVY, fontweight="bold")

    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.6)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(n):
        for j in range(n):
            v = float(cm_norm[i, j])
            color = "white" if v >= 0.50 else C_NAVY
            # Hide near-zero clutter; keep diagonal and meaningful off-diagonal
            if v < 0.005 and i != j:
                txt = "·"
                fs = 7
                weight = "normal"
            elif count_mode:
                txt = f"{v:.2f}\n({int(cm[i, j]):,})"
                fs = 8
                weight = "bold"
            else:
                txt = f"{v:.2f}"
                fs = cell_fs
                weight = "bold"
            ax.text(j, i, txt, ha="center", va="center", fontsize=fs,
                    fontweight=weight, color=color, linespacing=1.1)

    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.05)
    cbar.set_label("Recall (row-norm.)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()
    for d in out_dirs:
        d.mkdir(parents=True, exist_ok=True)
        for ext in ("pdf", "png"):
            path = d / f"{stem}.{ext}"
            fig.savefig(path, format=ext, facecolor="white")
            print(f"  saved {path}")
    plt.close(fig)


def ndn_cm() -> None:
    m = json.loads((_ROOT / "docs/ndn_poc/ndn_metrics.json").read_text())
    labels = m["confusion_matrix"]["labels"]
    cm = np.array(m["confusion_matrix"]["matrix"], dtype=float)
    outs = [
        _ROOT / "results/ieee_figures/ndn",
        _ROOT / "docs/ndn_poc/ieee",
        _ROOT / "docs/ieee_overleaf_package/figures",
        _ROOT / "results/ieee_figures",
    ]
    plot_clear_cm(
        cm, labels,
        title="NDN Confusion Matrix (held-out)",
        stem="fig_ndn_confusion_matrix",
        out_dirs=outs,
        show_counts=True,
        figsize=(3.35, 3.20),
    )


def r18_cm_from_model() -> None:
    """Recompute Tier-2 CM from held-out test + R18 weights."""
    import joblib
    import torch
    from sklearn.metrics import confusion_matrix

    import config as cfg
    from src.models.cnn_gru_v6 import CNNGRUClassifier

    enc = joblib.load(cfg.ENCODER_PATH)
    classes = list(enc.classes_)
    X = np.load(Path(cfg.SEQ_DIR) / "X_test.npy")
    y = np.load(Path(cfg.SEQ_DIR) / "y_test.npy")
    if y.dtype.kind in ("U", "S", "O"):
        y_idx = enc.transform(y)
    else:
        y_idx = y.astype(int)

    with open(cfg.TIER2_TEMP) as f:
        temperature = float(json.load(f)["temperature"])

    model = CNNGRUClassifier(num_classes=len(classes))
    model.load_state_dict(torch.load(cfg.TIER2_PTH, map_location="cpu"))
    model.eval()

    preds = []
    with torch.no_grad():
        for i in range(0, len(X), 512):
            logits = model(torch.FloatTensor(X[i : i + 512])) / temperature
            preds.extend(torch.softmax(logits, 1).argmax(1).numpy())
    preds = np.asarray(preds)

    cm = confusion_matrix(y_idx, preds, labels=list(range(len(classes))))
    outs = [
        _ROOT / "results/ieee_figures",
        _ROOT / "docs/r18/metrics",
        _ROOT / "docs/ieee_overleaf_package/figures",
        _ROOT / "results/submission_figures",
    ]
    plot_clear_cm(
        cm, classes,
        title="Tier-2 Confusion Matrix",
        stem="fig_r18_confusion_matrix",
        out_dirs=outs,
        show_counts=False,
        figsize=(3.40, 3.35),
    )
    src_pdf = _ROOT / "docs/ieee_overleaf_package/figures/fig_r18_confusion_matrix.pdf"
    src_png = _ROOT / "docs/ieee_overleaf_package/figures/fig_r18_confusion_matrix.png"
    for name in ("confusion_matrix", "fig_submission_tier2_confusion_matrix"):
        for src, ext in ((src_pdf, "pdf"), (src_png, "png")):
            if not src.is_file():
                continue
            for d in outs:
                dst = d / f"{name}.{ext}"
                shutil.copy2(src, dst)
                print(f"  copied {dst}")

def also_refresh_ndn_bars() -> None:
    """Clearer single-column NDN summary / per-class bars."""
    m = json.loads((_ROOT / "docs/ndn_poc/ndn_metrics.json").read_text())
    outs = [
        _ROOT / "results/ieee_figures/ndn",
        _ROOT / "docs/ndn_poc/ieee",
        _ROOT / "docs/ieee_overleaf_package/figures",
    ]

    # Per-class P/R/F1
    classes = [c for c in m["classes"] if c in m["per_class"]]
    metrics = ["precision", "recall", "f1-score"]
    colors = ["#4E79A7", "#499894", "#59A14F"]
    x = np.arange(len(classes))
    width = 0.26
    fig, ax = plt.subplots(figsize=(3.35, 2.7), facecolor="white")
    for i, (met, col) in enumerate(zip(metrics, colors)):
        vals = [m["per_class"][c][met] for c in classes]
        ax.bar(x + (i - 1) * width, vals, width, label=met.replace("-score", "").capitalize(),
               color=col, edgecolor="white")
        for xi, v in zip(x + (i - 1) * width, vals):
            ax.text(xi, v + 0.004, f"{v:.2f}", ha="center", va="bottom", fontsize=6.5)
    ax.set_ylim(0.96, 1.015)
    ax.set_xticks(x)
    ax.set_xticklabels([_short_label(c).replace("\n", " ") for c in classes], fontsize=7.5)
    ax.set_ylabel("Score", fontsize=9)
    ax.set_title("NDN Per-Class Metrics", fontsize=10, color=C_NAVY, fontweight="bold")
    ax.legend(loc="lower right", fontsize=7, framealpha=0.95)
    ax.axhline(1.0, color="#CCCCCC", lw=0.6, ls="--")
    fig.tight_layout()
    for d in outs:
        d.mkdir(parents=True, exist_ok=True)
        for ext in ("pdf", "png"):
            fig.savefig(d / f"fig_ndn_per_class_metrics.{ext}", format=ext, facecolor="white")
    plt.close(fig)
    print("  refreshed fig_ndn_per_class_metrics")

    # Summary
    names = ["Accuracy", "Macro-F1", "Attack det.", "Benign FPR"]
    vals = [
        m["accuracy"] * 100,
        m["macro_f1"] * 100,
        m["attack_detection_rate"] * 100,
        m["benign_false_positive_rate"] * 100,
    ]
    colors = ["#243B63", "#4E79A7", "#59A14F", "#F28E2B"]
    fig, ax = plt.subplots(figsize=(3.35, 2.55), facecolor="white")
    bars = ax.bar(names, vals, color=colors, edgecolor="white", width=0.72)
    ax.set_ylabel("Percent (%)", fontsize=9)
    ax.set_title(f"NDN Summary (n={m['n_test']:,})", fontsize=10, color=C_NAVY, fontweight="bold")
    ax.set_ylim(0, 112)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.2f}%",
                ha="center", fontsize=8, fontweight="bold")
    plt.setp(ax.get_xticklabels(), fontsize=8)
    fig.tight_layout()
    for d in outs:
        for ext in ("pdf", "png"):
            fig.savefig(d / f"fig_ndn_summary_metrics.{ext}", format=ext, facecolor="white")
    plt.close(fig)
    print("  refreshed fig_ndn_summary_metrics")


def main() -> None:
    print("=== Clear single-column IEEE CMs ===")
    ndn_cm()
    also_refresh_ndn_bars()
    print("=== Track A Tier-2 CM (from R18 model) ===")
    r18_cm_from_model()
    print("Done.")


if __name__ == "__main__":
    main()
