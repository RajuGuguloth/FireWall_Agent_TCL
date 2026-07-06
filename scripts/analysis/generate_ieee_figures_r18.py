#!/usr/bin/env python3
"""
Generate IEEE-ready figures for Hybrid-Sentinel R18.

Run from repository root:
    python scripts/analysis/generate_ieee_figures_r18.py

Outputs: results/ieee_figures/fig_r18_*.pdf and .png
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
import torch
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from sklearn.metrics import confusion_matrix

import config as cfg
from src.models.cnn_gru_v6 import CNNGRUClassifier

OUT = _ROOT / "results" / "ieee_figures"
OUT.mkdir(parents=True, exist_ok=True)

IEEE_RC = {
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.linewidth": 0.6,
    "axes.edgecolor": "#4A4A4A",
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "axes.grid": False,
    "axes.facecolor": "white",
}
plt.rcParams.update(IEEE_RC)

C_NAVY = "#243B63"
C_BLUE = "#4E79A7"
C_GREEN = "#59A14F"
C_RED = "#C44E52"
C_ORANGE = "#F28E2B"
C_TEAL = "#499894"
C_GREY = "#6E7781"
C_WHITE = "#FFFFFF"
C_LGREY = "#F4F6F8"
C_DKBLUE = "#0E1A33"

def _bar_edge(hex_color, darken=0.25):
    """Return a slightly darker version of a hex color for bar edges."""
    rgb = tuple(int(hex_color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
    return "#{:02x}{:02x}{:02x}".format(
        *[max(0, int(c * (1 - darken))) for c in rgb]
    )


def clean_axes(ax, grid_axis: str | None = "y"):
    ax.set_facecolor("white")
    if grid_axis:
        ax.grid(axis=grid_axis, color="#D9DEE5", linewidth=0.6, alpha=0.75)
        ax.set_axisbelow(True)
    ax.tick_params(colors="#333333", width=0.7, length=3)
    for spine in ax.spines.values():
        spine.set_color("#444444")
        spine.set_linewidth(0.8)


def save(fig, name: str):
    for ext in ("pdf", "png"):
        p = OUT / f"{name}.{ext}"
        fig.savefig(p, facecolor="white")
        print(f"  saved {p}")
    plt.close(fig)


def load_metrics():
    with open(_ROOT / "results" / "r18_tier_metrics.json") as f:
        return json.load(f)


def load_latency():
    p = _ROOT / "results" / "r18_latency_benchmark.json"
    if p.is_file():
        with open(p) as f:
            return json.load(f)
    return None


def tier2_predictions():
    """Run Tier-2 on test set for confusion matrix."""
    import joblib

    le = joblib.load(cfg.ENCODER_PATH)
    classes = list(le.classes_)
    Xte = np.load(os.path.join(cfg.SEQ_DIR, "X_test.npy"))
    yte = np.load(os.path.join(cfg.SEQ_DIR, "y_test.npy"))
    with open(cfg.TIER2_TEMP) as f:
        T = float(json.load(f)["temperature"])
    model = CNNGRUClassifier(num_classes=len(classes))
    model.load_state_dict(torch.load(cfg.TIER2_PTH, map_location="cpu"))
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(Xte), 512):
            lg = model(torch.FloatTensor(Xte[i : i + 512])) / T
            preds.extend(torch.softmax(lg, 1).argmax(1).numpy())
    return np.array(preds), yte, classes


# ── Fig 1: End-to-end orchestration pipeline ─────────────────────────
def fig_orchestration_pipeline():
    fig, ax = plt.subplots(figsize=(9.0, 4.8), facecolor="white")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.2)
    ax.axis("off")

    def box(x, y, w, h, text, color, fs=7.5):
        ax.add_patch(FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.06", linewidth=1.0,
            edgecolor=color, facecolor="#F8FAFC",
        ))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs,
                color="#1F2937", fontweight="bold", wrap=True)

    def arr(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=C_NAVY, lw=1.4))

    ax.text(6, 5.85, "Build and Deployment Pipeline",
            ha="center", fontsize=12, fontweight="bold", color=C_NAVY)
    ax.text(6, 5.52, "R18 data, training, export, serving, and runtime evaluation",
            ha="center", fontsize=8.5, color=C_GREY)

    # Row 1: Data
    box(1.3, 4.55, 1.9, 0.72, "PCAP\ncapture", C_TEAL)
    box(3.7, 4.55, 2.1, 0.72, "17 feature\nextraction", C_TEAL)
    box(6.3, 4.55, 2.2, 0.72, "20-packet\nwindows", C_TEAL)
    box(9.2, 4.55, 2.2, 0.72, "Train\nthree tiers", C_BLUE)
    for xs, xe in [(2.25, 2.65), (4.75, 5.2), (7.4, 8.1)]:
        arr(xs, 4.6, xe, 4.6)

    arr(6.3, 4.18, 6.3, 3.58)

    # Row 2: Export & serve
    box(2.5, 3.2, 2.3, 0.72, "ONNX export\n+ T3 calibrate", C_ORANGE)
    box(6.3, 3.2, 2.6, 0.72, "Cascade runtime\nsingle logic", C_ORANGE)
    box(10.0, 3.2, 2.3, 0.72, "FastAPI\nDashboard", C_GREEN)
    arr(3.65, 3.2, 5.0, 3.2)
    arr(7.6, 3.2, 8.85, 3.2)

    arr(6.3, 2.82, 6.3, 2.18)

    # Row 3: Runtime decision
    box(2.0, 1.65, 1.8, 0.65, "Tier-1\nGate", C_BLUE, 7.5)
    box(4.6, 1.65, 2.0, 0.65, "Tier-2\nCNN-GRU", C_BLUE, 7.5)
    box(7.2, 1.65, 2.0, 0.65, "Tier-3\nAnomaly", C_BLUE, 7.5)
    box(10.0, 1.65, 2.2, 0.65, "ALLOW\nFLAG\nBLOCK", C_GREEN, 7.5)
    arr(2.9, 1.65, 3.6, 1.65)
    arr(5.6, 1.65, 6.2, 1.65)
    arr(8.2, 1.65, 8.9, 1.65)

    ax.text(6, 0.55,
            "Runtime input: 20 x 17 feature window     Output: action, tier trace, latency",
            ha="center", fontsize=8, color=C_GREY)
    save(fig, "fig_r18_orchestration_pipeline")


# ── Fig 2: Runtime architecture flow ─────────────────────────────────
def fig_architecture_flow():
    fig, ax = plt.subplots(figsize=(5.2, 6.8), facecolor="white")
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 11.5)
    ax.axis("off")

    def rbox(cy, h, lines, bg, w=4.7):
        ax.add_patch(FancyBboxPatch(
            (3.0 - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0.08",
            linewidth=1.0, edgecolor=bg, facecolor="#F8FAFC",
        ))
        step = h / (len(lines) + 1)
        for i, (txt, bold) in enumerate(lines):
            ax.text(3.0, cy + h / 2 - (i + 1) * step, txt,
                    ha="center", va="center", fontsize=8 if not bold else 8.8,
                    fontweight="bold" if bold else "normal",
                    color=C_NAVY if bold else "#2F3A45")

    def diamond(cy, text):
        pts = np.array([[3.0, cy + 0.38], [3.65, cy], [3.0, cy - 0.38], [2.35, cy], [3.0, cy + 0.38]])
        ax.fill(pts[:, 0], pts[:, 1], color="#EEF3F8", edgecolor=C_BLUE, linewidth=1.0)
        ax.text(3.0, cy, text, ha="center", va="center", fontsize=7.5, fontweight="bold", color=C_NAVY)

    ax.text(3.0, 11.05, "R18 Runtime Cascade",
            ha="center", fontsize=12, fontweight="bold", color=C_NAVY)

    rbox(10.25, 0.65, [("Packet window: 20 x 17 features", True)], C_NAVY)
    ax.annotate("", xy=(3.0, 9.62), xytext=(3.0, 9.9),
                arrowprops=dict(arrowstyle="-|>", color=C_NAVY, lw=1.2))

    rbox(8.95, 0.95, [
        ("Tier-1: Random Forest Gate", True),
        ("Fast allow if P(BENIGN) >= 0.90", False),
        ("156 KB model, 3.66 ms avg", False),
    ], C_BLUE)
    ax.annotate("", xy=(3.0, 8.25), xytext=(3.0, 8.48),
                arrowprops=dict(arrowstyle="-|>", color=C_BLUE, lw=1.2))

    diamond(7.85, "Benign?")
    ax.text(4.38, 7.98, "ALLOW", fontsize=7.5, color=C_GREEN, fontweight="bold")
    ax.annotate("", xy=(5.15, 7.85), xytext=(3.65, 7.85),
                arrowprops=dict(arrowstyle="-|>", color=C_GREEN, lw=1.2))

    ax.text(2.0, 7.45, "ESCALATE", fontsize=7, color=C_ORANGE, fontweight="bold")
    ax.annotate("", xy=(3.0, 7.1), xytext=(3.0, 7.48),
                arrowprops=dict(arrowstyle="-|>", color=C_ORANGE, lw=1.2))

    rbox(6.45, 1.15, [
        ("Tier-2: CNN-GRU (ONNX)", True),
        ("Six classes including BENIGN", False),
        ("Block > 0.95, flag 0.80-0.95", False),
        ("0.18 ms avg ONNX inference", False),
    ], C_ORANGE)
    ax.annotate("", xy=(3.0, 5.68), xytext=(3.0, 5.88),
                arrowprops=dict(arrowstyle="-|>", color=C_RED, lw=1.2))

    rbox(5.1, 0.82, [
        ("Tier-3: Mahalanobis Anomaly", True),
        ("128-D embedding, zero-day flag", False),
    ], C_TEAL)
    ax.annotate("", xy=(3.0, 4.35), xytext=(3.0, 4.68),
                arrowprops=dict(arrowstyle="-|>", color=C_TEAL, lw=1.2))

    rbox(3.9, 0.7, [("Final action: ALLOW, FLAG, or BLOCK", True)], C_GREEN)

    ax.annotate("", xy=(5.15, 3.9), xytext=(5.15, 7.85),
                arrowprops=dict(arrowstyle="-|>", color=C_GREEN, lw=1.0,
                                connectionstyle="arc3,rad=0.15"))
    save(fig, "fig_r18_architecture_flow")


# ── Fig 3: Confusion matrix ──────────────────────────────────────────
def fig_confusion_matrix(preds, y_true, classes):
    cm = confusion_matrix(y_true, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(3.8, 3.5), facecolor="white")
    short = [c.replace("_", "\n") for c in classes]

    cmap = sns.color_palette("blend:#F7FBFF,#1B2A4A", as_cmap=True)
    im = ax.imshow(cm_norm, cmap=cmap, vmin=0, vmax=1, aspect="equal",
                   interpolation="nearest")

    n = len(classes)
    for i in range(n):
        for j in range(n):
            val = cm_norm[i, j]
            count = cm[i, j]
            txt_col = C_WHITE if val > 0.5 else C_DKBLUE
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7.5, fontweight="bold" if val > 0.9 else "normal",
                    color=txt_col)
            if count > 0 and val < 0.9:
                ax.text(j, i + 0.22, f"({count})", ha="center", va="center",
                        fontsize=5.5, color=txt_col, alpha=0.7)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short, fontsize=6.5)
    ax.set_yticklabels(short, fontsize=6.5)
    ax.set_xlabel("Predicted", fontsize=8, labelpad=6)
    ax.set_ylabel("True", fontsize=8, labelpad=6)
    ax.set_title("Tier-2 Normalized Confusion Matrix\n"
                 r"($n = 14{,}219$ held-out sequences)", fontsize=9, pad=8)
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right")

    cbar = fig.colorbar(im, ax=ax, shrink=0.75, aspect=20, pad=0.04)
    cbar.ax.tick_params(labelsize=6.5)
    cbar.outline.set_linewidth(0.4)

    for spine in ax.spines.values():
        spine.set_linewidth(0.4)

    save(fig, "fig_r18_confusion_matrix")


# ── Fig 4: Per-class metrics ─────────────────────────────────────────
def fig_per_class_metrics(metrics):
    pc = metrics["tier2_cnn_gru"]["per_class"]
    classes = list(pc.keys())
    x = np.arange(len(classes))
    w = 0.22
    prec = [pc[c]["precision"] for c in classes]
    rec = [pc[c]["recall"] for c in classes]
    f1 = [pc[c]["f1"] for c in classes]

    colors = [C_NAVY, C_BLUE, C_TEAL]
    labels = ["Precision", "Recall", "F1-Score"]

    fig, ax = plt.subplots(figsize=(5.8, 3.3), facecolor="white")
    for i, (vals, col, lbl) in enumerate(zip([prec, rec, f1], colors, labels)):
        offset = (i - 1) * w
        bars = ax.bar(x + offset, vals, w * 0.92, label=lbl,
                      color=col, edgecolor=_bar_edge(col), linewidth=0.5,
                      zorder=3)
        for bar, v in zip(bars, vals):
            if v < 0.99:
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.003,
                        f"{v:.3f}", ha="center", va="bottom",
                        fontsize=5, color=col, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in classes], fontsize=7)
    ax.set_ylim(0.88, 1.025)
    ax.set_ylabel("Score", fontsize=8)
    ax.set_title(f"Per-Class Metrics (Macro F1 = {metrics['tier2_cnn_gru']['f1_macro']:.4f})",
                 fontsize=10, pad=8)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=3,
              fontsize=7.5, frameon=False)
    ax.grid(axis="y", alpha=0.2, linewidth=0.4, linestyle="--", zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.subplots_adjust(bottom=0.28)
    save(fig, "fig_r18_per_class_metrics")


# ── Fig 5: Cascade funnel ────────────────────────────────────────────
def fig_cascade_funnel(metrics):
    stages = metrics["cascade_flow"]["stages"]
    pick = [
        "stage_0_input",
        "tier1_fast_allow",
        "tier1_escalate_to_tier2",
        "tier2_block",
        "tier2_flag",
        "tier3_flag_anomaly",
        "final_allow",
    ]
    labels_map = {
        "stage_0_input": "Input",
        "tier1_fast_allow": "T1 ALLOW",
        "tier1_escalate_to_tier2": "T1 Escalate",
        "tier2_block": "T2 BLOCK",
        "tier2_flag": "T2 FLAG",
        "tier3_flag_anomaly": "T3 FLAG",
        "final_allow": "Final ALLOW",
    }
    by_name = {s["name"]: s for s in stages}
    names = [labels_map[k] for k in pick]
    totals = [by_name[k]["total"] for k in pick]
    pcts = [by_name[k]["pct_of_input"] for k in pick]

    colors = [C_GREY, C_GREEN, C_ORANGE, C_RED, C_ORANGE, C_TEAL, C_GREEN]

    fig, ax = plt.subplots(figsize=(3.8, 3.0), facecolor="white")
    y = np.arange(len(names))
    bars = ax.barh(y, pcts, color=colors,
                   edgecolor=[_bar_edge(c) for c in colors],
                   linewidth=0.5, height=0.58, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("% of input sequences", fontsize=8)
    ax.set_title("Cascade Traffic Funnel (Test Set)", fontsize=9, pad=8)
    ax.invert_yaxis()
    for bar, tot, pct in zip(bars, totals, pcts):
        ax.text(bar.get_width() + 0.8, bar.get_y() + bar.get_height() / 2,
                f"{pct:.1f}%  ({tot:,})", va="center", fontsize=6, color=C_GREY)
    ax.set_xlim(0, 115)
    ax.grid(axis="x", alpha=0.15, linewidth=0.4, linestyle="--", zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    save(fig, "fig_r18_cascade_funnel")


# ── Fig 6: Latency benchmark ─────────────────────────────────────────
def fig_latency(lat):
    if not lat:
        print("  skip latency (no r18_latency_benchmark.json)")
        return
    r = lat["results"]
    names = ["Tier-1\nGate", "Tier-2\nONNX", "Tier-3\nOne-class", "Full\nCascade"]
    avgs = [r["tier1_gate"]["avg_ms"], r["tier2_onnx"]["avg_ms"],
            r["tier3_oneclass"]["avg_ms"], r["full_cascade"]["avg_ms"]]
    p99s = [r["tier1_gate"]["p99_ms"], r["tier2_onnx"]["p99_ms"],
            r["tier3_oneclass"]["p99_ms"], r["full_cascade"]["p99_ms"]]
    baseline = lat.get("g_scaler_baseline_avg_ms", 13.0)

    fig, ax = plt.subplots(figsize=(5.6, 3.3), facecolor="white")
    x = np.arange(len(names))
    w = 0.35
    b1 = ax.bar(x - w / 2, avgs, w, label="Avg (ms)", color=C_NAVY, alpha=0.92)
    b2 = ax.bar(x + w / 2, p99s, w, label="p99 (ms)", color=C_ORANGE, alpha=0.92)
    ax.axhline(baseline, color=C_RED, linestyle="--", linewidth=1.2,
               label=f"Baseline ({baseline} ms)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("Latency (ms)")
    ax.set_ylim(0, max(max(p99s), baseline) * 1.18)
    ax.set_title("Inference Latency")
    for bars, vals in [(b1, avgs), (b2, p99s)]:
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.25,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=6.5)
    ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.18),
              ncol=3, frameon=False)
    clean_axes(ax)
    fig.subplots_adjust(bottom=0.25)
    save(fig, "fig_r18_latency_benchmark")


# ── Fig 7: Class distribution ────────────────────────────────────────
def fig_class_distribution(metrics):
    pc = metrics["tier2_cnn_gru"]["per_class"]
    classes = list(pc.keys())
    support = [pc[c]["support"] for c in classes]
    colors = [C_GREEN if c == "BENIGN" else C_BLUE for c in classes]

    fig, ax = plt.subplots(figsize=(5.8, 3.2), facecolor="white")
    bars = ax.bar(range(len(classes)), support, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels([c.replace("_", "\n") for c in classes], fontsize=7)
    ax.set_ylabel("Test sequences")
    ax.set_title("Held-Out Test Set Distribution")
    ax.set_ylim(0, max(support) * 1.16)
    for bar, v in zip(bars, support):
        ax.text(bar.get_x() + bar.get_width() / 2, v + max(support) * 0.02,
                f"{v:,}", ha="center", fontsize=7)
    clean_axes(ax)
    save(fig, "fig_r18_class_distribution")


# ── Fig 8: R17 vs R18 benign handling ────────────────────────────────
def fig_r17_vs_r18():
    fig, ax = plt.subplots(figsize=(5.4, 3.0), facecolor="white")
    versions = ["R17\n(5-class,\nno BENIGN)", "R18\n(6-class,\nwith BENIGN)"]
    fpr = [100.0, 0.86]
    det = [100.0, 100.0]
    x = np.arange(2)
    w = 0.35
    bars_fpr = ax.bar(x - w / 2, fpr, w, label="Benign FPR (%)", color=C_RED, alpha=0.88)
    bars_det = ax.bar(x + w / 2, det, w, label="Attack detection (%)", color=C_GREEN, alpha=0.88)
    ax.set_xticks(x)
    ax.set_xticklabels(versions, fontsize=8)
    ax.set_ylim(0, 112)
    ax.set_ylabel("Rate (%)")
    ax.set_title("R18 Benign False-Positive Reduction")
    for bars, vals in [(bars_fpr, fpr), (bars_det, det)]:
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 2.0,
                    f"{val:.2f}%", ha="center", va="bottom", fontsize=7)
    ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.2),
              ncol=2, frameon=False)
    clean_axes(ax)
    fig.subplots_adjust(bottom=0.28)
    save(fig, "fig_r17_vs_r18_comparison")


def main():
    print("=" * 60)
    print("  IEEE Figure Generator — Hybrid-Sentinel R18")
    print("=" * 60)
    metrics = load_metrics()
    lat = load_latency()

    print("\n[1/8] Orchestration pipeline ...")
    fig_orchestration_pipeline()

    print("[2/8] Architecture flow ...")
    fig_architecture_flow()

    print("[3/8] Confusion matrix (live inference) ...")
    preds, yte, classes = tier2_predictions()
    fig_confusion_matrix(preds, yte, classes)

    print("[4/8] Per-class metrics ...")
    fig_per_class_metrics(metrics)

    print("[5/8] Cascade funnel ...")
    fig_cascade_funnel(metrics)

    print("[6/8] Latency benchmark ...")
    fig_latency(lat)

    print("[7/8] Class distribution ...")
    fig_class_distribution(metrics)

    print("[8/8] R17 vs R18 comparison ...")
    fig_r17_vs_r18()

    print(f"\nDone. Figures in: {OUT}")
    print("Use in IEEE report: \\includegraphics{fig_r18_*.pdf}")


if __name__ == "__main__":
    main()
