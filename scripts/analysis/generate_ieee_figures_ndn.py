#!/usr/bin/env python3
"""
Generate IEEE-ready figures for the NDN proof-of-concept (ndn_poc/).

Reads metrics from docs/ndn_poc/ndn_metrics.json (or results/ndn/ndn_metrics.json).

Run from repository root:
    python scripts/analysis/generate_ieee_figures_ndn.py

Outputs:
    results/ieee_figures/ndn/fig_ndn_*.pdf and .png
    docs/ndn_poc/ieee/fig_ndn_*.pdf and .png  (copies for paper handover)
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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = _ROOT / "results" / "ieee_figures" / "ndn"
DOCS_OUT = _ROOT / "docs" / "ndn_poc" / "ieee"

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
C_PURPLE = "#9C755F"
C_GREY = "#6E7781"


def _load_metrics() -> dict:
    for p in (
        _ROOT / "docs" / "ndn_poc" / "ndn_metrics.json",
        _ROOT / "results" / "ndn" / "ndn_metrics.json",
    ):
        if p.is_file():
            with open(p) as f:
                return json.load(f)
    raise FileNotFoundError(
        "ndn_metrics.json not found. Run: python -m ndn_poc.generate_dataset && "
        "python -m ndn_poc.train_poc"
    )


def _save(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    DOCS_OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = OUT / f"{stem}.{ext}"
        fig.savefig(path, format=ext, facecolor="white")
        shutil.copy2(path, DOCS_OUT / f"{stem}.{ext}")
    plt.close(fig)
    print(f"  saved {stem}.pdf / .png")


def fig_confusion_matrix(m: dict) -> None:
    labels = m["confusion_matrix"]["labels"]
    cm = np.array(m["confusion_matrix"]["matrix"], dtype=float)
    row_sum = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sum, where=row_sum > 0)

    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    short = [lb.replace("_", "\n") for lb in labels]
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(short, fontsize=7)
    ax.set_yticklabels(short, fontsize=7)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("NDN PoC — Normalized Confusion Matrix (held-out)")

    thresh = 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            txt = f"{int(cm[i, j])}\n({cm_norm[i, j]:.2f})"
            ax.text(
                j, i, txt, ha="center", va="center", fontsize=6.5,
                color="white" if cm_norm[i, j] > thresh else "black",
            )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Recall (row-normalized)", fontsize=7)
    fig.tight_layout()
    _save(fig, "fig_ndn_confusion_matrix")


def fig_per_class_metrics(m: dict) -> None:
    classes = [c for c in m["classes"] if c in m["per_class"]]
    metrics = ["precision", "recall", "f1-score"]
    colors = [C_BLUE, C_TEAL, C_GREEN]
    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    for i, (met, col) in enumerate(zip(metrics, colors)):
        vals = [m["per_class"][c][met] for c in classes]
        ax.bar(x + (i - 1) * width, vals, width, label=met.capitalize(), color=col, edgecolor="white")

    ax.set_ylim(0.9, 1.02)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", " ") for c in classes], rotation=12, ha="right", fontsize=7)
    ax.set_ylabel("Score")
    ax.set_title("NDN PoC — Per-Class Metrics (held-out)")
    ax.legend(loc="lower right", fontsize=7, framealpha=0.95)
    ax.axhline(1.0, color="#CCCCCC", linewidth=0.5, linestyle="--")
    fig.tight_layout()
    _save(fig, "fig_ndn_per_class_metrics")


def fig_summary_bars(m: dict) -> None:
    names = ["Accuracy", "Macro-F1", "Attack det.", "Benign FPR"]
    vals = [
        m["accuracy"] * 100,
        m["macro_f1"] * 100,
        m["attack_detection_rate"] * 100,
        m["benign_false_positive_rate"] * 100,
    ]
    colors = [C_NAVY, C_BLUE, C_GREEN, C_ORANGE]

    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    bars = ax.bar(names, vals, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_ylabel("Percent (%)")
    ax.set_title(f"NDN PoC — Summary (n_test = {m['n_test']:,})")
    ax.set_ylim(0, 105)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.2f}%", ha="center", fontsize=7)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right", fontsize=7)
    fig.tight_layout()
    _save(fig, "fig_ndn_summary_metrics")


def fig_architecture() -> None:
    """NDN simulator topology for IEEE — single monitored router."""
    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("NDN PoC — Simulated Forwarder Topology")

    def box(x, y, w, h, text, color):
        p = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor=color, edgecolor=C_NAVY, linewidth=0.8,
        )
        ax.add_patch(p)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7, color="white", fontweight="bold")

    box(0.3, 2.2, 1.8, 1.2, "Benign\nconsumers", C_GREEN)
    box(0.3, 0.5, 1.8, 1.2, "Attackers\n(Flood / Pollute)", C_RED)
    box(3.8, 1.2, 2.4, 2.4, "Monitored\nrouter R\nPIT + CS", C_NAVY)
    box(7.2, 1.6, 2.2, 1.6, "Content\nproducer", C_TEAL)

    for y0 in (2.8, 1.1):
        ax.add_patch(FancyArrowPatch((2.1, y0), (3.8, 2.4), arrowstyle="->", color=C_GREY, lw=1.0))
    ax.add_patch(FancyArrowPatch((6.2, 2.4), (7.2, 2.4), arrowstyle="->", color=C_GREY, lw=1.0))

    ax.text(5.0, 0.35, "17 NDN-native features → 20-packet windows → RF classifier", ha="center", fontsize=7, color=C_NAVY)
    fig.tight_layout()
    _save(fig, "fig_ndn_architecture")


def write_latex_snippet(m: dict) -> None:
    snippet = DOCS_OUT / "ndn_figures_latex.tex"
    snippet.write_text(
        r"""% Paste into IEEE LaTeX (adjust path if needed)
\subsection{NDN Forwarder Proof-of-Concept}
Separate from the IP-validated R18 cascade, we simulate a single NDN router
with Pending Interest Table (PIT) and Content Store (CS), reproducing
\textsc{Interest Flooding} and \textsc{Cache Pollution} attacks.
A Random Forest on 20-packet windows of 17 NDN-native features achieves
"""
        + f"{m['macro_f1']:.4f} macro-F1 on {m['n_test']:,} held-out windows "
        + f"({m['attack_detection_rate']*100:.2f}\\% attack detection, "
        + f"{m['benign_false_positive_rate']*100:.2f}\\% benign FPR).\n\n"
        + r"""\begin{figure}[!t]
\centering
\includegraphics[width=\linewidth]{docs/ndn_poc/ieee/fig_ndn_architecture.pdf}
\caption{NDN PoC simulation topology (PIT + Content Store).}
\label{fig:ndn_arch}
\end{figure}

\begin{figure}[!t]
\centering
\includegraphics[width=0.48\linewidth]{docs/ndn_poc/ieee/fig_ndn_confusion_matrix.pdf}
\hfill
\includegraphics[width=0.48\linewidth]{docs/ndn_poc/ieee/fig_ndn_per_class_metrics.pdf}
\caption{NDN PoC held-out results: normalized confusion matrix (left) and per-class metrics (right).}
\label{fig:ndn_results}
\end{figure}

\begin{figure}[!t]
\centering
\includegraphics[width=0.75\linewidth]{docs/ndn_poc/ieee/fig_ndn_summary_metrics.pdf}
\caption{NDN PoC summary metrics on held-out test windows.}
\label{fig:ndn_summary}
\end{figure}
""",
        encoding="utf-8",
    )
    print(f"  saved LaTeX snippet -> {snippet}")


def main() -> None:
    print("=" * 60)
    print("  IEEE NDN PoC Figure Generator")
    print("=" * 60)
    m = _load_metrics()
    print(f"  metrics: macro-F1={m['macro_f1']}, n_test={m['n_test']:,}")
    fig_architecture()
    fig_confusion_matrix(m)
    fig_per_class_metrics(m)
    fig_summary_bars(m)
    write_latex_snippet(m)
    print(f"\nDone. Figures in:\n  {OUT}\n  {DOCS_OUT}")


if __name__ == "__main__":
    main()
