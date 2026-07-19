#!/usr/bin/env python3
"""
Unified Hybrid-Sentinel architecture for IEEE: NDN motivation + dual validation tracks.

Honest framing:
  - Method shared: behavioural windows + ML detection at the edge
  - Track A (R18): production 3-tier cascade on IP Docker PCAPs
  - Track B (NDN PoC): Interest Flooding / Cache Pollution on simulated PIT+CS

Outputs:
  results/ieee_figures/fig_unified_system_architecture.{pdf,png}
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

_ROOT = Path(__file__).resolve().parents[2]
OUT = _ROOT / "results" / "ieee_figures"

IEEE_RC = {
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}
plt.rcParams.update(IEEE_RC)

C_NAVY = "#243B63"
C_BLUE = "#4E79A7"
C_GREEN = "#59A14F"
C_RED = "#C44E52"
C_ORANGE = "#F28E2B"
C_TEAL = "#499894"
C_PURPLE = "#B07AA1"
C_GREY = "#6E7781"
C_LIGHT = "#F7F9FC"


def _box(ax, x, y, w, h, title, body, edge, fill=C_LIGHT, title_size=8.5, body_size=6.8):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            linewidth=1.2, edgecolor=edge, facecolor=fill, zorder=2,
        )
    )
    cx, cy = x + w / 2, y + h / 2
    if body:
        ax.text(cx, cy + h * 0.22, title, ha="center", va="center",
                fontsize=title_size, fontweight="bold", color=C_NAVY, zorder=3)
        ax.text(cx, cy - h * 0.18, body, ha="center", va="center",
                fontsize=body_size, color="#344054", linespacing=1.25, zorder=3)
    else:
        ax.text(cx, cy, title, ha="center", va="center",
                fontsize=title_size, fontweight="bold", color=C_NAVY, zorder=3)


def _arrow(ax, x1, y1, x2, y2, color=C_NAVY, lw=1.2):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle="-|>", mutation_scale=10,
            color=color, lw=lw, zorder=3,
        )
    )


def main() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.4), facecolor="white")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis("off")

    ax.text(6.0, 9.65, "Hybrid-Sentinel: Dual-Track Architecture",
            ha="center", va="center", fontsize=12, fontweight="bold", color=C_NAVY)
    ax.text(
        6.0, 9.25,
        "Shared behavioural-window methodology  ·  IP production cascade  ·  NDN forwarder PoC",
        ha="center", va="center", fontsize=7.5, color=C_GREY,
    )

    # Row 1: NDN motivation
    _box(ax, 0.4, 7.55, 3.4, 1.35,
         "NDN Edge Motivation",
         "Interest / Data units\nPIT + Content Store (CS)\nName-based forwarding",
         C_TEAL, fill="#EDF7F6")
    _box(ax, 4.3, 7.55, 3.4, 1.35,
         "vs  Signature IDS (Snort)",
         "Static rules miss zero-days\nHigh rule maintenance cost\nWeak on novel NDN attacks",
         C_RED, fill="#FCF0F0")
    _box(ax, 8.2, 7.55, 3.4, 1.35,
         "Design Choice",
         "Cheap gate on most traffic\nDeep / specialised ML on suspicion\nAuditable ALLOW / FLAG / BLOCK",
         C_BLUE, fill="#EEF3FA")

    _arrow(ax, 3.8, 8.2, 4.3, 8.2, C_GREY)
    _arrow(ax, 7.7, 8.2, 8.2, 8.2, C_GREY)

    # Shared method band
    ax.add_patch(
        FancyBboxPatch(
            (0.4, 6.35), 11.2, 0.95,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            linewidth=1.4, edgecolor=C_NAVY, facecolor="#F4F6FA", zorder=2,
        )
    )
    ax.text(6.0, 6.95, "Shared Detection Method", ha="center", va="center",
            fontsize=9, fontweight="bold", color=C_NAVY, zorder=3)
    ax.text(
        6.0, 6.58,
        "17 behavioural features  →  20-packet windows  →  ML classifier (confidence-gated)  →  auditable decision",
        ha="center", va="center", fontsize=7.2, color="#344054", zorder=3,
    )
    _arrow(ax, 6.0, 7.55, 6.0, 7.35, C_NAVY)

    # Track A — R18
    ax.add_patch(
        FancyBboxPatch(
            (0.35, 0.35), 5.5, 5.7,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=1.3, edgecolor=C_BLUE, facecolor="#FBFCFF", zorder=1,
        )
    )
    ax.text(3.1, 5.75, "Track A — R18 Production (IP lab)",
            ha="center", va="center", fontsize=9, fontweight="bold", color=C_BLUE, zorder=3)
    ax.text(3.1, 5.40, "Docker PCAP  ·  TCP/IP features  ·  6 classes",
            ha="center", va="center", fontsize=6.8, color=C_GREY, zorder=3)

    _box(ax, 0.65, 4.15, 4.9, 0.95,
         "Tier-1  Random Forest gate",
         "P(BENIGN) ≥ 0.90 → fast ALLOW   ·   156 KB   ·   3.66 ms",
         C_BLUE)
    _box(ax, 0.65, 2.95, 4.9, 0.95,
         "Tier-2  CNN-GRU (ONNX)",
         "6 classes + TEMP scaling   ·   BLOCK > 0.95 / FLAG ≥ 0.80",
         C_ORANGE)
    _box(ax, 0.65, 1.75, 4.9, 0.95,
         "Tier-3  Mahalanobis anomaly",
         "128-D embedding novelty → FLAG as ANOMALY (zero-day)",
         C_PURPLE)
    _box(ax, 0.65, 0.55, 4.9, 0.95,
         "E2E (held-out n=14,219)",
         "100% attack detection  ·  1.11% benign FPR  ·  4.19 ms cascade",
         C_GREEN, fill="#F1F9F2")

    for y1, y2 in ((4.15, 3.90), (2.95, 2.70), (1.75, 1.50)):
        _arrow(ax, 3.1, y1, 3.1, y2, C_NAVY)

    # Track B — NDN PoC
    ax.add_patch(
        FancyBboxPatch(
            (6.15, 0.35), 5.5, 5.7,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=1.3, edgecolor=C_TEAL, facecolor="#FAFDFC", zorder=1,
        )
    )
    ax.text(8.9, 5.75, "Track B — NDN Forwarder PoC",
            ha="center", va="center", fontsize=9, fontweight="bold", color=C_TEAL, zorder=3)
    ax.text(8.9, 5.40, "Simulated PIT + CS  ·  NDN-native features  ·  3 classes",
            ha="center", va="center", fontsize=6.8, color=C_GREY, zorder=3)

    _box(ax, 6.45, 4.15, 4.9, 0.95,
         "NDN traffic sources",
         "Benign consumers (Zipf)  +  Interest Flood / Cache Pollution",
         C_TEAL, fill="#EDF7F6")
    _box(ax, 6.45, 2.95, 4.9, 0.95,
         "Monitored router R",
         "PIT occupancy · CS hit ratio · name entropy · unsatisfied Interests",
         C_NAVY, fill="#EEF2F7")
    _box(ax, 6.45, 1.75, 4.9, 0.95,
         "NDN RF classifier",
         "Same window size (20×17) · BENIGN / FLOOD / POLLUTION",
         C_ORANGE)
    _box(ax, 6.45, 0.55, 4.9, 0.95,
         "Held-out (n=51,656)",
         "99.58% acc  ·  macro-F1 0.995  ·  0.67% benign FPR",
         C_GREEN, fill="#F1F9F2")

    for y1, y2 in ((4.15, 3.90), (2.95, 2.70), (1.75, 1.50)):
        _arrow(ax, 8.9, y1, 8.9, y2, C_TEAL)

    # Bridge annotation
    ax.annotate(
        "", xy=(6.15, 3.0), xytext=(5.85, 3.0),
        arrowprops=dict(arrowstyle="<->", color=C_GREY, lw=1.0),
    )
    ax.text(6.0, 3.25, "method\ntransfer", ha="center", va="bottom",
            fontsize=6.2, color=C_GREY, fontstyle="italic")

    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = OUT / f"fig_unified_system_architecture.{ext}"
        fig.savefig(path, format=ext, facecolor="white")
        print(f"saved {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
