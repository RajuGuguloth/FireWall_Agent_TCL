"""
plot_latency_benchmark.py
─────────────────────────────────────────────────────────────────────
Hybrid-Sentinel AI Firewall — Chapter 5 Figure
Dual-Axis Bar Chart: Latency (ms) + Throughput (PPS)
Data source: results/latency_benchmark.json (run: 2026-03-27)

Output: results/proof_of_work_visuals/fig_latency_benchmark.pdf
        results/proof_of_work_visuals/fig_latency_benchmark.png
─────────────────────────────────────────────────────────────────────
"""

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ── Output directory ──────────────────────────────────────────────
OUT_DIR = os.path.join("results", "proof_of_work_visuals")
os.makedirs(OUT_DIR, exist_ok=True)

# ── IITM-Inspired Color Palette ───────────────────────────────────
#   Primary:  IITM Deep Maroon / Crimson
#   Secondary: Navy Blue / Steel Blue
#   Accent:   Warm White / Light Grey
IITM_MAROON   = "#8B0000"      # Deep Crimson — Min latency bars
IITM_RED      = "#C0392B"      # Vivid Red — Avg latency bars
IITM_ORANGE   = "#E67E22"      # Amber — p95 latency bars
IITM_NAVY     = "#1A2F5A"      # Deep Navy — PPS line
IITM_BLUE     = "#2980B9"      # Steel Blue — PPS markers
GRID_COLOR    = "#E8E8E8"
BG_COLOR      = "#FAFAFA"
TEXT_COLOR    = "#1C1C1C"

# ── Data from results/latency_benchmark.json ──────────────────────
labels = ["Mamba\n(Baseline)", "Tier-1\nRandom Forest", "Tier-2\nCNN-GRU", "T1+T2\nPipeline"]
x = np.arange(len(labels))

# Latency (ms)
lat_min = [0.04,   4.2523, 0.9327, 4.2590]
lat_avg = [12.57,  4.4830, 1.0527, 4.5729]
lat_p95 = [38.43,  4.6861, 1.1033, 5.6877]

# Throughput
pps     = [80,     223,    949,    218]

# ── Figure Setup ──────────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family"     : "DejaVu Sans",
    "font.size"       : 11,
    "axes.titlesize"  : 13,
    "axes.labelsize"  : 11,
    "xtick.labelsize" : 10,
    "ytick.labelsize" : 10,
    "axes.spines.top" : False,
    "figure.dpi"      : 150,
})

fig = plt.figure(figsize=(12, 6.5), facecolor=BG_COLOR)
fig.patch.set_facecolor(BG_COLOR)
ax1 = fig.add_subplot(111)
ax2 = ax1.twinx()

ax1.set_facecolor(BG_COLOR)

# ── Bar positions ─────────────────────────────────────────────────
bar_w  = 0.22
gap    = 0.02
pos_min = x - bar_w - gap
pos_avg = x
pos_p95 = x + bar_w + gap

# ── LEFT AXIS: Latency Bars ───────────────────────────────────────
bars_min = ax1.bar(pos_min, lat_min, width=bar_w,
                   color=IITM_MAROON, alpha=0.92, label="Min Latency",
                   zorder=3, edgecolor="white", linewidth=0.6)

bars_avg = ax1.bar(pos_avg, lat_avg, width=bar_w,
                   color=IITM_RED, alpha=0.92, label="Avg Latency",
                   zorder=3, edgecolor="white", linewidth=0.6)

bars_p95 = ax1.bar(pos_p95, lat_p95, width=bar_w,
                   color=IITM_ORANGE, alpha=0.88, label="p95 Latency",
                   zorder=3, edgecolor="white", linewidth=0.6)

# ── RIGHT AXIS: PPS Line ──────────────────────────────────────────
ax2.plot(x, pps, color=IITM_NAVY, linewidth=2.4,
         marker="D", markersize=9, markerfacecolor=IITM_BLUE,
         markeredgecolor="white", markeredgewidth=1.4,
         label="Throughput (PPS)", zorder=5, linestyle="--")

# ── Value Labels on Bars ──────────────────────────────────────────
def label_bars(bars, fmt="{:.2f}", offset_frac=0.015, color="#fff"):
    """Place value labels on top of bars."""
    y_max = ax1.get_ylim()[1]
    for bar in bars:
        h = bar.get_height()
        x_pos = bar.get_x() + bar.get_width() / 2.0
        # Place inside bar if tall enough, else above
        if h > y_max * 0.08:
            ax1.text(x_pos, h * 0.5, fmt.format(h),
                     ha="center", va="center", fontsize=8.5,
                     color=color, fontweight="bold", rotation=90, zorder=6)
        else:
            ax1.text(x_pos, h + y_max * offset_frac, fmt.format(h),
                     ha="center", va="bottom", fontsize=8,
                     color=TEXT_COLOR, fontweight="bold", zorder=6)

# Draw without rotation for small values, with rotation for large
for bar in bars_min:
    h = bar.get_height()
    x_pos = bar.get_x() + bar.get_width() / 2.0
    ax1.text(x_pos, h + 0.4, f"{h:.2f}", ha="center", va="bottom",
             fontsize=8, color=TEXT_COLOR, fontweight="bold", zorder=6)

for bar in bars_avg:
    h = bar.get_height()
    x_pos = bar.get_x() + bar.get_width() / 2.0
    ax1.text(x_pos, h + 0.4, f"{h:.2f}", ha="center", va="bottom",
             fontsize=8, color=TEXT_COLOR, fontweight="bold", zorder=6)

for bar in bars_p95:
    h = bar.get_height()
    x_pos = bar.get_x() + bar.get_width() / 2.0
    ax1.text(x_pos, h + 0.4, f"{h:.2f}", ha="center", va="bottom",
             fontsize=8, color=TEXT_COLOR, fontweight="bold", zorder=6)

# ── PPS value labels ──────────────────────────────────────────────
for xi, yi in zip(x, pps):
    ax2.text(xi, yi + 28, f"{yi} PPS", ha="center", va="bottom",
             fontsize=9, color=IITM_NAVY, fontweight="bold", zorder=6)

# ── Axes Formatting ───────────────────────────────────────────────
ax1.set_ylabel("Inference Latency (ms)", color=TEXT_COLOR,
               fontsize=12, fontweight="bold", labelpad=10)
ax2.set_ylabel("Throughput (Packets Per Second)", color=IITM_NAVY,
               fontsize=12, fontweight="bold", labelpad=10)

ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=10.5, fontweight="bold")
ax1.set_ylim(0, 48)
ax2.set_ylim(0, 1250)

ax1.yaxis.label.set_color(TEXT_COLOR)
ax1.tick_params(axis="y", colors=TEXT_COLOR)
ax2.tick_params(axis="y", colors=IITM_NAVY)

ax1.grid(axis="y", color=GRID_COLOR, linewidth=0.8, zorder=0)
ax1.set_axisbelow(True)

# Highlight Mamba vs best performer annotation
ax1.axvline(x=0.5, color="#AAAAAA", linewidth=1.0,
            linestyle=":", alpha=0.7, zorder=1)
ax1.text(0.51, 44, "← Baseline  |  Our System →",
         fontsize=8.5, color="#888888", style="italic")

# ── Speed-up Annotation (Tier-2 vs Mamba) ────────────────────────
ax1.annotate(
    "11.97× faster avg.\n(1.05 ms vs 12.57 ms)",
    xy=(2, lat_avg[2]), xytext=(2.38, 18),
    arrowprops=dict(arrowstyle="->", color=IITM_NAVY, lw=1.5),
    fontsize=8.5, color=IITM_NAVY, fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="#EAF3FB",
              edgecolor=IITM_NAVY, linewidth=0.8),
    zorder=7
)

ax1.annotate(
    "11.86× higher PPS\n(949 vs 80)",
    xy=(2, 0), xytext=(2.45, 8),
    arrowprops=dict(arrowstyle="->", color=IITM_BLUE, lw=1.5),
    fontsize=8.5, color=IITM_BLUE, fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="#EAF3FB",
              edgecolor=IITM_BLUE, linewidth=0.8),
    zorder=7
)

# ── Title ─────────────────────────────────────────────────────────
ax1.set_title(
    "Hybrid-Sentinel AI Firewall — Latency \u0026 Throughput Benchmark\n"
    "vs. Mamba SSM Baseline  (N\u202f=\u202f2,000 packets, run: 2026-03-27)",
    fontsize=12.5, fontweight="bold", color=TEXT_COLOR, pad=14,
    loc="left"
)

# ── Combined Legend ───────────────────────────────────────────────
lat_min_patch = mpatches.Patch(color=IITM_MAROON, label="Min Latency (ms)")
lat_avg_patch = mpatches.Patch(color=IITM_RED,    label="Avg Latency (ms)")
lat_p95_patch = mpatches.Patch(color=IITM_ORANGE, label="p95 Latency (ms)")
pps_line = plt.Line2D([0], [0], color=IITM_NAVY, linewidth=2.2,
                       linestyle="--", marker="D",
                       markerfacecolor=IITM_BLUE, markersize=7,
                       label="Throughput (PPS)")

ax1.legend(
    handles=[lat_min_patch, lat_avg_patch, lat_p95_patch, pps_line],
    loc="upper right", fontsize=9.5, framealpha=0.92,
    edgecolor="#CCCCCC", fancybox=False,
    ncol=2
)

# ── Footer note ───────────────────────────────────────────────────
fig.text(
    0.01, 0.01,
    "Source: results/latency_benchmark.json  |  "
    "Hybrid-Sentinel NDN AI Firewall Project  |  IITM M.Tech Thesis",
    fontsize=7.5, color="#999999", style="italic"
)

# ── Save ──────────────────────────────────────────────────────────
plt.tight_layout(rect=[0, 0.03, 1, 1])

pdf_path = os.path.join(OUT_DIR, "fig_latency_benchmark.pdf")
png_path = os.path.join(OUT_DIR, "fig_latency_benchmark.png")

plt.savefig(pdf_path, dpi=300, bbox_inches="tight",
            facecolor=BG_COLOR, format="pdf")
plt.savefig(png_path, dpi=300, bbox_inches="tight",
            facecolor=BG_COLOR, format="png")

print(f"✅  Saved PDF → {pdf_path}")
print(f"✅  Saved PNG → {png_path}")
plt.show()
