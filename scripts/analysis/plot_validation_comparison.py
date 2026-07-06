"""
plot_validation_comparison.py
─────────────────────────────────────────────────────────────────────
Hybrid-Sentinel AI Firewall — Chapter 5 Figure
Validation Comparison Chart (Leakage Mitigation)
Data source: proof_of_work_log.json (R14 vs R16)

Output: results/proof_of_work_visuals/fig_validation_comparison.pdf
        results/proof_of_work_visuals/fig_validation_comparison.png
─────────────────────────────────────────────────────────────────────
"""

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# ── Output directory ──────────────────────────────────────────────
OUT_DIR = os.path.join("results", "proof_of_work_visuals")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Data ──────────────────────────────────────────────────────────
labels = ["Round 14\n(Sequence-Level Split)", "Round 16\n(Group-Disjoint Split)"]
f1_scores = [1.0000, 0.9943]

# ── Figure Setup ──────────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family"     : "serif", # Use serif for academic look
    "font.size"       : 12,
    "axes.titlesize"  : 14,
    "axes.labelsize"  : 12,
    "xtick.labelsize" : 11,
    "ytick.labelsize" : 11,
    "axes.spines.top" : False,
    "axes.spines.right": False,
    "figure.dpi"      : 150,
})

fig, ax = plt.subplots(figsize=(8, 6), facecolor='white')

# ── Bar plotting with BW/Grayscale suitable styles ─────────────────
x = np.arange(len(labels))
bar_width = 0.5

# Round 14 Bar (Artefactual) - Light grey with diagonal hatching
bar1 = ax.bar(x[0], f1_scores[0], bar_width, 
              color="#E0E0E0", edgecolor="black", hatch="///", linewidth=1.5,
              label="Leaked Evaluation")

# Round 16 Bar (Generalizable) - Darker shade with different hatch, 
# although user asked for 'green bar' but also 'black-and-white print thesis'. 
# We'll use a muted green that prints well in grayscale, plus a distinct hatch.
bar2 = ax.bar(x[1], f1_scores[1], bar_width, 
              color="#6b8e23", edgecolor="black", hatch="\\\\\\", linewidth=1.5,
              label="Generalizable Performance")

# ── Horizontal Line for Artefactual Perfection ────────────────────
ax.axhline(y=1.000, color="#C0392B", linestyle="--", linewidth=2.0, zorder=0)
ax.text(x[0]-0.25, 1.000 + 0.001, "Artefactual Perfection (F1 = 1.000)", 
        color="#C0392B", fontweight="bold", fontsize=11, ha="left", va="bottom")

# ── Value Annotations ─────────────────────────────────────────────
ax.text(x[0], f1_scores[0] - 0.005, f"{f1_scores[0]:.4f}", 
        ha='center', va='top', fontsize=12, fontweight='bold', 
        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2'))

ax.text(x[1], f1_scores[1] - 0.005, f"{f1_scores[1]:.4f}", 
        ha='center', va='top', fontsize=12, fontweight='bold', color='white',
        bbox=dict(facecolor='#6b8e23', edgecolor='black', boxstyle='round,pad=0.2'))

# ── Axes Formatting ───────────────────────────────────────────────
ax.set_ylim(0.90, 1.01) # Zoomed in Y-axis
ax.set_ylabel("Test Macro F1 Score", fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontweight="bold")

# Y-axis ticks formatting
yticks = np.arange(0.90, 1.01, 0.02)
ax.set_yticks(yticks)
ax.set_yticklabels([f"{tick:.2f}" for tick in yticks])

# Grid lines
ax.grid(axis='y', linestyle=':', color='#999999', alpha=0.7, zorder=0)
ax.set_axisbelow(True)

# ── Title ─────────────────────────────────────────────────────────
ax.set_title("Mitigation of Evaluation Bias: Sequence vs. Group Split", 
             fontweight="bold", pad=20)

# ── Legend ────────────────────────────────────────────────────────
ax.legend(loc="lower left", fontsize=11, framealpha=1.0, edgecolor='black')

# ── Save ──────────────────────────────────────────────────────────
plt.tight_layout()

pdf_path = os.path.join(OUT_DIR, "fig_validation_comparison.pdf")
png_path = os.path.join(OUT_DIR, "fig_validation_comparison.png")

plt.savefig(pdf_path, dpi=300, bbox_inches="tight", format="pdf")
plt.savefig(png_path, dpi=300, bbox_inches="tight", format="png")

print(f"✅  Saved PDF → {pdf_path}")
print(f"✅  Saved PNG → {png_path}")
plt.show()
