"""
plot_ce_vs_focal_confusion.py
─────────────────────────────────────────────────────────────────────
Figure 6.3: Side-by-Side Confusion Matrix
Left:  Round 13 — Cross-Entropy (Baseline, Invalid)
Right: Round 16 — Focal Loss (First Valid Result)

Data source: results/proof_of_work_log.json
─────────────────────────────────────────────────────────────────────
"""
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

OUT_DIR = os.path.join("results", "proof_of_work_visuals")
os.makedirs(OUT_DIR, exist_ok=True)

matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "figure.dpi": 150,
})

# ── Data from proof_of_work_log.json ─────────────────────────────
# Round 13 — Cross-Entropy, support: BF=3124, DDOS=6921, SLOW=14
cm_r13 = np.array([
    [2139, 985,  0 ],
    [2115, 4800, 6 ],
    [0,    2,    12],
])

# Round 16 — Focal Loss, support: BF=58, DDOS=116, SLOW=58
cm_r16 = np.array([
    [57,  0,   1 ],
    [0,   116, 0 ],
    [0,   0,   58],
])

labels = ["BRUTE\nFORCE", "DDOS\nHTTP", "SLOW\nHTTP"]

metrics = {
    "R13": {
        "macro_f1": 0.6951,
        "loss": "Cross-Entropy\n(No class weighting)",
        "valid": False,
        "per_class": {
            "BF":   (0.503, 0.685, 0.580),
            "DDOS": (0.829, 0.694, 0.755),
            "SLOW": (0.667, 0.857, 0.750),
        }
    },
    "R16": {
        "macro_f1": 0.9943,
        "loss": "Focal Loss\n(γ=2, α_SLOW=1.5)",
        "valid": True,
        "per_class": {
            "BF":   (1.000, 0.983, 0.991),
            "DDOS": (1.000, 1.000, 1.000),
            "SLOW": (0.983, 1.000, 0.992),
        }
    },
}

# ── Custom colormaps ──────────────────────────────────────────────
# R13 (invalid/grey-red): muted
cmap_r13 = LinearSegmentedColormap.from_list(
    "grey_red", ["#FFFFFF", "#C0392B"], N=256
)
# R16 (valid/blue): clean IITM blue
cmap_r16 = LinearSegmentedColormap.from_list(
    "white_blue", ["#FFFFFF", "#1A2F5A"], N=256
)

# ── Figure ────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 6.5), facecolor="white")
gs = gridspec.GridSpec(1, 2, wspace=0.35)

def draw_cm(ax, cm, title, subtitle, cmap, macro_f1, valid, per_class):
    # Normalize per row for color
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    im = ax.imshow(cm_norm, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    n = len(labels)
    for i in range(n):
        for j in range(n):
            val = cm[i, j]
            pct = cm_norm[i, j]
            color = "white" if pct > 0.55 else "#1C1C1C"
            ax.text(j, i, f"{val:,}\n({pct:.0%})",
                    ha="center", va="center",
                    fontsize=10, fontweight="bold", color=color)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, fontsize=10, fontweight="bold")
    ax.set_yticklabels(labels, fontsize=10, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontweight="bold", labelpad=8)
    ax.set_ylabel("Actual Label", fontweight="bold", labelpad=8)

    # Validity badge
    badge_color = "#E74C3C" if not valid else "#27AE60"
    badge_text  = "⚠ LEAKED / INVALID" if not valid else "✅ VALID"
    ax.set_title(f"{title}\n{subtitle}",
                 fontsize=12, fontweight="bold", pad=14)
    ax.text(0.5, 1.01, badge_text,
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=9, color=badge_color, fontweight="bold")

    # Per-class metrics table below
    col_labels = ["Prec.", "Recall", "F1"]
    row_labels  = ["BF", "DDOS", "SLOW"]
    table_data  = [[f"{v:.3f}" for v in per_class[k]]
                   for k in row_labels]
    tbl = ax.table(
        cellText=table_data,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellLoc="center",
        loc="bottom",
        bbox=[0.0, -0.48, 1.0, 0.38],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#CCCCCC")
        if r == 0:
            cell.set_facecolor("#F0F0F0")
            cell.set_text_props(fontweight="bold")
        elif c == -1:
            cell.set_facecolor("#F5F5F5")
            cell.set_text_props(fontweight="bold")

    # Macro F1 annotation
    f1_color = "#C0392B" if not valid else "#1A2F5A"
    ax.text(0.5, -0.56, f"Test Macro F1: {macro_f1:.4f}",
            transform=ax.transAxes, ha="center",
            fontsize=11, fontweight="bold", color=f1_color)

# Draw both matrices
ax1 = fig.add_subplot(gs[0])
draw_cm(ax1,
        cm_r13,
        "Round 13 — Cross-Entropy",
        "Sequence-Level Split  |  33 features",
        cmap_r13,
        metrics["R13"]["macro_f1"],
        metrics["R13"]["valid"],
        metrics["R13"]["per_class"])

ax2 = fig.add_subplot(gs[1])
draw_cm(ax2,
        cm_r16,
        "Round 16 — Focal Loss",
        "GroupShuffleSplit  |  17 leak-free features",
        cmap_r16,
        metrics["R16"]["macro_f1"],
        metrics["R16"]["valid"],
        metrics["R16"]["per_class"])

# ── Overall title ─────────────────────────────────────────────────
fig.suptitle(
    "Figure 6.3 — Impact of Focal Loss on Attack Detection Performance\n"
    "Cross-Entropy Baseline (R13) vs. Focal Loss with GroupShuffleSplit (R16)",
    fontsize=13, fontweight="bold", y=1.01
)

# ── Improvement arrow between plots ──────────────────────────────
fig.text(0.495, 0.54, "→", fontsize=28, ha="center",
         va="center", color="#2ECC71", fontweight="bold")
fig.text(0.495, 0.47, "+32.2 pp\nSLOW F1", fontsize=8.5,
         ha="center", color="#27AE60", fontweight="bold")

# ── Footer ────────────────────────────────────────────────────────
fig.text(0.01, -0.01,
         "Source: results/proof_of_work_log.json  |  "
         "Hybrid-Sentinel NDN AI Firewall  |  IITM M.Tech Thesis",
         fontsize=7.5, color="#999999", style="italic")

plt.tight_layout(rect=[0, 0.0, 1, 0.97])

pdf_path = os.path.join(OUT_DIR, "fig_ce_vs_focal_confusion.pdf")
png_path = os.path.join(OUT_DIR, "fig_ce_vs_focal_confusion.png")
plt.savefig(pdf_path, dpi=300, bbox_inches="tight", format="pdf")
plt.savefig(png_path, dpi=300, bbox_inches="tight", format="png")
print(f"✅  Saved PDF → {pdf_path}")
print(f"✅  Saved PNG → {png_path}")
plt.show()
