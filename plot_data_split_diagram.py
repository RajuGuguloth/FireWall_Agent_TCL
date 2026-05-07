"""
plot_data_split_diagram.py
─────────────────────────────────────────────────────────────────────
Figure 5.3: Data Splitting Architecture Diagram
Shows correct (GroupShuffleSplit) vs invalid (random) splitting.

─────────────────────────────────────────────────────────────────────
"""
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT_DIR = os.path.join("results", "proof_of_work_visuals")
os.makedirs(OUT_DIR, exist_ok=True)

matplotlib.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})

NAVY    = "#1A2F5A"
MAROON  = "#8B0000"
GREEN   = "#27AE60"
RED     = "#C0392B"
GREY    = "#BDC3C7"
LGREY   = "#ECF0F1"
AMBER   = "#E67E22"
WHITE   = "#FFFFFF"

fig, (ax_good, ax_bad) = plt.subplots(
    1, 2, figsize=(15, 8.5), facecolor="white"
)
fig.patch.set_facecolor("white")

# ─────────────────────────────────────────────────────────────────
def box(ax, x, y, w, h, color, text, fontsize=9,
        text_color=WHITE, style="round,pad=0.1", alpha=1.0, hatch=None):
    b = FancyBboxPatch((x - w/2, y - h/2), w, h,
                        boxstyle=style, linewidth=1.2,
                        edgecolor=color, facecolor=color,
                        alpha=alpha, hatch=hatch)
    ax.add_patch(b)
    ax.text(x, y, text, ha="center", va="center",
            fontsize=fontsize, color=text_color, fontweight="bold",
            wrap=True, multialignment="center")

def arrow(ax, x1, y1, x2, y2, color=NAVY, lw=1.8, style="->"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color,
                                lw=lw, connectionstyle="arc3,rad=0.0"))

def label(ax, x, y, text, color=NAVY, fs=9, style="normal"):
    ax.text(x, y, text, ha="center", va="center",
            fontsize=fs, color=color, fontstyle=style)

# ─────────────────────────────────────────────────────────────────
# LEFT PANEL: CORRECT — GroupShuffleSplit
# ─────────────────────────────────────────────────────────────────
ax = ax_good
ax.set_xlim(0, 10)
ax.set_ylim(0, 12)
ax.axis("off")
ax.set_facecolor("white")
ax.set_title("✅  Correct Method: GroupShuffleSplit\n(Round 16 — Group-Disjoint, No Leakage)",
             fontsize=11, fontweight="bold", color=GREEN, pad=10)

# Raw stream
box(ax, 5, 11.2, 7, 0.8, NAVY, "Raw Packet Stream  (dst_port, ip_proto, attack_type)", fontsize=9)

# Groups
for i, (gx, label_txt, col) in enumerate([
    (2.0, "Group A\nPort 80/TCP\n(DDoS)", "#2980B9"),
    (5.0, "Group B\nPort 22/TCP\n(BF)", "#8E44AD"),
    (8.0, "Group C\nPort 53/UDP\n(DNS)", "#16A085"),
]):
    arrow(ax, 5, 10.8, gx, 10.1, color=col)
    box(ax, gx, 9.7, 2.2, 0.7, col, label_txt, fontsize=8)

label(ax, 5, 9.2, "↓  Sliding Window Sequences  (S=20, stride=10)", NAVY, fs=8.5, style="italic")

# Sequences inside groups
for gx, col, seqs in [
    (2.0, "#2980B9", ["Seq A₁", "Seq A₂", "Seq A₃"]),
    (5.0, "#8E44AD", ["Seq B₁", "Seq B₂", "Seq B₃"]),
    (8.0, "#16A085", ["Seq C₁", "Seq C₂", "Seq C₃"]),
]:
    for k, (sy, s) in enumerate(zip([8.5, 7.9, 7.3], seqs)):
        box(ax, gx, sy, 1.8, 0.45, col, s, fontsize=7.5, alpha=0.85)

label(ax, 5, 6.8,
      "GroupShuffleSplit operates on GROUPS, not sequences",
      NAVY, fs=9, style="italic")

# GroupShuffleSplit decision
box(ax, 5, 6.3, 7, 0.6, GREEN,
    "GroupShuffleSplit  →  Entire groups dispatched atomically", fontsize=9)

# Train/Test bins
box(ax, 2.5, 5.0, 3.5, 1.6, "#2980B9",
    "TRAIN SET\nGroup A (Port 80) — all sequences\nGroup B (Port 22) — all sequences",
    fontsize=8, alpha=0.9)
box(ax, 7.5, 5.0, 3.5, 1.6, "#16A085",
    "TEST SET\nGroup C (Port 53) — all sequences\n(never seen in training)",
    fontsize=8, alpha=0.9)

arrow(ax, 2.0, 6.0, 2.5, 5.82, GREEN)
arrow(ax, 5.0, 6.0, 2.5, 5.82, GREEN)
arrow(ax, 8.0, 6.0, 7.5, 5.82, GREEN)

# Guarantee box
box(ax, 5, 3.6, 8, 0.75, GREEN,
    "GUARANTEE:  Train Groups ∩ Test Groups = ∅\n(assert verified in refine_dataset.py)",
    fontsize=8.5, alpha=0.95)

label(ax, 5, 2.8, "Result: Macro F1 = 0.9943  ✅  (Generalizable)", GREEN, fs=10)

# ─────────────────────────────────────────────────────────────────
# RIGHT PANEL: INVALID — Random train_test_split
# ─────────────────────────────────────────────────────────────────
ax = ax_bad
ax.set_xlim(0, 10)
ax.set_ylim(0, 12)
ax.axis("off")
ax.set_facecolor("white")
ax.set_title("❌  Invalid Method: Sequence-Level Random Split\n(Round 14 — Same-Group Leakage)",
             fontsize=11, fontweight="bold", color=RED, pad=10)

# Raw stream
box(ax, 5, 11.2, 7, 0.8, MAROON, "Raw Packet Stream  (dst_port, ip_proto, attack_type)", fontsize=9)

# Groups (same)
for i, (gx, label_txt, col) in enumerate([
    (2.0, "Group A\nPort 80/TCP\n(DDoS)", "#2980B9"),
    (5.0, "Group B\nPort 22/TCP\n(BF)", "#8E44AD"),
    (8.0, "Group C\nPort 53/UDP\n(DNS)", "#16A085"),
]):
    arrow(ax, 5, 10.8, gx, 10.1, color=col)
    box(ax, gx, 9.7, 2.2, 0.7, col, label_txt, fontsize=8)

label(ax, 5, 9.2, "↓  Sliding Window Sequences  (S=20, stride=10)", MAROON, fs=8.5, style="italic")

# Sequences — all poured into one pool
for gx, col, seqs in [
    (2.0, "#2980B9", ["Seq A₁", "Seq A₂", "Seq A₃"]),
    (5.0, "#8E44AD", ["Seq B₁", "Seq B₂", "Seq B₃"]),
    (8.0, "#16A085", ["Seq C₁", "Seq C₂", "Seq C₃"]),
]:
    for k, (sy, s) in enumerate(zip([8.5, 7.9, 7.3], seqs)):
        box(ax, gx, sy, 1.8, 0.45, col, s, fontsize=7.5, alpha=0.85)

# Pool all sequences
arrow(ax, 2.0, 7.05, 5.0, 6.6, MAROON)
arrow(ax, 5.0, 7.05, 5.0, 6.6, MAROON)
arrow(ax, 8.0, 7.05, 5.0, 6.6, MAROON)

box(ax, 5, 6.3, 7, 0.6, GREY,
    "All sequences pooled — groups ignored", fontsize=9, text_color="#333333")

label(ax, 5, 5.85, "train_test_split(stratify=y)  ← operates on sequences",
      RED, fs=8.5, style="italic")

# Random lottery boxes — mixed colors showing leakage
box(ax, 2.5, 5.0, 3.5, 1.6, MAROON,
    "TRAIN SET\nSeq A₁ (Port 80)\nSeq B₂ (Port 22)\nSeq C₁ (Port 53)",
    fontsize=8, alpha=0.8)
box(ax, 7.5, 5.0, 3.5, 1.6, MAROON,
    "TEST SET\nSeq A₂ (Port 80)  ← LEAK\nSeq B₁ (Port 22)  ← LEAK\nSeq C₃ (Port 53)  ← LEAK",
    fontsize=8, alpha=0.8)

# Leakage arrow
ax.annotate("",
    xy=(5.75, 5.0), xytext=(4.25, 5.0),
    arrowprops=dict(arrowstyle="<->", color=RED, lw=2.5))
ax.text(5.0, 5.22, "LEAKAGE\nPATHWAY", ha="center", fontsize=8,
        color=RED, fontweight="bold",
        bbox=dict(facecolor="#FDEDEC", edgecolor=RED, boxstyle="round,pad=0.2"))

# Consequence
box(ax, 5, 3.6, 8, 0.75, RED,
    "Model memorises Port 80 → DDoS, Port 22 → BF\nNo behavioural learning occurs",
    fontsize=8.5, alpha=0.9)

label(ax, 5, 2.8, "Result: Macro F1 = 1.0000  ❌  (Artefactual)", RED, fs=10)

# ─────────────────────────────────────────────────────────────────
fig.suptitle(
    "Figure 5.3 — Data Splitting Architecture: Correct vs. Invalid Methodology",
    fontsize=13, fontweight="bold", y=1.0
)
fig.text(0.01, -0.01,
         "Source: results/proof_of_work_log.json  |  "
         "Hybrid-Sentinel NDN AI Firewall  |  IITM M.Tech Thesis",
         fontsize=7.5, color="#999999", style="italic")

plt.tight_layout()
pdf_path = os.path.join(OUT_DIR, "fig_data_split_diagram.pdf")
png_path = os.path.join(OUT_DIR, "fig_data_split_diagram.png")
plt.savefig(pdf_path, dpi=300, bbox_inches="tight", format="pdf")
plt.savefig(png_path, dpi=300, bbox_inches="tight", format="png")
print(f"✅  Saved PDF → {pdf_path}")
print(f"✅  Saved PNG → {png_path}")
plt.show()
