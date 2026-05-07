"""
plot_architecture_diagram.py — Hybrid-Sentinel 3-Tier Architecture Flowchart
"""
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

OUT_DIR = os.path.join("results", "proof_of_work_visuals")
os.makedirs(OUT_DIR, exist_ok=True)
matplotlib.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "figure.dpi": 150})

C_NAVY="#1A2F5A"; C_BLUE="#2980B9"; C_GREEN="#27AE60"
C_RED="#8B0000";  C_ORANGE="#E67E22"; C_PURPLE="#8E44AD"
C_TEAL="#16A085"; C_GREY="#95A5A6";  C_LGREY="#F4F6F7"; C_WHITE="#FFFFFF"

fig, ax = plt.subplots(figsize=(13, 19), facecolor="white")
ax.set_xlim(0,13); ax.set_ylim(0,19); ax.axis("off")

def rbox(ax, cx, cy, w, h, bg, lines, title_idx=0, tfs=10, bfs=8.5, border=None):
    border = border or bg
    ax.add_patch(FancyBboxPatch((cx-w/2, cy-h/2), w, h,
        boxstyle="round,pad=0.08", linewidth=1.6,
        edgecolor=border, facecolor=bg, zorder=2))
    n = len(lines); step = h/(n+1)
    for i,(txt,bold,col) in enumerate(lines):
        ypos = cy + h/2 - (i+1)*step
        ax.text(cx, ypos, txt, ha="center", va="center",
                fontsize=tfs if i==title_idx else bfs,
                fontweight="bold" if bold else "normal",
                color=col or C_WHITE, zorder=4)

def arr(ax, x1,y1,x2,y2, col=C_NAVY, lw=2.0, label="", loffx=0.15, loffy=0):
    ax.annotate("", xy=(x2,y2), xytext=(x1,y1),
        arrowprops=dict(arrowstyle="-|>", color=col, lw=lw), zorder=3)
    if label:
        mx=(x1+x2)/2; my=(y1+y2)/2
        ax.text(mx+loffx, my+loffy, label, ha="left", va="center",
                fontsize=8, color=col, style="italic", zorder=5)

def diamond(ax, cx, cy, text, col=C_NAVY, fs=9):
    dx,dy = 1.4, 0.4
    xs = [cx, cx+dx, cx, cx-dx, cx]
    ys = [cy+dy, cy, cy-dy, cy, cy+dy]
    ax.fill(xs, ys, color="#EAF4FC", zorder=2)
    ax.plot(xs, ys, color=col, linewidth=1.5, zorder=3)
    ax.text(cx, cy, text, ha="center", va="center",
            fontsize=fs, fontweight="bold", color=col, zorder=4)

# ── Title ────────────────────────────────────────────────────────────
ax.text(6.5, 18.6, "Hybrid-Sentinel: 3-Tier AI Firewall Architecture",
        ha="center", va="center", fontsize=14, fontweight="bold", color=C_NAVY)
ax.text(6.5, 18.18, "Figure 4.1  |  Packet flow from ingress to final routing decision",
        ha="center", va="center", fontsize=9, color=C_GREY)

# ── 1. Packet Ingress ────────────────────────────────────────────────
rbox(ax, 6.5, 17.35, 9.5, 0.82, C_NAVY, [
    ("Incoming Network Traffic  (NDN / IP-Overlay Packet Stream)", True, C_WHITE),
], tfs=11)
arr(ax, 6.5,16.94, 6.5,16.48, C_NAVY)

# ── 2. Feature Extraction ────────────────────────────────────────────
rbox(ax, 6.5, 16.1, 9.5, 0.65, C_TEAL, [
    ("Feature Extraction   extract_v5_features.py   =>   x in R^17", True, C_WHITE),
], tfs=9.5)
arr(ax, 6.5,15.77, 6.5,15.25, C_TEAL)

# ── 3. Tier-1 RF ─────────────────────────────────────────────────────
rbox(ax, 6.5, 14.55, 9.5, 1.30, C_BLUE, [
    ("TIER-1 — Random Forest Classifier  (tier1_rf_v4.pkl)", True,  C_WHITE),
    ("200 trees  |  Gini impurity  |  class_weight='balanced'", False, "#D6EAF8"),
    ("Per-packet inference   x in R^17   (stateless, O(1))", False, "#D6EAF8"),
    ("Avg latency: 4.48 ms   |   Throughput: 223 PPS", False, "#D6EAF8"),
], tfs=10, bfs=8.5)
arr(ax, 6.5,13.90, 6.5,13.42, C_BLUE)

# ── Decision diamond 1 ────────────────────────────────────────────────
diamond(ax, 6.5, 13.0, "p1* >= t1=0.85 ?", col=C_BLUE, fs=9)

# YES => ALLOW right
ax.annotate("", xy=(11.2,13.0), xytext=(7.9,13.0),
    arrowprops=dict(arrowstyle="-|>", color=C_GREEN, lw=2.0), zorder=3)
ax.text(9.55,13.16,"YES (~95% traffic)", ha="center",
        va="bottom", fontsize=8.5, color=C_GREEN, fontweight="bold")
rbox(ax, 12.1, 13.0, 2.4, 0.72, C_GREEN, [
    ("ALLOW", True, C_WHITE), ("Fast Path Exit", False, "#D5F5E3"),
], tfs=9.5, bfs=8.5)

# NO => Tier-2
arr(ax, 6.5,12.60, 6.5,12.05, C_ORANGE, lw=2.0,
    label="NO — escalate (~5%)", loffx=0.18, loffy=0)

# ── 4. Sequence Window ────────────────────────────────────────────────
rbox(ax, 6.5, 11.62, 9.5, 0.72, C_ORANGE, [
    ("Build Sliding Window Sequence   X in R^{20x17}   (stride=10)", True, C_WHITE),
    ("Sequence groups by (dst_port, ip_proto)  |  GroupShuffleSplit applied at training", False, C_WHITE),
], tfs=9.5, bfs=8)
arr(ax, 6.5,11.26, 6.5,10.72, C_RED)

# ── 5. Tier-2 CNN-GRU ─────────────────────────────────────────────────
rbox(ax, 6.5, 10.0, 9.5, 1.45, C_RED, [
    ("TIER-2 — CNN-GRU Deep Classifier  (tier2_cnn_gru_v1_r17.pth)", True,  C_WHITE),
    ("Conv1D (k=3, C=64) -> BatchNorm -> ReLU -> MaxPool", False, "#FADBD8"),
    ("GRU (hidden=128, layers=2)  ->  temporal encoding  h_S in R^128", False, "#FADBD8"),
    ("Focal Loss (gamma=2, alpha_SLOW=1.5) | Temp scaling T*=0.6638", False, "#FADBD8"),
    ("Avg latency: 1.05 ms  |  Throughput: 949 PPS  |  11.97x vs Mamba", False, "#FADBD8"),
], tfs=10, bfs=8.5)
arr(ax, 6.5,9.27, 6.5,8.78, C_RED)

# ── Decision diamond 2 ────────────────────────────────────────────────
diamond(ax, 6.5, 8.36, "p2* >= t2=0.70 ?", col=C_RED, fs=9)

# YES => BLOCK/ALLOW right
ax.annotate("", xy=(11.2,8.36), xytext=(7.9,8.36),
    arrowprops=dict(arrowstyle="-|>", color=C_GREEN, lw=2.0), zorder=3)
ax.text(9.55,8.52,"YES", ha="center",
        va="bottom", fontsize=8.5, color=C_GREEN, fontweight="bold")
rbox(ax, 12.1, 8.36, 2.4, 0.72, C_GREEN, [
    ("BLOCK/ALLOW", True, C_WHITE), ("Tier-2 Decision", False, "#D5F5E3"),
], tfs=9.5, bfs=8.5)

# NO => Tier-3
arr(ax, 6.5,7.96, 6.5,7.40, C_PURPLE, lw=2.0,
    label="NO — uncertain", loffx=0.18, loffy=0)

# ── 6. Tier-3 GNN ─────────────────────────────────────────────────────
rbox(ax, 6.5, 6.65, 9.5, 1.42, C_PURPLE, [
    ("TIER-3 — Graph Neural Network  (gnn_model_v1.pt)", True,  C_WHITE),
    ("Build traffic graph  G=(V,E)  from flow embeddings h_S", False, "#E8DAEF"),
    ("GAT (<=500 nodes)  or  GraphSAGE (>500 nodes)  — auto-selected", False, "#E8DAEF"),
    ("Detects collusive multi-node coordination (invisible to Tier-1/2)", False, "#E8DAEF"),
    ("Avg latency: 1.05 ms  |  Throughput: 952 PPS", False, "#E8DAEF"),
], tfs=10, bfs=8.5)
arr(ax, 6.5,5.94, 6.5,5.42, C_NAVY)

# ── 7. Final Router ───────────────────────────────────────────────────
rbox(ax, 6.5, 5.0, 9.5, 0.78, C_NAVY, [
    ("Final Decision Router  (decision/router.py)", True, C_WHITE),
    ("Aggregates Tier-1/2/3 confidences  |  Issues FQ-CoDel drop-hints", False, "#AED6F1"),
], tfs=10, bfs=8.5)

# ALLOW / BLOCK outputs
arr(ax, 4.2,4.61, 2.8,3.96, C_GREEN, lw=2.0, label="Benign", loffx=-1.5, loffy=0.1)
arr(ax, 8.8,4.61,10.2,3.96, C_RED,   lw=2.0, label="Attack", loffx=0.12, loffy=0.1)

rbox(ax, 2.1, 3.58, 3.6, 0.70, C_GREEN, [
    ("ALLOW  —  Packet forwarded", True, C_WHITE),
    ("to NDN content router", False, "#D5F5E3"),
], tfs=9.5, bfs=8.5)
rbox(ax,10.9, 3.58, 3.6, 0.70, C_RED, [
    ("BLOCK  —  Packet dropped", True, C_WHITE),
    ("+ alert logged to SIEM", False, "#FADBD8"),
], tfs=9.5, bfs=8.5)

# ── 8. Performance bar ────────────────────────────────────────────────
ax.add_patch(FancyBboxPatch((1.0,2.42),11.0,0.94,
    boxstyle="round,pad=0.08", linewidth=1.3,
    edgecolor=C_NAVY, facecolor=C_LGREY, zorder=2))
ax.text(6.5,3.12,"Production Performance  (N=2,000 packets, 2026-03-27)",
        ha="center",va="center",fontsize=9.5,fontweight="bold",color=C_NAVY,zorder=4)
ax.text(6.5,2.72,
    "Macro F1: 0.984  |  Pipeline Avg: 4.57 ms  |  218 PPS  |  "
    "Mamba baseline: 12.57 ms  (11.97x slower)",
    ha="center",va="center",fontsize=9,color=C_NAVY,zorder=4)

# ── Legend ────────────────────────────────────────────────────────────
items = [
    mpatches.Patch(color=C_TEAL,   label="Feature Extraction"),
    mpatches.Patch(color=C_BLUE,   label="Tier-1: Random Forest"),
    mpatches.Patch(color=C_RED,    label="Tier-2: CNN-GRU"),
    mpatches.Patch(color=C_PURPLE, label="Tier-3: GNN"),
    mpatches.Patch(color=C_NAVY,   label="Router / Decision"),
    mpatches.Patch(color=C_GREEN,  label="Allow / Pass"),
    mpatches.Patch(color=C_ORANGE, label="Escalation path"),
]
ax.legend(handles=items, loc="lower left", bbox_to_anchor=(0.0,0.0),
          ncol=4, fontsize=8.5, framealpha=1.0,
          edgecolor=C_NAVY, fancybox=False,
          title="Component Legend", title_fontsize=9)

ax.text(6.5,0.12,
    "Source: decision/router.py | tier1/train_rf_v4.py | "
    "tier2/train_cnn_gru_v4.py | tier2/train_gnn_v1.py  |  IITM M.Tech Thesis",
    ha="center",va="center",fontsize=7.5,color=C_GREY,style="italic")

plt.tight_layout(pad=0.4)
pdf_path = os.path.join(OUT_DIR, "fig_architecture_diagram.pdf")
png_path = os.path.join(OUT_DIR, "fig_architecture_diagram.png")
plt.savefig(pdf_path, dpi=300, bbox_inches="tight", format="pdf")
plt.savefig(png_path, dpi=300, bbox_inches="tight", format="png")
print(f"✅  Saved PDF -> {pdf_path}")
print(f"✅  Saved PNG -> {png_path}")
plt.show()
