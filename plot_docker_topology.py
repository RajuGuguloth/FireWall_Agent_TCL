"""
plot_docker_topology.py
─────────────────────────────────────────────────────────────────────
Figure 5.1: Docker Network Topology for In-Silico NDN Traffic Emulation
Clean grid layout, no overlapping text.
─────────────────────────────────────────────────────────────────────
"""
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

OUT_DIR = os.path.join("results", "proof_of_work_visuals")
os.makedirs(OUT_DIR, exist_ok=True)

matplotlib.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "figure.dpi": 150})

C_NAVY  = "#1A2F5A";  C_BLUE  = "#2980B9"; C_GREEN = "#27AE60"
C_RED   = "#C0392B";  C_TEAL  = "#16A085"; C_ORANGE= "#E67E22"
C_PURPLE= "#8E44AD";  C_GREY  = "#95A5A6"; C_LGREY = "#F4F6F7"
C_WHITE = "#FFFFFF"

fig, ax = plt.subplots(figsize=(14, 10), facecolor="white")
ax.set_xlim(0, 14); ax.set_ylim(0, 10); ax.axis("off")

def box(ax, cx, cy, w, h, bg, lines, title_line=0, title_fs=9.5, body_fs=8.5, border=None):
    border = border or bg
    ax.add_patch(FancyBboxPatch((cx-w/2, cy-h/2), w, h,
        boxstyle="round,pad=0.08", linewidth=1.5,
        edgecolor=border, facecolor=bg, zorder=2))
    total = len(lines)
    step  = h / (total + 1)
    for i, (txt, bold, color) in enumerate(lines):
        y = cy + h/2 - (i+1)*step
        fw = "bold" if bold else "normal"
        fc = color or C_WHITE
        ax.text(cx, y, txt, ha="center", va="center",
                fontsize=title_fs if i==title_line else body_fs,
                fontweight=fw, color=fc, zorder=4)

def arrow(ax, x1,y1, x2,y2, col=C_NAVY, lw=1.8, label="", lx=None, ly=None):
    ax.annotate("", xy=(x2,y2), xytext=(x1,y1),
        arrowprops=dict(arrowstyle="-|>", color=col, lw=lw), zorder=3)
    if label:
        ax.text(lx or (x1+x2)/2, ly or (y1+y2)/2+0.15, label,
                ha="center", va="bottom", fontsize=8,
                color=col, style="italic", zorder=5)

def dbl_arrow(ax, x1,y1, x2,y2, col=C_NAVY, lw=1.6, label="", ly_off=0.15):
    ax.annotate("", xy=(x2,y2), xytext=(x1,y1),
        arrowprops=dict(arrowstyle="<->", color=col, lw=lw), zorder=3)
    if label:
        mx=(x1+x2)/2; my=(y1+y2)/2
        ax.text(mx, my+ly_off, label, ha="center", va="bottom",
                fontsize=8, color=col, style="italic", zorder=5)

# ── Title ─────────────────────────────────────────────────────────
ax.text(7, 9.65, "Figure 5.1 — Docker-Containerized Traffic Emulation Topology",
        ha="center", va="center", fontsize=13, fontweight="bold", color=C_NAVY)
ax.text(7, 9.30, "NDN Producers and Consumers on isolated Docker bridge network",
        ha="center", va="center", fontsize=9, color=C_GREY)

# ── Docker Bridge Network band ────────────────────────────────────
ax.add_patch(FancyBboxPatch((0.3, 1.8), 13.4, 6.8,
    boxstyle="round,pad=0.1", linewidth=2.0,
    edgecolor=C_BLUE, facecolor="#EAF4FC", zorder=0, alpha=0.45))
ax.text(7, 8.45, "Docker Bridge Network  (ndn_lab_net  —  172.20.0.0/16)",
        ha="center", va="center", fontsize=9.5,
        fontweight="bold", color=C_BLUE,
        bbox=dict(facecolor=C_LGREY, edgecolor=C_BLUE,
                  boxstyle="round,pad=0.25", linewidth=1.2), zorder=3)

# ── Containers Row 1: Producers ───────────────────────────────────
producers = [
    ("NDN Producer 1\n172.20.0.2", "ndn-producer-1", "Port 80/TCP\nHTTP Content Server", C_TEAL),
    ("NDN Producer 2\n172.20.0.3", "ndn-producer-2", "Port 22/TCP\nSSH / Auth Service",  C_TEAL),
    ("NDN Producer 3\n172.20.0.4", "ndn-producer-3", "Port 53/UDP\nDNS Content Server",  C_TEAL),
]
prod_xs = [2.3, 7.0, 11.7]
for (title, cname, detail, col), px in zip(producers, prod_xs):
    box(ax, px, 7.55, 3.4, 1.30, col, [
        (title,  True,  C_WHITE),
        (cname,  False, "#D5F5E3"),
        (detail, False, C_WHITE),
    ], title_line=0, title_fs=9.5, body_fs=8.2)

# ── tcpdump tap on each producer ──────────────────────────────────
for px in prod_xs:
    ax.text(px, 6.68, "tcpdump -i eth0", ha="center", va="center",
            fontsize=7.8, color=C_ORANGE, fontstyle="italic",
            bbox=dict(facecolor="#FEF9E7", edgecolor=C_ORANGE,
                      boxstyle="round,pad=0.2", linewidth=0.9), zorder=4)

# ── Consumers / Attacker Row ──────────────────────────────────────
attackers = [
    ("Benign Consumer\n172.20.0.10", "ndn-consumer-benign", "Normal Interest pkts",  C_GREEN),
    ("DDoS Attacker\n172.20.0.20",   "ndn-attacker-ddos",   "Flood: HTTP POST x1000",C_RED),
    ("Slow-HTTP Bot\n172.20.0.21",   "ndn-attacker-slow",   "Hold conn 10s (PIT)",   C_RED),
]
att_xs = [2.3, 7.0, 11.7]
for (title, cname, detail, col), ax_ in zip(attackers, att_xs):
    box(ax, ax_, 4.35, 3.4, 1.30, col, [
        (title,  True,  C_WHITE),
        (cname,  False, "#FDFEFE"),
        (detail, False, C_WHITE),
    ], title_line=0, title_fs=9.5, body_fs=8.2)

# ── Bidirectional traffic arrows ──────────────────────────────────
for px in prod_xs:
    dbl_arrow(ax, px, 6.55, px, 5.72, col=C_NAVY, lw=1.6,
              label="veth pair", ly_off=0.12)

# ── Capture output arrows ─────────────────────────────────────────
# All three producers feed into single PCAP store
for i, px in enumerate(prod_xs):
    arrow(ax, px, 6.55, 7.0, 3.55, col=C_ORANGE, lw=1.5,
          label=("raw .pcap" if i==1 else ""), lx=7.0, ly=4.85)

# ── PCAP Store ────────────────────────────────────────────────────
box(ax, 7.0, 3.10, 5.0, 0.88, C_ORANGE, [
    ("PCAP Storage  (data/raw/attacks/ + benign/)", True,  C_WHITE),
    (".pcap files captured per container session",  False, C_WHITE),
], title_line=0, title_fs=9.5, body_fs=8.5)

arrow(ax, 7.0, 2.66, 7.0, 2.12, col=C_TEAL, lw=2.0,
      label="extract_v5_features.py", lx=9.2, ly=2.38)

# ── Feature Extraction ────────────────────────────────────────────
box(ax, 7.0, 1.72, 5.0, 0.72, C_TEAL, [
    ("Feature Extraction  \u2192  x \u2208 R\u00b9\u2077  (17-dim per packet)", True,  C_WHITE),
    ("combined_dataset_v5_final.csv  \u2192  Model Training",               False, "#D1F2EB"),
], title_line=0, title_fs=9.5, body_fs=8.5)

# ── Host machine label ────────────────────────────────────────────
ax.add_patch(FancyBboxPatch((0.1, 0.1), 13.8, 1.50,
    boxstyle="round,pad=0.05", linewidth=1.4,
    edgecolor=C_GREY, facecolor=C_LGREY, zorder=0))
ax.text(7, 1.15, "Host Machine  (macOS  |  Python 3.x  |  scapy  |  PyTorch  |  scikit-learn)",
        ha="center", va="center", fontsize=9.5, fontweight="bold", color=C_NAVY)
ax.text(7, 0.68, "Hybrid-Sentinel Tier-1 RF  +  Tier-2 CNN-GRU  +  Tier-3 GNN  run here "
        "on extracted features",
        ha="center", va="center", fontsize=8.8, color=C_NAVY)
ax.text(7, 0.28, "Source: attack_lab/  |  capture_large.sh  |  extract_v5_features.py  |  "
        "IITM M.Tech Thesis",
        ha="center", va="center", fontsize=7.5, color=C_GREY, style="italic")

# ── Legend ────────────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(color=C_TEAL,   label="NDN Producer (content server)"),
    mpatches.Patch(color=C_GREEN,  label="Benign Consumer"),
    mpatches.Patch(color=C_RED,    label="Attacker / Malicious bot"),
    mpatches.Patch(color=C_ORANGE, label="PCAP capture / feature extract"),
    mpatches.Patch(color=C_BLUE,   label="Docker bridge network"),
]
ax.legend(handles=legend_items, loc="upper left",
          bbox_to_anchor=(0.005, 0.995), ncol=1,
          fontsize=8, framealpha=1.0, edgecolor=C_NAVY,
          fancybox=False, title="Container Legend", title_fontsize=8.5)

plt.tight_layout(pad=0.3)
pdf_path = os.path.join(OUT_DIR, "fig_docker_topology.pdf")
png_path = os.path.join(OUT_DIR, "fig_docker_topology.png")
plt.savefig(pdf_path, dpi=300, bbox_inches="tight", format="pdf")
plt.savefig(png_path, dpi=300, bbox_inches="tight", format="png")
print(f"✅  Saved PDF -> {pdf_path}")
print(f"✅  Saved PNG -> {png_path}")
plt.show()
