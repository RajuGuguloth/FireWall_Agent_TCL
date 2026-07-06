"""
Proof of Work — NDN AI Firewall (Round 17)
Generates: Confusion Matrix Heatmap, Per-Class Metrics Bar Chart,
           Precision/Recall/F1 Summary, Model Parameter Card
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# R17 Metrics
cm_data = np.array([
    [3544,  278,    0,    0,    0],
    [  35, 3680,    0,    0,    0],
    [   0,    0, 1845,    0,    0],
    [   0,    0,    0, 4023,    0],
    [   0,    0,    0,    0, 3956]
])

class_names = ["BRUTE_FORCE", "DDOS_HTTP", "DNS_TUNNEL", "PORT_SCAN", "SLOW_HTTP"]
short       = ["BRUTE\nFORCE", "DDOS\nHTTP", "DNS\nTUNNEL", "PORT\nSCAN", "SLOW\nHTTP"]

precision = [0.99, 0.93, 1.00, 1.00, 1.00]
recall    = [0.93, 0.99, 1.00, 1.00, 1.00]
f1        = [0.96, 0.96, 1.00, 1.00, 1.00]
support   = [3822, 3715, 1845, 4023, 3956]

macro_f1    = 0.9840
val_f1      = 0.9834
temperature = 0.6638
epoch_best  = 7

OUT_DIR = "results/proof_of_work_visuals/r17"
os.makedirs(OUT_DIR, exist_ok=True)

# ── LOG UPDATE ──
LOG_FILE = "results/proof_of_work_log_r17.json"
pow_entry = {
    "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
    "round": 17,
    "classes": 5,
    "script": "train_cnn_gru_v4.py",
    "epoch_best": epoch_best,
    "val_f1_macro": val_f1,
    "test_f1_macro": macro_f1,
    "temperature": temperature,
    "confusion_matrix": cm_data.tolist()
}
with open(LOG_FILE, "w") as f:
    json.dump([pow_entry], f, indent=2)

# ── THEME ──
DARK_BG   = "#0d1117"
CARD_BG   = "#161b22"
ACCENT    = "#58a6ff"
GREEN     = "#3fb950"
ORANGE    = "#d29922"
TEXT      = "#e6edf3"
SUBTEXT   = "#8b949e"
PALETTE   = [ACCENT, GREEN, ORANGE, "#bc8cff", "#ff7b72"]

plt.rcParams.update({
    "figure.facecolor":  DARK_BG,
    "axes.facecolor":    CARD_BG,
    "axes.edgecolor":    SUBTEXT,
    "axes.labelcolor":   TEXT,
    "xtick.color":       SUBTEXT,
    "ytick.color":       SUBTEXT,
    "text.color":        TEXT,
    "font.family":       "sans-serif",
    "grid.color":        "#21262d",
    "grid.linestyle":    "--",
    "grid.alpha":        0.5,
})

# 1. HEATMAP
fig, ax = plt.subplots(figsize=(8, 6.5))
fig.patch.set_facecolor(DARK_BG)
cm_norm = cm_data.astype(float) / cm_data.sum(axis=1, keepdims=True)
sns.heatmap(
    cm_norm, annot=cm_data, fmt="d",
    cmap=sns.light_palette(ACCENT, as_cmap=True),
    linewidths=1.5, linecolor=DARK_BG,
    cbar=False, ax=ax, xticklabels=short, yticklabels=short,
    annot_kws={"size": 12, "weight": "bold", "color": DARK_BG}
)
ax.set_xlabel("Predicted Label", fontsize=12, labelpad=10)
ax.set_ylabel("Actual Label",    fontsize=12, labelpad=10)
ax.set_title("Confusion Matrix — Round 17 (5-Class)", fontsize=14, fontweight="bold", color=TEXT, pad=15)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "confusion_matrix_r17.png"), dpi=150, bbox_inches="tight")
plt.close()

# 2. PER-CLASS METRICS
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor(DARK_BG)
x = np.arange(len(class_names))
w = 0.24
b1 = ax.bar(x - w, precision, w, label="Precision", color=ACCENT,   alpha=0.9, zorder=3)
b2 = ax.bar(x,     recall,    w, label="Recall",    color=GREEN,   alpha=0.9, zorder=3)
b3 = ax.bar(x + w, f1,        w, label="F1-Score",  color=ORANGE,  alpha=0.9, zorder=3)
for bars in [b1, b2, b3]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005, f"{h:.2f}",
                ha="center", va="bottom", fontsize=9, color=TEXT, fontweight="bold")
ax.set_ylim(0, 1.15)
ax.set_xticks(x)
ax.set_xticklabels(short, fontsize=11)
ax.set_title("Per-Class Metrics — Round 17 (5 Classes)", fontsize=14, fontweight="bold", color=TEXT, pad=15)
ax.legend(facecolor=CARD_BG, edgecolor=SUBTEXT, labelcolor=TEXT, fontsize=11, loc="lower right")
ax.yaxis.grid(True, zorder=0)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "per_class_metrics_r17.png"), dpi=150, bbox_inches="tight")
plt.close()

# 3. DISTRIBUTION
fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor(DARK_BG)
bars = ax.bar(short, support, color=PALETTE, alpha=0.9, width=0.5, zorder=3)
for bar, count in zip(bars, support):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
            f"{count:,}", ha="center", va="bottom", fontsize=11, color=TEXT, fontweight="bold")
ax.set_title("Test Set Class Distribution — Round 17", fontsize=14, fontweight="bold", color=TEXT, pad=15)
ax.yaxis.grid(True, zorder=0)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "class_distribution_r17.png"), dpi=150, bbox_inches="tight")
plt.close()

# 4. CARD
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor(DARK_BG)
ax.axis("off")
params = [
    ("Architecture",    "Conv1D(17→64) → GRU(64→128, 2L) → Linear(128→5)"),
    ("Classes",         "5 (BF, DDoS, SlowHTTP, PortScan, DNS_Tunnel)"),
    ("Loss Function",   "Focal Loss (γ=2, α=[0.6, 0.8, 1.5, 1.0, 1.2])"),
    ("Optimizer",       "AdamW (lr=5e-5, wd=1e-4)"),
    ("Patience/Epochs", "Patience 10, target 50 (Stopped at Epoch 8)"),
    ("Test Size",       f"{sum(support):,} (expanded dataset)"),
    ("Best Epoch",      str(epoch_best)),
    ("Temperature T",   f"{temperature:.4f}"),
    ("Macro F1",        f"{macro_f1:.4f}"),
]
ax.text(0.5, 0.95, "🧠  Hybrid-Sentinel  |  Model Parameter Card R17", transform=ax.transAxes, fontsize=16, fontweight="bold", color=ACCENT, ha="center")
y = 0.80
for key, val in params:
    ax.text(0.05, y, f"▸  {key}:", transform=ax.transAxes, fontsize=11, color=SUBTEXT)
    ax.text(0.35, y, val, transform=ax.transAxes, fontsize=11, color=TEXT, fontweight="bold")
    y -= 0.08
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "model_parameter_card_r17.png"), dpi=150, bbox_inches="tight")
plt.close()

print("✅ Saved R17 visuals to results/proof_of_work_visuals/r17/")
