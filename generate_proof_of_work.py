"""
Proof of Work — NDN AI Firewall (Round 16)
Generates: Confusion Matrix Heatmap, Per-Class Metrics Bar Chart,
           Precision/Recall/F1 Summary, Model Parameter Card
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from datetime import datetime

# ────────────────────────────────────────────────────────────────
# 1. LOAD Round 16 metrics from log
# ────────────────────────────────────────────────────────────────
LOG_FILE   = "results/proof_of_work_log.json"
OUT_DIR    = "results/proof_of_work_visuals"
os.makedirs(OUT_DIR, exist_ok=True)

with open(LOG_FILE) as f:
    logs = json.load(f)

# Pick the R16 training entry
r16 = next((e for e in reversed(logs) if e.get("round") == 16 and "confusion_matrix" in e), None)
if r16 is None:
    raise RuntimeError("Round 16 entry with confusion_matrix not found in log.")

cm_data     = np.array(r16["confusion_matrix"])
class_names = ["BRUTE_FORCE", "DDOS_HTTP_FLOOD", "SLOW_HTTP"]
pcm         = r16["per_class_metrics"]   # dict keyed by class name

precision = [pcm[c]["precision"] for c in class_names]
recall    = [pcm[c]["recall"]    for c in class_names]
f1        = [pcm[c]["f1"]        for c in class_names]
support   = [pcm[c]["support"]   for c in class_names]

macro_f1  = r16["test_f1_macro"]
val_f1    = r16["val_f1_macro"]
temperature = r16.get("temperature", "-")
thresholds  = r16.get("thresholds_used", {})
epoch_best  = r16.get("epoch_best", "-")

# ────────────────────────────────────────────────────────────────
# 2. THEME
# ────────────────────────────────────────────────────────────────
DARK_BG   = "#0d1117"
CARD_BG   = "#161b22"
ACCENT    = "#58a6ff"
GREEN     = "#3fb950"
ORANGE    = "#d29922"
RED       = "#f85149"
TEXT      = "#e6edf3"
SUBTEXT   = "#8b949e"
PALETTE   = [ACCENT, GREEN, ORANGE]

plt.rcParams.update({
    "figure.facecolor":  DARK_BG,
    "axes.facecolor":    CARD_BG,
    "axes.edgecolor":    SUBTEXT,
    "axes.labelcolor":   TEXT,
    "xtick.color":       SUBTEXT,
    "ytick.color":       SUBTEXT,
    "text.color":        TEXT,
    "font.family":       "DejaVu Sans",
    "grid.color":        "#21262d",
    "grid.linestyle":    "--",
    "grid.alpha":        0.5,
})

short = ["BRUTE\nFORCE", "DDOS\nHTTP", "SLOW\nHTTP"]

# ────────────────────────────────────────────────────────────────
# 3. FIGURE 1 — Confusion Matrix Heatmap
# ────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5.5))
fig.patch.set_facecolor(DARK_BG)

# Normalize for colour but display raw counts
cm_norm = cm_data.astype(float) / cm_data.sum(axis=1, keepdims=True)
sns.heatmap(
    cm_norm, annot=cm_data, fmt="d",
    cmap=sns.light_palette(ACCENT, as_cmap=True),
    linewidths=1.5, linecolor=DARK_BG,
    cbar_kws={"label": "Normalized %"},
    ax=ax, xticklabels=short, yticklabels=short,
    annot_kws={"size": 16, "weight": "bold", "color": DARK_BG}
)
ax.set_xlabel("Predicted Label", fontsize=12, labelpad=10)
ax.set_ylabel("Actual Label",    fontsize=12, labelpad=10)
ax.set_title("Confusion Matrix — Round 16\n(3-Class Attack Detection)",
             fontsize=14, fontweight="bold", color=TEXT, pad=15)
fig.text(0.98, 0.02, f"Macro F1: {macro_f1:.4f}", ha="right", fontsize=10,
         color=GREEN, fontweight="bold")
plt.tight_layout()
path_cm = os.path.join(OUT_DIR, "confusion_matrix.png")
plt.savefig(path_cm, dpi=150, bbox_inches="tight")
plt.close()
print(f"✅ Saved: {path_cm}")

# ────────────────────────────────────────────────────────────────
# 4. FIGURE 2 — Per-Class Metrics Bar Chart
# ────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5.5))
fig.patch.set_facecolor(DARK_BG)

x = np.arange(len(class_names))
w = 0.24
b1 = ax.bar(x - w, precision, w, label="Precision", color=ACCENT,   alpha=0.9, zorder=3)
b2 = ax.bar(x,     recall,    w, label="Recall",    color=GREEN,   alpha=0.9, zorder=3)
b3 = ax.bar(x + w, f1,        w, label="F1-Score",  color=ORANGE,  alpha=0.9, zorder=3)

for bars in [b1, b2, b3]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005, f"{h:.3f}",
                ha="center", va="bottom", fontsize=8.5, color=TEXT, fontweight="bold")

ax.set_ylim(0, 1.12)
ax.set_xticks(x)
ax.set_xticklabels(short, fontsize=11)
ax.set_ylabel("Score", fontsize=12)
ax.set_title("Per-Class Metrics — Round 16\n(Precision / Recall / F1-Score)",
             fontsize=14, fontweight="bold", color=TEXT, pad=15)
ax.legend(facecolor=CARD_BG, edgecolor=SUBTEXT, labelcolor=TEXT, fontsize=11)
ax.yaxis.grid(True, zorder=0)
ax.set_axisbelow(True)
fig.text(0.98, 0.02, f"Val F1: {val_f1:.4f} | Test F1: {macro_f1:.4f}",
         ha="right", fontsize=9, color=SUBTEXT)
plt.tight_layout()
path_bar = os.path.join(OUT_DIR, "per_class_metrics.png")
plt.savefig(path_bar, dpi=150, bbox_inches="tight")
plt.close()
print(f"✅ Saved: {path_bar}")

# ────────────────────────────────────────────────────────────────
# 5. FIGURE 3 — Support Histogram
# ────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4.5))
fig.patch.set_facecolor(DARK_BG)

bars = ax.bar(short, support, color=PALETTE, alpha=0.9, width=0.5, zorder=3)
for bar, count in zip(bars, support):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
            str(count), ha="center", va="bottom", fontsize=13,
            color=TEXT, fontweight="bold")
ax.set_ylabel("Sample Count (Test Set)", fontsize=12)
ax.set_title("Test Set Class Distribution — Round 16", fontsize=14,
             fontweight="bold", color=TEXT, pad=15)
ax.yaxis.grid(True, zorder=0)
ax.set_axisbelow(True)
plt.tight_layout()
path_hist = os.path.join(OUT_DIR, "class_distribution.png")
plt.savefig(path_hist, dpi=150, bbox_inches="tight")
plt.close()
print(f"✅ Saved: {path_hist}")

# ────────────────────────────────────────────────────────────────
# 6. FIGURE 4 — Model Parameter Card
# ────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))
fig.patch.set_facecolor(DARK_BG)
ax.axis("off")

params = [
    ("Architecture",    "Conv1D(17→64) → GRU(64→128, 2L) → Linear(128→3)"),
    ("Input Features",  "17 (packet-level flow features)"),
    ("Sequence Window", "20 packets per sequence"),
    ("Loss Function",   "Focal Loss  (γ=2,  α: BF=0.6, DDoS=0.8, Slow=1.5)"),
    ("Optimizer",       "AdamW  (lr=5e-5,  weight_decay=1e-4)"),
    ("Scheduler",       "ReduceLROnPlateau (mode=max, patience=3)"),
    ("Dropout",         "0.3"),
    ("Batch Size",      "64 (train) / 256 (val+test)"),
    ("Best Epoch",      str(epoch_best)),
    ("Temperature T",   str(round(float(temperature), 4))),
    ("Thresholds",      f"BF={thresholds.get('BRUTE_FORCE',0.3)}  DDoS={thresholds.get('DDOS_HTTP_FLOOD',0.3)}  Slow={thresholds.get('SLOW_HTTP',0.3)}"),
    ("Macro F1 (Test)", f"{macro_f1:.4f}  ← Round 16 First Valid Result"),
]

ax.text(0.5, 0.97, "🧠  Hybrid-Sentinel  |  Model Parameter Card",
        transform=ax.transAxes, fontsize=15, fontweight="bold", color=ACCENT,
        ha="center", va="top")
ax.text(0.5, 0.91, "Tier-2 CNN-GRU v4  —  NDN AI Firewall Project",
        transform=ax.transAxes, fontsize=10, color=SUBTEXT,
        ha="center", va="top")

y = 0.83
for key, val in params:
    ax.text(0.04, y, f"▸  {key}:", transform=ax.transAxes,
            fontsize=10, color=SUBTEXT, va="top")
    ax.text(0.37, y, val, transform=ax.transAxes,
            fontsize=10, color=TEXT, va="top", fontweight="bold")
    y -= 0.067

ax.text(0.5, 0.02, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        transform=ax.transAxes, fontsize=8, color=SUBTEXT, ha="center")

plt.tight_layout()
path_card = os.path.join(OUT_DIR, "model_parameter_card.png")
plt.savefig(path_card, dpi=150, bbox_inches="tight")
plt.close()
print(f"✅ Saved: {path_card}")

print(f"\n{'='*55}")
print(f"  ALL PROOF-OF-WORK VISUALS GENERATED → {OUT_DIR}/")
print(f"{'='*55}")
