#!/usr/bin/env python3
"""
Submission-grade metrics & plots — Hybrid-Sentinel R18.

Computes standard IDS/ML metrics per tier from held-out TEST set (n=14,219)
and latency benchmark. Outputs JSON + PDF/PNG tables and charts.

Run from repo root:
    python scripts/analysis/generate_submission_metrics_r18.py

Outputs: results/submission_figures/
         results/submission_metrics_r18.json
         results/submission_metrics_table.tex
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)

import config as cfg
from src.inference.cascade_r18 import CascadeRuntime, gate_summary, load_gate, maha_score
from src.models.cnn_gru_v6 import CNNGRUClassifier

OUT_DIR = _ROOT / "results" / "submission_figures"
OUT_JSON = _ROOT / "results" / "submission_metrics_r18.json"
OUT_TEX = _ROOT / "results" / "submission_metrics_table.tex"
LAT_PATH = _ROOT / "results" / "r18_latency_benchmark.json"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#444444",
    "axes.linewidth": 0.8,
})

C_NAVY = "#243B63"
C_BLUE = "#4E79A7"
C_GREEN = "#59A14F"
C_RED = "#C44E52"
C_ORANGE = "#F28E2B"
C_TEAL = "#499894"
C_GREY = "#6E7781"
C_LGREY = "#F4F6F8"


def style_axes(ax, grid_axis: str | None = "y"):
    """Apply a quiet academic chart style."""
    ax.set_facecolor("white")
    if grid_axis:
        ax.grid(axis=grid_axis, color="#D9DEE5", linewidth=0.6, alpha=0.75)
        ax.set_axisbelow(True)
    ax.tick_params(colors="#333333", width=0.7, length=3)
    for spine in ax.spines.values():
        spine.set_color("#444444")
        spine.set_linewidth(0.8)


def annotate_bars(ax, bars, values, fmt="{:.2f}", y_pad=0.012, fontsize=7):
    y0, y1 = ax.get_ylim()
    pad = (y1 - y0) * y_pad
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + pad,
            fmt.format(val),
            ha="center",
            va="bottom",
            fontsize=fontsize,
            color="#222222",
        )


def save(fig, name: str):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"{name}.{ext}", facecolor="white")
    plt.close(fig)
    print(f"  saved {name}.pdf/.png")


def bin_metrics(y_true, y_pred, positive_label: str = "positive") -> dict:
    """Standard binary classification metrics."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn)
    return {
        "task": "binary",
        "positive_class": positive_label,
        "n": int(len(y_true)),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "specificity": round(tn / (tn + fp), 4) if (tn + fp) else 0.0,
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def embed_batch(model, X):
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 512):
            x = torch.FloatTensor(X[i : i + 512])
            x = model.relu(model.bn1(model.conv1(x.transpose(1, 2)))).transpose(1, 2)
            out.append(model.gru(x)[0][:, -1, :].numpy())
    return np.concatenate(out)


def compute_all_metrics() -> dict:
    """Recompute every metric from v6 TEST — single source of truth."""
    le = joblib.load(cfg.ENCODER_PATH)
    classes = list(le.classes_)
    B = classes.index("BENIGN")

    Xte = np.load(os.path.join(cfg.SEQ_DIR, "X_test.npy"))
    yte = np.load(os.path.join(cfg.SEQ_DIR, "y_test.npy"))
    n = len(yte)
    y_attack = (yte != B).astype(int)

    with open(cfg.TIER2_TEMP) as f:
        T = float(json.load(f)["temperature"])

    # ── Tier-1: escalate (=attack) detection ─────────────────────────────
    gate = load_gate()
    gate_bi = list(gate.classes_).index(1)
    p_ben = gate.predict_proba(gate_summary(Xte))[:, gate_bi]
    pred_escalate = (p_ben < cfg.GATE_THRESHOLD).astype(int)  # 1 = escalate
    tier1 = bin_metrics(y_attack, pred_escalate, positive_label="ATTACK (escalate)")
    tier1["role"] = "Tier-1 Random Forest Gate"
    tier1["model"] = "tier1_gate_v6.pkl"
    tier1["model_size_kb"] = round(os.path.getsize(cfg.TIER1_GATE) / 1024, 1)
    tier1["benign_fpr_percent"] = round(100 * tier1["confusion"]["fp"] / (814), 2)
    tier1["threshold_p_benign_allow"] = cfg.GATE_THRESHOLD
    tier1["note"] = (
        "Binary task: predict ATTACK (escalate to Tier-2). "
        "Benign fast-ALLOW is correct when P(BENIGN)≥0.90."
    )

    # ── Tier-2: 6-class multiclass ───────────────────────────────────────
    model = CNNGRUClassifier(num_classes=len(classes))
    model.load_state_dict(torch.load(cfg.TIER2_PTH, map_location="cpu"))
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, n, 512):
            lg = model(torch.FloatTensor(Xte[i : i + 512])) / T
            preds.extend(torch.softmax(lg, 1).argmax(1).numpy())
    preds = np.array(preds)

    report = classification_report(yte, preds, target_names=classes, output_dict=True, zero_division=0)
    per_class = {
        c: {
            "precision": round(report[c]["precision"], 4),
            "recall": round(report[c]["recall"], 4),
            "f1_score": round(report[c]["f1-score"], 4),
            "support": int(report[c]["support"]),
        }
        for c in classes
    }
    tier2 = {
        "role": "Tier-2 CNN-GRU (6-class)",
        "model": "tier2_cnn_gru_v1_r18.pth",
        "task": "multiclass",
        "n": n,
        "accuracy": round(float(accuracy_score(yte, preds)), 4),
        "precision_macro": round(float(precision_score(yte, preds, average="macro", zero_division=0)), 4),
        "recall_macro": round(float(recall_score(yte, preds, average="macro", zero_division=0)), 4),
        "f1_macro": round(float(f1_score(yte, preds, average="macro", zero_division=0)), 4),
        "precision_weighted": round(float(precision_score(yte, preds, average="weighted", zero_division=0)), 4),
        "recall_weighted": round(float(recall_score(yte, preds, average="weighted", zero_division=0)), 4),
        "f1_weighted": round(float(f1_score(yte, preds, average="weighted", zero_division=0)), 4),
        "per_class": per_class,
        "temperature": T,
        "eval_scope": "All test sequences (standard classifier eval)",
    }

    # Tier-2 on escalated-only subset (operational workload)
    esc = pred_escalate.astype(bool)
    if esc.sum() > 0:
        tier2_esc = {
            "n": int(esc.sum()),
            "accuracy": round(float(accuracy_score(yte[esc], preds[esc])), 4),
            "f1_macro": round(float(f1_score(yte[esc], preds[esc], average="macro", zero_division=0)), 4),
            "eval_scope": "Escalated sequences only (post Tier-1)",
        }
    else:
        tier2_esc = {"n": 0}

    # ── Tier-3: anomaly detection ────────────────────────────────────────
    t3 = joblib.load(cfg.TIER3_ONECLASS)
    Ete = embed_batch(model, Xte)
    d = Ete - t3["mu"]
    scores = np.einsum("ij,jk,ik->i", d, t3["inv_cov"], d)
    thr = float(t3["threshold"])
    pred_anom_full = (scores > thr).astype(int)

    # Operational scope: allow candidates (matches measure_cascade_flow / API)
    gate_allow = p_ben >= cfg.GATE_THRESHOLD
    esc_mask = ~gate_allow
    probs_pre = np.zeros((n, len(classes)), np.float32)
    with torch.no_grad():
        if esc_mask.sum():
            probs_pre[esc_mask] = torch.softmax(model(torch.FloatTensor(Xte[esc_mask])) / T, 1).numpy()
    pred_cls_pre = probs_pre.argmax(1)
    conf_pre = probs_pre.max(1)
    t2_block = esc_mask & (pred_cls_pre != B) & (conf_pre > cfg.BLOCK_THRESHOLD)
    t2_flag = esc_mask & (pred_cls_pre != B) & (conf_pre >= cfg.FLAG_THRESHOLD) & (conf_pre <= cfg.BLOCK_THRESHOLD)
    t2_allow = esc_mask & ~t2_block & ~t2_flag
    allow_cand = gate_allow | t2_allow
    y_ac = y_attack[allow_cand]
    pred_ac = pred_anom_full[allow_cand]

    fpr, tpr, _ = roc_curve(y_attack, scores)
    roc_auc = float(auc(fpr, tpr))
    tier3_op = bin_metrics(y_ac, pred_ac, positive_label="ANOMALY on allow-candidates")
    tier3 = {
        **tier3_op,
        "role": "Tier-3 Mahalanobis One-Class",
        "model": "tier3_oneclass_v6.pkl",
        "roc_auc_full_test": round(roc_auc, 4),
        "threshold_maha": round(thr, 4),
        "eval_scope_allow_candidates_n": int(allow_cand.sum()),
        "benign_false_alarm_percent": round(100 * pred_ac[y_ac == 0].mean(), 2) if (y_ac == 0).any() else 0.0,
        "attack_detection_percent": round(100 * pred_ac[y_ac == 1].mean(), 2) if (y_ac == 1).any() else 0.0,
        "note": (
            "Primary Tier-3 metrics on ALLOW-candidates only (production scope). "
            "ROC-AUC on full test set for ranking quality. "
            "Re-run export_tier3_oneclass.py after any Tier-2 retrain."
        ),
    }
    tier3_roc = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}

    # ── Full cascade E2E (measure_cascade_flow logic) ────────────────────
    gate_allow = p_ben >= cfg.GATE_THRESHOLD
    esc_mask = ~gate_allow

    probs = np.zeros((n, len(classes)), np.float32)
    with torch.no_grad():
        if esc_mask.sum():
            probs[esc_mask] = torch.softmax(model(torch.FloatTensor(Xte[esc_mask])) / T, 1).numpy()
    pred_cls = probs.argmax(1)
    conf = probs.max(1)

    action = np.array(["ALLOW"] * n, dtype=object)
    for i in range(n):
        if gate_allow[i]:
            action[i] = "ALLOW"
        elif pred_cls[i] == B:
            action[i] = "ALLOW"
        elif conf[i] > cfg.BLOCK_THRESHOLD:
            action[i] = "BLOCK"
        elif conf[i] >= cfg.FLAG_THRESHOLD:
            action[i] = "FLAG"
        else:
            action[i] = "ALLOW"

    allow_cand = gate_allow | ((action == "ALLOW") & ~gate_allow)
    anom_mask = scores > thr
    t3_flag = allow_cand & anom_mask
    action[t3_flag] = "FLAG"

    malicious_pred = np.isin(action, ["BLOCK", "FLAG"]).astype(int)
    cascade = bin_metrics(y_attack, malicious_pred, positive_label="MALICIOUS (BLOCK|FLAG)")
    cascade["role"] = "Full 3-Tier Cascade (E2E)"
    cascade["attack_detection_rate_percent"] = round(100 * cascade["recall"], 2)
    cascade["benign_fpr_percent"] = round(100 * cascade["confusion"]["fp"] / 814, 2)
    cascade["eval_script"] = "measure_cascade_flow.py (equivalent logic)"
    cascade["note"] = (
        "E2E security metric: attack=detected if final action is BLOCK or FLAG. "
        "Test set: group-disjoint v6_sequences, never seen in training."
    )

    # ── Throughput / latency ─────────────────────────────────────────────
    perf = {}
    if LAT_PATH.is_file():
        with open(LAT_PATH) as f:
            lat = json.load(f)
        for key, label in [
            ("tier1_gate", "Tier-1 Gate"),
            ("tier2_onnx", "Tier-2 ONNX"),
            ("tier3_oneclass", "Tier-3 One-Class"),
            ("full_cascade", "Full Cascade"),
        ]:
            r = lat["results"][key]
            avg_ms = r["avg_ms"]
            perf[label] = {
                "latency_avg_ms": r["avg_ms"],
                "latency_p99_ms": r["p99_ms"],
                "latency_min_ms": r["min_ms"],
                "throughput_pps": r["pps"],
                "throughput_seq_per_s": round(1000 / avg_ms, 1) if avg_ms else 0,
                "benchmark_n": lat.get("n_packets", 2000),
                "hardware": "CPU (single-threaded gate n_jobs=1)",
            }
        perf["baseline_g_scaler_ms"] = lat.get("g_scaler_baseline_avg_ms", 13.0)

    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dataset": "combined_dataset_v5_final.csv",
        "split": "v6_sequences TEST (group-disjoint 70/15/15)",
        "test_sequences": n,
        "benign_test": 814,
        "attack_test": 13405,
        "features": cfg.N_FEATURES,
        "window": cfg.WINDOW_SIZE,
        "classes": classes,
        "tier1_gate": tier1,
        "tier2_cnn_gru": tier2,
        "tier2_escalated_subset": tier2_esc,
        "tier3_oneclass": tier3,
        "tier3_roc": {"auc": tier3["roc_auc_full_test"], "n_points": len(fpr)},
        "cascade_end_to_end": cascade,
        "performance": perf,
        "methodology_notes": [
            "All classification metrics on held-out TEST unless stated otherwise.",
            "Latency: 2000-iteration micro-benchmark, results/r18_latency_benchmark.json.",
            "Throughput (PPS) = sequences processed per second in benchmark.",
            "Tier-3 is anomaly detection — use TPR/FPR/ROC-AUC, not multiclass accuracy.",
            "Do not report Tier-2 metrics as final production score without cascade E2E.",
        ],
        "_roc_arrays": tier3_roc,
        "_cm_tier2": confusion_matrix(yte, preds).tolist(),
        "_classes": classes,
    }


# ── Plot: summary comparison table (figure) ──────────────────────────────
def fig_summary_table(m: dict):
    rows = [
        ["Tier-1 Gate", "Binary (escalate)", str(m["tier1_gate"]["n"]),
         f"{m['tier1_gate']['accuracy']:.4f}", f"{m['tier1_gate']['precision']:.4f}",
         f"{m['tier1_gate']['recall']:.4f}", f"{m['tier1_gate']['f1_score']:.4f}", "—"],
        ["Tier-2 CNN-GRU", "6-class", str(m["tier2_cnn_gru"]["n"]),
         f"{m['tier2_cnn_gru']['accuracy']:.4f}", f"{m['tier2_cnn_gru']['precision_macro']:.4f}",
         f"{m['tier2_cnn_gru']['recall_macro']:.4f}", f"{m['tier2_cnn_gru']['f1_macro']:.4f}", "—"],
        ["Tier-3 One-Class", "Anomaly", str(m["tier3_oneclass"]["n"]),
         f"{m['tier3_oneclass']['accuracy']:.4f}", f"{m['tier3_oneclass']['precision']:.4f}",
         f"{m['tier3_oneclass']['recall']:.4f}", f"{m['tier3_oneclass']['f1_score']:.4f}",
         f"AUC={m['tier3_oneclass'].get('roc_auc_full_test', m['tier3_oneclass'].get('roc_auc', 0)):.4f}"],
        ["Full Cascade", "E2E security", str(m["cascade_end_to_end"]["n"]),
         f"{m['cascade_end_to_end']['accuracy']:.4f}", f"{m['cascade_end_to_end']['precision']:.4f}",
         f"{m['cascade_end_to_end']['recall']:.4f}", f"{m['cascade_end_to_end']['f1_score']:.4f}",
         f"FPR={m['cascade_end_to_end']['benign_fpr_percent']:.2f}%"],
    ]
    cols = ["Component", "Task", "N", "Accuracy", "Precision", "Recall", "F1", "Extra"]

    fig, ax = plt.subplots(figsize=(9.4, 2.8), facecolor="white")
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.2)
    tbl.scale(1, 1.75)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#D0D7DE")
        cell.set_linewidth(0.5)
        if row == 0:
            cell.set_facecolor(C_NAVY)
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor(C_LGREY)
    ax.set_title(
        "Per-Tier and End-to-End Metrics (Held-Out Test)",
        fontsize=11, fontweight="bold", pad=14,
    )
    fig.text(0.5, 0.02,
             "Macro precision/recall/F1 for Tier-2. E2E: malicious = BLOCK or FLAG.",
             ha="center", fontsize=8, color=C_GREY)
    save(fig, "fig_submission_metrics_summary_table")


# ── Plot: tier metric comparison bars ────────────────────────────────────
def fig_tier_metric_bars(m: dict):
    tiers = ["Tier-1\nGate", "Tier-2\n(Macro)", "Tier-3\nAnomaly", "Cascade\n(E2E)"]
    acc = [m["tier1_gate"]["accuracy"], m["tier2_cnn_gru"]["accuracy"],
           m["tier3_oneclass"]["accuracy"], m["cascade_end_to_end"]["accuracy"]]
    prec = [m["tier1_gate"]["precision"], m["tier2_cnn_gru"]["precision_macro"],
            m["tier3_oneclass"]["precision"], m["cascade_end_to_end"]["precision"]]
    rec = [m["tier1_gate"]["recall"], m["tier2_cnn_gru"]["recall_macro"],
           m["tier3_oneclass"]["recall"], m["cascade_end_to_end"]["recall"]]
    f1 = [m["tier1_gate"]["f1_score"], m["tier2_cnn_gru"]["f1_macro"],
          m["tier3_oneclass"]["f1_score"], m["cascade_end_to_end"]["f1_score"]]

    x = np.arange(4)
    w = 0.2
    fig, ax = plt.subplots(figsize=(6.2, 3.4), facecolor="white")
    ax.bar(x - 1.5 * w, acc, w, label="Accuracy", color=C_NAVY, alpha=0.92)
    ax.bar(x - 0.5 * w, prec, w, label="Precision", color=C_BLUE, alpha=0.92)
    ax.bar(x + 0.5 * w, rec, w, label="Recall", color=C_TEAL, alpha=0.92)
    ax.bar(x + 1.5 * w, f1, w, label="F1", color=C_GREEN, alpha=0.92)
    ax.set_xticks(x)
    ax.set_xticklabels(tiers, fontsize=8)
    ax.set_ylim(0.88, 1.025)
    ax.set_ylabel("Score")
    ax.set_title("Per-Tier Classification Metrics")
    ax.legend(fontsize=7.5, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.18),
              frameon=False)
    style_axes(ax)
    fig.subplots_adjust(bottom=0.25)
    save(fig, "fig_submission_tier_metric_comparison")


# ── Plot: Tier-2 per-class heatmap ───────────────────────────────────────
def fig_tier2_heatmap(m: dict, cm: list, classes: list):
    metrics = ["precision", "recall", "f1_score"]
    data = np.array([
        [m["tier2_cnn_gru"]["per_class"][c][met] for c in classes]
        for met in metrics
    ])
    fig, ax = plt.subplots(figsize=(6.0, 2.8), facecolor="white")
    im = ax.imshow(data, cmap="Blues", vmin=0.9, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels([c.replace("_", "\n") for c in classes], fontsize=7)
    ax.set_yticks(range(3))
    ax.set_yticklabels(["Precision", "Recall", "F1"], fontsize=8)
    for i in range(3):
        for j in range(len(classes)):
            ax.text(j, i, f"{data[i, j]:.3f}", ha="center", va="center", fontsize=7,
                    color="white" if data[i, j] > 0.975 else "#1F2937")
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, label="Score", pad=0.02)
    cbar.ax.tick_params(labelsize=7)
    ax.set_title(f"Tier-2 Per-Class Metrics (Macro F1 = {m['tier2_cnn_gru']['f1_macro']:.4f})")
    for spine in ax.spines.values():
        spine.set_visible(False)
    save(fig, "fig_submission_tier2_per_class_heatmap")


# ── Plot: Tier-1 confusion matrix ────────────────────────────────────────
def fig_tier1_confusion(m: dict):
    c = m["tier1_gate"]["confusion"]
    cm = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
    fig, ax = plt.subplots(figsize=(4.6, 3.4), facecolor="white")
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Allow", "Escalate"], fontsize=8)
    ax.set_yticklabels(["Benign", "Attack"], fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            ax.text(j, i, f"{val:,}", ha="center", va="center", fontsize=13,
                    fontweight="bold", color="white" if val > cm.max() * 0.5 else "#1F2937")
    ax.set_title("Tier-1 Gate Confusion Matrix")
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.colorbar(im, ax=ax, shrink=0.75, pad=0.03)
    save(fig, "fig_submission_tier1_confusion_matrix")


# ── Plot: Tier-2 confusion matrix ────────────────────────────────────────
def fig_tier2_confusion(cm: list, classes: list):
    cm_arr = np.array(cm)
    cm_norm = cm_arr.astype(float) / cm_arr.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.4, 4.4), facecolor="white")
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels([c.replace("_", "\n") for c in classes], rotation=0, ha="center", fontsize=7)
    ax.set_yticklabels([c.replace("_", "\n") for c in classes], fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(len(classes)):
        for j in range(len(classes)):
            val = cm_norm[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if val > 0.55 else "#1F2937")
    ax.set_title("Tier-2 Confusion Matrix")
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.colorbar(im, ax=ax, shrink=0.72, pad=0.03)
    save(fig, "fig_submission_tier2_confusion_matrix")


# ── Plot: Tier-3 ROC ─────────────────────────────────────────────────────
def fig_tier3_roc(m: dict, roc: dict):
    fpr = np.array(roc["fpr"])
    tpr = np.array(roc["tpr"])
    fig, ax = plt.subplots(figsize=(4.8, 3.8), facecolor="white")
    ax.plot(fpr, tpr, color=C_BLUE, lw=2,
            label=f"ROC (AUC={m['tier3_oneclass'].get('roc_auc_full_test', 0):.4f})")
    ax.plot([0, 1], [0, 1], "--", color=C_GREY, lw=1, label="Random baseline")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Tier-3 ROC Curve")
    ax.legend(fontsize=8, loc="lower right", frameon=False)
    style_axes(ax, grid_axis="both")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    save(fig, "fig_submission_tier3_roc_curve")


# ── Plot: latency + throughput ───────────────────────────────────────────
def fig_latency_throughput(m: dict):
    perf = m.get("performance", {})
    if not perf:
        return
    labels = [k for k in perf if k != "baseline_g_scaler_ms" and isinstance(perf[k], dict)]
    lat = [perf[k]["latency_avg_ms"] for k in labels]
    p99 = [perf[k]["latency_p99_ms"] for k in labels]
    pps = [perf[k]["throughput_pps"] for k in labels]

    short_labels = ["Tier-1\nGate", "Tier-2\nONNX", "Tier-3\nOne-Class", "Full\nCascade"]
    x = np.arange(len(labels))
    baseline = perf.get("baseline_g_scaler_ms", 13)

    fig, (ax_lat, ax_thr) = plt.subplots(
        2, 1, figsize=(6.8, 5.2), facecolor="white",
        gridspec_kw={"height_ratios": [2.2, 1.25], "hspace": 0.42},
    )

    # Panel A: latency. Keep p99 next to mean so the reader can see tail cost.
    w = 0.34
    bars_avg = ax_lat.bar(
        x - w / 2, lat, w, label="Mean latency",
        color=C_NAVY, edgecolor="#172A45", linewidth=0.6, alpha=0.95,
    )
    bars_p99 = ax_lat.bar(
        x + w / 2, p99, w, label="p99 latency",
        color=C_ORANGE, edgecolor="#A85F16", linewidth=0.6, alpha=0.95,
    )
    ax_lat.axhline(
        baseline, color=C_RED, ls=(0, (5, 3)), lw=1.2,
        label=f"Reference baseline ({baseline:.1f} ms)",
    )
    ax_lat.set_ylabel("Latency (ms)")
    ax_lat.set_xticks(x)
    ax_lat.set_xticklabels(short_labels, fontsize=8)
    ax_lat.set_ylim(0, max(max(p99), baseline) * 1.20)
    ax_lat.set_title("R18 Runtime Performance on CPU", loc="left", fontsize=11, pad=10)
    annotate_bars(ax_lat, bars_avg, lat, fmt="{:.2f}", fontsize=7)
    annotate_bars(ax_lat, bars_p99, p99, fmt="{:.2f}", fontsize=7)
    ax_lat.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.16),
        ncol=3, fontsize=7.5, frameon=False,
    )
    style_axes(ax_lat)

    # Panel B: throughput. Separate axis avoids the misleading dual-axis zig-zag.
    bars_thr = ax_thr.bar(
        x, pps, 0.52, color=C_GREEN, edgecolor="#39723A",
        linewidth=0.6, alpha=0.95,
    )
    ax_thr.set_ylabel("Sequences/s")
    ax_thr.set_xticks(x)
    ax_thr.set_xticklabels(short_labels, fontsize=8)
    ax_thr.set_ylim(0, max(pps) * 1.18)
    ax_thr.set_title("Throughput", loc="left", fontsize=9, pad=6)
    annotate_bars(ax_thr, bars_thr, pps, fmt="{:,.0f}", fontsize=7)
    style_axes(ax_thr)

    fig.text(
        0.5, 0.015,
        "Lower latency is better in the top panel; higher throughput is better in the bottom panel.",
        ha="center", fontsize=7.2, color=C_GREY, style="italic",
    )
    fig.subplots_adjust(bottom=0.11)
    save(fig, "fig_submission_latency_throughput")


# ── Plot: E2E security rates ─────────────────────────────────────────────
def fig_e2e_security(m: dict):
    c = m["cascade_end_to_end"]
    labels = ["Attack\nDetection", "Benign\nFPR", "E2E\nPrecision", "E2E\nF1"]
    vals = [
        c["attack_detection_rate_percent"],
        c["benign_fpr_percent"],
        c["precision"] * 100,
        c["f1_score"] * 100,
    ]
    colors = [C_GREEN, C_RED, C_BLUE, C_TEAL]
    fig, ax = plt.subplots(figsize=(5.6, 3.4), facecolor="white")
    bars = ax.bar(labels, vals, color=colors, edgecolor="white", linewidth=0.8, width=0.58)
    ax.set_ylabel("Rate (%)")
    ax.set_ylim(0, 112)
    ax.set_title("End-to-End Security Metrics")
    for bar, v in zip(bars, vals):
        y = bar.get_height() + 2.2 if v > 5 else bar.get_height() + 3.5
        ax.text(bar.get_x() + bar.get_width() / 2, y,
                f"{v:.2f}%", ha="center", va="bottom", fontsize=8)
    style_axes(ax)
    ax.tick_params(axis="x", pad=6)
    save(fig, "fig_submission_e2e_security_rates")


# ── LaTeX table export ───────────────────────────────────────────────────
def write_latex_table(m: dict):
    t1, t2, t3, ce = m["tier1_gate"], m["tier2_cnn_gru"], m["tier3_oneclass"], m["cascade_end_to_end"]
    perf = m.get("performance", {})
    tex = r"""\begin{table}[!t]
\caption{Hybrid-Sentinel R18 — Standard Evaluation Metrics (Held-Out Test, $n=14{,}219$)}
\label{tab:submission_metrics}
\centering
\small
\begin{tabular}{lcccccc}
\toprule
\textbf{Component} & \textbf{Acc.} & \textbf{Prec.} & \textbf{Rec.} & \textbf{F1} & \textbf{Latency (ms)} & \textbf{Throughput} \\
\midrule
"""
    lat_map = {
        "Tier-1 Gate": perf.get("Tier-1 Gate", {}),
        "Tier-2 ONNX": perf.get("Tier-2 ONNX", {}),
        "Tier-3 One-Class": perf.get("Tier-3 One-Class", {}),
        "Full Cascade": perf.get("Full Cascade", {}),
    }
    rows = [
        ("Tier-1 Gate (binary)", t1["accuracy"], t1["precision"], t1["recall"], t1["f1_score"], "Tier-1 Gate"),
        ("Tier-2 CNN-GRU (macro)", t2["accuracy"], t2["precision_macro"], t2["recall_macro"], t2["f1_macro"], "Tier-2 ONNX"),
        ("Tier-3 Anomaly", t3["accuracy"], t3["precision"], t3["recall"], t3["f1_score"], "Tier-3 One-Class"),
        ("Full Cascade (E2E)", ce["accuracy"], ce["precision"], ce["recall"], ce["f1_score"], "Full Cascade"),
    ]
    for name, acc, prec, rec, f1, pk in rows:
        p = lat_map.get(pk, {})
        lat_s = f"{p.get('latency_avg_ms', '—')}" if p else "—"
        thr_s = f"{p.get('throughput_pps', '—')} seq/s" if p else "—"
        tex += f"{name} & {acc:.4f} & {prec:.4f} & {rec:.4f} & {f1:.4f} & {lat_s} & {thr_s} \\\\\n"
    tex += r"""\midrule
\multicolumn{7}{l}{\textit{E2E: attack detection = """ + f"{ce['attack_detection_rate_percent']:.1f}" + r"""\%, benign FPR = """ + f"{ce['benign_fpr_percent']:.2f}" + r"""\%, Tier-3 ROC-AUC = """ + f"{t3.get('roc_auc_full_test', t3.get('roc_auc', 0)):.4f}" + r"""} \\
\bottomrule
\end{tabular}
\end{table}
"""
    OUT_TEX.write_text(tex)
    print(f"  saved {OUT_TEX}")


def main():
    print("=" * 60)
    print("  Submission Metrics & Plots — R18")
    print("=" * 60)
    t0 = time.time()
    m = compute_all_metrics()

    # Strip large arrays from JSON
    roc = m.pop("_roc_arrays")
    cm = m.pop("_cm_tier2")
    classes = m.pop("_classes")
    OUT_JSON.write_text(json.dumps(m, indent=2))
    print(f"  saved {OUT_JSON}")

    print("\nGenerating figures ...")
    fig_summary_table(m)
    fig_tier_metric_bars(m)
    fig_tier2_heatmap(m, cm, classes)
    fig_tier1_confusion(m)
    fig_tier2_confusion(cm, classes)
    fig_tier3_roc(m, roc)
    fig_latency_throughput(m)
    fig_e2e_security(m)
    write_latex_table(m)

    print(f"\nDone in {time.time() - t0:.1f}s")
    print(f"Figures: {OUT_DIR}")
    print("\nKey E2E results:")
    ce = m["cascade_end_to_end"]
    print(f"  Attack detection: {ce['attack_detection_rate_percent']}%")
    print(f"  Benign FPR:       {ce['benign_fpr_percent']}%")
    print(f"  Tier-2 macro-F1:  {m['tier2_cnn_gru']['f1_macro']}")
    print(f"  Full cascade latency: {m['performance'].get('Full Cascade', {}).get('latency_avg_ms', 'N/A')} ms")


if __name__ == "__main__":
    main()
