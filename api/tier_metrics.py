"""
Offline validation metrics per tier (v6 held-out TEST set).
Cached to results/r18_tier_metrics.json — shown on SOC dashboard as model quality.
"""
import json
import os
import time
from typing import Any, Dict

import joblib
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)

import config as cfg
from src.inference.cascade_r18 import gate_summary, load_gate
from src.models.cnn_gru_v6 import CNNGRUClassifier

CACHE = os.path.join(cfg.BASE_DIR, "results", "r18_tier_metrics.json")


def _gate_summary(X: np.ndarray) -> np.ndarray:
    return gate_summary(X)


def _embed(model, X):
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 512):
            x = torch.FloatTensor(X[i:i + 512])
            x = model.relu(model.bn1(model.conv1(x.transpose(1, 2)))).transpose(1, 2)
            out.append(model.gru(x)[0][:, -1, :].numpy())
    return np.concatenate(out)


def _cls_metrics(y_true, y_pred, labels=None) -> Dict[str, Any]:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision_macro": round(float(precision_score(y_true, y_pred, average="macro", zero_division=0, labels=labels)), 4),
        "recall_macro": round(float(recall_score(y_true, y_pred, average="macro", zero_division=0, labels=labels)), 4),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0, labels=labels)), 4),
    }


def compute_tier_metrics(force: bool = False) -> Dict[str, Any]:
    if os.path.isfile(CACHE) and not force:
        with open(CACHE) as f:
            return json.load(f)

    t0 = time.time()
    le = joblib.load(cfg.ENCODER_PATH)
    classes = list(le.classes_)
    B = classes.index("BENIGN")
    gate = load_gate()
    gate_bi = list(gate.classes_).index(1)

    Xtr = np.load(os.path.join(cfg.SEQ_DIR, "X_train.npy"))
    ytr = np.load(os.path.join(cfg.SEQ_DIR, "y_train.npy"))
    Xte = np.load(os.path.join(cfg.SEQ_DIR, "X_test.npy"))
    yte = np.load(os.path.join(cfg.SEQ_DIR, "y_test.npy"))

    with open(cfg.TIER2_TEMP) as f:
        T = float(json.load(f)["temperature"])

    # ── Tier-1: BENIGN vs ATTACK ─────────────────────────────────────────
    S = _gate_summary(Xte)
    p_ben = gate.predict_proba(S)[:, gate_bi]
    pred_bin = (p_ben < cfg.GATE_THRESHOLD).astype(int)  # 0=benign allow, 1=escalate/attack
    y_bin = (yte != B).astype(int)
    tn = int(((yte == B) & (pred_bin == 0)).sum())
    fp = int(((yte == B) & (pred_bin == 1)).sum())
    fn = int(((yte != B) & (pred_bin == 0)).sum())
    tp = int(((yte != B) & (pred_bin == 1)).sum())
    tier1 = {
        "role": "Fast BENIGN vs ATTACK gate",
        "model": os.path.basename(cfg.TIER1_GATE),
        "task": "binary",
        "test_n": int(len(yte)),
        "accuracy": round((tn + tp) / len(yte), 4),
        "precision": round(tp / (tp + fp), 4) if (tp + fp) else 0,
        "recall": round(tp / (tp + fn), 4) if (tp + fn) else 0,
        "f1": round(2 * tp / (2 * tp + fp + fn), 4) if (2 * tp + fp + fn) else 0,
        "benign_fpr_percent": round(100 * fp / (tn + fp), 2) if (tn + fp) else 0,
        "attack_recall_percent": round(100 * tp / (tp + fn), 2) if (tp + fn) else 0,
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "threshold": cfg.GATE_THRESHOLD,
    }

    # ── Tier-2: 6-class CNN-GRU ──────────────────────────────────────────
    m = CNNGRUClassifier(num_classes=len(classes))
    m.load_state_dict(torch.load(cfg.TIER2_PTH, map_location="cpu"))
    m.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(Xte), 512):
            lg = m(torch.FloatTensor(Xte[i:i + 512])) / T
            preds.extend(torch.softmax(lg, 1).argmax(1).numpy())
    preds = np.array(preds)
    report = classification_report(yte, preds, target_names=classes, output_dict=True, zero_division=0)
    per_class = {
        c: {
            "precision": round(report[c]["precision"], 4),
            "recall": round(report[c]["recall"], 4),
            "f1": round(report[c]["f1-score"], 4),
            "support": int(report[c]["support"]),
        }
        for c in classes
    }
    tier2 = {
        "role": "Deep attack-type classifier (6 classes incl. BENIGN)",
        "model": os.path.basename(cfg.TIER2_PTH),
        "task": "multiclass",
        "test_n": int(len(yte)),
        **_cls_metrics(yte, preds),
        "per_class": per_class,
        "temperature": T,
        "ece_note": "Calibrated on val set; macro-F1 on clean held-out test",
    }

    # ── Tier-3: one-class zero-day ───────────────────────────────────────
    t3_path = cfg.TIER3_ONECLASS
    tier3: Dict[str, Any] = {"role": "Zero-day novelty (Mahalanobis on r18 embeddings)", "enabled": False}
    if os.path.isfile(t3_path):
        t3 = joblib.load(t3_path)
        Etr = _embed(m, Xtr[ytr == B])
        Ete = _embed(m, Xte)
        d = Ete - t3["mu"]
        scores = np.einsum("ij,jk,ik->i", d, t3["inv_cov"], d)
        y_anom = (yte != B).astype(int)
        auc = float(roc_auc_score(y_anom, scores))
        pred_anom = (scores > t3["threshold"]).astype(int)
        tier3 = {
            "role": "Zero-day novelty (Mahalanobis on r18 embeddings)",
            "model": "tier3_oneclass_v6.pkl",
            "task": "anomaly_detection",
            "enabled": True,
            "test_n": int(len(yte)),
            "roc_auc": round(auc, 4),
            "detection_rate_percent": round(100 * pred_anom[y_anom == 1].mean(), 2),
            "false_alarm_percent": round(100 * pred_anom[y_anom == 0].mean(), 2),
            "threshold": round(float(t3["threshold"]), 4),
        }

    # ── Cascade flow (packet counts per stage, v6 TEST) ───────────────────
    N = len(yte)
    benign_m = yte == B
    attack_m = ~benign_m
    gate_allow = p_ben >= cfg.GATE_THRESHOLD
    esc = ~gate_allow

    probs_esc = np.zeros((N, len(classes)), np.float32)
    with torch.no_grad():
        if esc.sum():
            probs_esc[esc] = torch.softmax(m(torch.FloatTensor(Xte[esc])) / T, 1).numpy()
    pred_esc = probs_esc.argmax(1)
    conf_esc = probs_esc.max(1)
    t2_block = esc & (pred_esc != B) & (conf_esc > cfg.BLOCK_THRESHOLD)
    t2_flag = esc & (pred_esc != B) & (conf_esc >= cfg.FLAG_THRESHOLD) & (conf_esc <= cfg.BLOCK_THRESHOLD)
    t2_allow = esc & ~t2_block & ~t2_flag
    allow_candidates = gate_allow | t2_allow

    t3_scores = None
    if os.path.isfile(t3_path):
        Ete = _embed(m, Xte)
        d = Ete - t3["mu"]
        t3_scores = np.einsum("ij,jk,ik->i", d, t3["inv_cov"], d)
        anom = t3_scores > t3["threshold"]
        t3_flag = allow_candidates & anom
        t3_allow = allow_candidates & ~anom
        api_t3_mask = t2_allow & (pred_esc != B)
    else:
        t3_flag = np.zeros(N, dtype=bool)
        t3_allow = allow_candidates.copy()
        api_t3_mask = np.zeros(N, dtype=bool)

    def _stage(name, mask):
        n = int(mask.sum())
        return {
            "name": name,
            "total": n,
            "pct_of_input": round(100 * n / N, 2),
            "benign": int((mask & benign_m).sum()),
            "attack": int((mask & attack_m).sum()),
        }

    cascade_flow = {
        "input_total": N,
        "benign_total": int(benign_m.sum()),
        "attack_total": int(attack_m.sum()),
        "thresholds": {
            "tier1_p_benign_allow": cfg.GATE_THRESHOLD,
            "tier2_block_conf": cfg.BLOCK_THRESHOLD,
            "tier2_flag_conf": cfg.FLAG_THRESHOLD,
        },
        "stages": [
            _stage("stage_0_input", np.ones(N, dtype=bool)),
            _stage("tier1_fast_allow", gate_allow),
            _stage("tier1_escalate_to_tier2", esc),
            _stage("tier2_block", t2_block),
            _stage("tier2_flag", t2_flag),
            _stage("tier2_allow_candidate", t2_allow),
            _stage("tier3_input_allow_candidates", allow_candidates),
            _stage("tier3_flag_anomaly", t3_flag),
            _stage("tier3_final_allow", t3_allow),
            _stage("final_block", t2_block),
            _stage("final_flag", t2_flag | t3_flag),
            _stage("final_allow", t3_allow),
        ],
        "key_rates": {
            "tier2_workload_pct": round(100 * esc.sum() / N, 2),
            "tier3_workload_pct": round(100 * allow_candidates.sum() / N, 2),
            "tier3_api_runtime_pct": round(100 * api_t3_mask.sum() / N, 2),
            "tier3_catches_from_t2_leaks": int((t3_flag & t2_allow & attack_m).sum()),
            "benign_fpr_percent": round(100 * ((t2_flag | t3_flag) & benign_m).sum() / benign_m.sum(), 2),
            "attack_detection_percent": round(100 * ((t2_block | t2_flag | t3_flag) & attack_m).sum() / attack_m.sum(), 2),
        },
        "zero_day_path": (
            "Unknown attack → Tier-1 P(BENIGN)<0.90 → Tier-2 (may predict wrong/low conf) → "
            "if ALLOW leak → Tier-3 Mahalanobis on embedding → FLAG if novelty score > threshold. "
            "Tier-3 does NOT predict attack type — it only flags 'never seen before'."
        ),
        "api_note": (
            "Live API uses cascade_r18.py — same Tier-3 scope as this eval "
            "(all ALLOW candidates, including Tier-1 fast-path)."
        ),
    }

    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": os.path.basename(cfg.DATASET_CSV),
        "split": "v6_sequences TEST (group-disjoint, never seen in training)",
        "test_sequences": int(len(yte)),
        "features": cfg.N_FEATURES,
        "window": cfg.WINDOW_SIZE,
        "classes": classes,
        "compute_seconds": round(time.time() - t0, 2),
        "tier1_gate": tier1,
        "tier2_cnn_gru": tier2,
        "tier3_oneclass": tier3,
        "cascade_flow": cascade_flow,
        "cascade_summary": {
            "benign_fpr_percent": cascade_flow["key_rates"]["benign_fpr_percent"],
            "attack_detection_percent": cascade_flow["key_rates"]["attack_detection_percent"],
            "note": "From cascade_flow.key_rates — same logic as measure_cascade_flow.py",
        },
        "research": {
            "title": "Hybrid-Sentinel NDN AI Firewall",
            "round": 18,
            "pipeline_scripts": [
                "scripts/data/prepare_v6_sequences.py",
                "scripts/training/train_tier1_gate_v6.py",
                "scripts/training/train_cnn_gru_v6.py",
                "scripts/export/export_tier2_classifier_onnx.py",
                "scripts/export/export_tier2_embedding_onnx.py",
                "scripts/export/export_tier3_oneclass.py",
                "scripts/eval/eval_pipeline_v6.py",
                "scripts/eval/measure_cascade_flow.py",
            ],
            "unique_points": [
                "Confidence-gated 3-tier cascade (fast gate → deep CNN-GRU → zero-day net)",
                "Single v6 feature contract across all tiers (17 features, shared scaler)",
                "BENIGN class in Tier-2 fixes 100% false-positive failure mode",
                "Per-alert tier trace for explainability (SOC audit trail)",
            ],
        },
    }
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    with open(CACHE, "w") as f:
        json.dump(payload, f, indent=2)
    return payload
