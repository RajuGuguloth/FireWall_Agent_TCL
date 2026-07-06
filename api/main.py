"""
NDN AI Firewall API — R18 production cascade.

All inference logic lives in cascade_r18.py (shared with offline eval).
"""
import os
import sys
import time
import json
import threading
from typing import List, Dict, Any, Optional

import joblib
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config as cfg
from src.inference.cascade_r18 import CascadeRuntime
from api.alert_store import init_db, log_alert, fetch_alerts, alert_counts, DB_PATH

APP_START_TIME = time.time()
_STATS_LOCK = threading.Lock()


def _require(path: str, label: str) -> None:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing {label}: {path}")


def _validate_contract() -> None:
    for p, lbl in [
        (cfg.SCALER_PATH, "v6 scaler"),
        (cfg.ENCODER_PATH, "v6 encoder"),
        (cfg.TIER1_GATE, "Tier-1 gate"),
        (cfg.TIER2_ONNX, "Tier-2 ONNX"),
        (cfg.TIER2_TEMP, "Tier-2 temperature"),
    ]:
        _require(p, lbl)
    scaler = joblib.load(cfg.SCALER_PATH)
    if getattr(scaler, "n_features_in_", None) != cfg.N_FEATURES:
        raise ValueError(f"Scaler feature count != {cfg.N_FEATURES}")
    encoder = joblib.load(cfg.ENCODER_PATH)
    n_cls = len(encoder.classes_)
    logits_dim = ort.InferenceSession(cfg.TIER2_ONNX).get_outputs()[0].shape[-1]
    if logits_dim != n_cls:
        raise ValueError(f"ONNX classes {logits_dim} != encoder {n_cls}")


_validate_contract()
RT = CascadeRuntime.load()
init_db()
ATTACK_CLASSES = [c for c in RT.classes if c != "BENIGN"]

stats: Dict[str, Any] = {
    "total_sequences": 0,
    "tier1_fast_allowed": 0,
    "tier2_escalated": 0,
    "tier3_checked": 0,
    "total_blocked": 0,
    "total_flagged": 0,
    "total_allowed": 0,
    "latency_sum_ms": 0.0,
    "attacks_detected": {c: 0 for c in ATTACK_CLASSES},
}

app = FastAPI(title="NDN AI Firewall API", version="r18")

_DASHBOARD = os.path.join(os.path.dirname(__file__), "static", "dashboard.html")
_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


class PredictRequest(BaseModel):
    sequence: List[List[float]] = Field(..., min_length=cfg.WINDOW_SIZE, max_length=cfg.WINDOW_SIZE)

    @field_validator("sequence")
    @classmethod
    def validate_packets(cls, seq: List[List[float]]) -> List[List[float]]:
        for i, pkt in enumerate(seq):
            if len(pkt) != cfg.N_FEATURES:
                raise ValueError(
                    f"Packet {i} must have {cfg.N_FEATURES} features, got {len(pkt)}"
                )
            if not all(np.isfinite(pkt)):
                raise ValueError(f"Packet {i} contains non-finite values")
        return seq


class BatchPredictRequest(BaseModel):
    sequences: List[List[List[float]]] = Field(..., min_length=1, max_length=500)


@app.get("/")
def dashboard():
    return FileResponse(_DASHBOARD)


@app.get("/dashboard")
def dashboard_alias():
    return FileResponse(_DASHBOARD)


def process_single_sequence(raw_seq: List[List[float]]) -> Dict[str, Any]:
    return RT.classify_raw(raw_seq)


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "round": 18,
        "classes": RT.classes,
        "features": cfg.N_FEATURES,
        "window": cfg.WINDOW_SIZE,
        "scaler": os.path.basename(cfg.SCALER_PATH),
        "tier3_enabled": RT.t3 is not None,
        "alerts_db": DB_PATH,
        "temperature": RT.temperature,
        "thresholds": {
            "gate_benign": cfg.GATE_THRESHOLD,
            "block": cfg.BLOCK_THRESHOLD,
            "flag": cfg.FLAG_THRESHOLD,
        },
    }


@app.post("/predict")
def predict(req: PredictRequest):
    try:
        t0 = time.perf_counter()
        result = process_single_sequence(req.sequence)
        result["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        result["alert_id"] = log_alert(result)
        _update_stats(result)
        return result
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))


@app.post("/predict/batch")
def predict_batch(req: BatchPredictRequest):
    batch_t0 = time.perf_counter()
    results, blk, flg, alw = [], 0, 0, 0
    try:
        for seq in req.sequences:
            item_t0 = time.perf_counter()
            res = process_single_sequence(seq)
            res["latency_ms"] = round((time.perf_counter() - item_t0) * 1000, 2)
            res["alert_id"] = log_alert(res)
            results.append({
                "alert_id": res["alert_id"],
                "label": res["label"],
                "confidence": res["confidence"],
                "action": res["action"],
                "tiers_used": res["tiers_used"],
                "latency_ms": res["latency_ms"],
            })
            blk += res["action"] == "BLOCK"
            flg += res["action"] == "FLAG"
            alw += res["action"] == "ALLOW"
            _update_stats(res)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))

    per_item = [r["latency_ms"] for r in results]
    return {
        "results": results,
        "total": len(req.sequences),
        "blocked": blk,
        "flagged": flg,
        "allowed": alw,
        "latency_ms": round((time.perf_counter() - batch_t0) * 1000, 2),
        "avg_latency_ms": round(sum(per_item) / len(per_item), 2) if per_item else 0.0,
    }


def _update_stats(res: Dict[str, Any]) -> None:
    with _STATS_LOCK:
        stats["total_sequences"] += 1
        if res.get("latency_ms") is not None:
            stats["latency_sum_ms"] += float(res["latency_ms"])
        tiers = res.get("tiers_used", [])
        if tiers == ["tier1_gate"]:
            stats["tier1_fast_allowed"] += 1
        if "tier1_gate_escalate" in tiers:
            stats["tier2_escalated"] += 1
        if "tier3_oneclass" in tiers:
            stats["tier3_checked"] += 1
        act = res["action"]
        if act == "BLOCK":
            stats["total_blocked"] += 1
            if res["label"] in stats["attacks_detected"]:
                stats["attacks_detected"][res["label"]] += 1
        elif act == "FLAG":
            stats["total_flagged"] += 1
        else:
            stats["total_allowed"] += 1


@app.get("/stats")
def get_stats():
    with _STATS_LOCK:
        snap = dict(stats)
        latency_sum = stats["latency_sum_ms"]
        n = stats["total_sequences"]
    persisted = alert_counts()
    return {
        **snap,
        "avg_latency_ms": round(latency_sum / n, 2) if n else 0.0,
        "block_rate_percent": round(snap["total_blocked"] / n * 100, 2) if n else 0.0,
        "persisted_alerts": persisted,
        "persisted_total": sum(persisted.values()),
        "alerts_db": DB_PATH,
        "uptime_seconds": int(time.time() - APP_START_TIME),
    }


@app.post("/demo/traffic")
def demo_traffic(n: int = Query(15, ge=1, le=40)):
    seq_path = os.path.join(cfg.SEQ_DIR, "X_test.npy")
    if not os.path.isfile(seq_path):
        raise HTTPException(status_code=404, detail="v6 test sequences not found")
    Xte = np.load(seq_path)
    sc = RT.scaler
    idx = np.random.choice(len(Xte), size=min(n, len(Xte)), replace=False)
    results = []
    for i in idx:
        raw = sc.inverse_transform(Xte[i].reshape(-1, cfg.N_FEATURES))
        raw = raw.reshape(cfg.WINDOW_SIZE, cfg.N_FEATURES).tolist()
        t0 = time.perf_counter()
        res = process_single_sequence(raw)
        res["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        res["alert_id"] = log_alert(res)
        _update_stats(res)
        results.append({
            "alert_id": res["alert_id"],
            "label": res["label"],
            "action": res["action"],
            "confidence": res["confidence"],
            "tiers_used": res["tiers_used"],
            "tier_trace": res.get("tier_trace"),
            "latency_ms": res["latency_ms"],
        })
    return {"fired": len(results), "results": results}


@app.get("/metrics/tiers")
def get_tier_metrics(refresh: bool = Query(False, description="Recompute from v6 test set")):
    try:
        from api.tier_metrics import compute_tier_metrics
        return compute_tier_metrics(force=refresh)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/alerts")
def get_alerts(
    limit: int = Query(50, ge=1, le=500),
    action: Optional[str] = Query(None, description="Filter: ALLOW, FLAG, or BLOCK"),
):
    if action and action.upper() not in ("ALLOW", "FLAG", "BLOCK"):
        raise HTTPException(status_code=400, detail="action must be ALLOW, FLAG, or BLOCK")
    return {
        "total_returned": min(limit, 500),
        "alerts": fetch_alerts(limit=limit, action=action),
    }


@app.on_event("startup")
def _seed_demo_alerts_if_empty() -> None:
    if sum(alert_counts().values()) > 0:
        return
    seq_path = os.path.join(cfg.SEQ_DIR, "X_test.npy")
    if not os.path.isfile(seq_path):
        return
    Xte = np.load(seq_path)
    sc = RT.scaler
    n = min(30, len(Xte))
    for i in np.linspace(0, len(Xte) - 1, n, dtype=int):
        raw = sc.inverse_transform(Xte[i].reshape(-1, cfg.N_FEATURES))
        raw = raw.reshape(cfg.WINDOW_SIZE, cfg.N_FEATURES).tolist()
        res = process_single_sequence(raw)
        log_alert(res)
        _update_stats(res)
