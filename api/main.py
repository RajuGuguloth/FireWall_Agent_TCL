import os
import time
import json
import joblib
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

# ─── MODELS & CONSTANTS ──────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RF_MODEL_PATH = os.path.join(BASE_DIR, "models", "tier1_rf_v4.pkl")
ONNX_MODEL_PATH = os.path.join(BASE_DIR, "models", "onnx", "tier2_cnn_gru_r17.onnx")
TEMP_PATH = os.path.join(BASE_DIR, "models", "tier2_r17_temperature.json")

# Load Temperature
try:
    with open(TEMP_PATH, "r") as f:
        config = json.load(f)
        TEMPERATURE = config.get("temperature", 1.0)
except Exception:
    TEMPERATURE = 1.0

# Load ONNX Session
try:
    ort_session = ort.InferenceSession(ONNX_MODEL_PATH)
    onnx_input_name = ort_session.get_inputs()[0].name
except Exception as e:
    print(f"Failed to load ONNX: {e}")
    ort_session = None

# Load Tier 1 RF
try:
    rf_model = joblib.load(RF_MODEL_PATH)
except Exception as e:
    print(f"Failed to load RF: {e}")
    rf_model = None

CLASSES = ["BRUTE_FORCE", "DDOS_HTTP_FLOOD", "SLOW_HTTP", "PORT_SCAN", "DNS_TUNNELING"]
APP_START_TIME = time.time()

# ─── IN-MEMORY STATS ─────────────────────────────────────────────────────────
stats = {
    "total_packets_inspected": 0,  # We'll count sequences
    "total_blocked": 0,
    "total_flagged": 0,
    "total_allowed": 0,
    "attacks_detected": {c: 0 for c in CLASSES}
}

app = FastAPI(title="NDN AI Firewall API")

# ─── REQUEST SCHEMAS ─────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    sequence: List[List[float]]

class BatchPredictRequest(BaseModel):
    sequences: List[List[List[float]]]

# ─── HELPER FUNCTIONS ────────────────────────────────────────────────────────
def process_single_sequence(seq_list: List[List[float]]):
    """Returns single prediction dictionary."""
    if len(seq_list) != 20:
        raise ValueError(f"Sequence length must be exactly 20, got {len(seq_list)}")
    if any(len(pkt) != 17 for pkt in seq_list):
        raise ValueError("Each packet in sequence must have exactly 17 features")

    # 1. Tier-1 Fast Path Check
    packet_1 = np.array(seq_list[0]).reshape(1, -1)
    
    # We attempt RF inference. If RF model is invalid or missing, we skip to Tier 2.
    if rf_model is not None and hasattr(rf_model, "predict_proba"):
        try:
            # Assuming RF output is multiclass with "BENIGN" or binary 0=BENIGN
            rf_probs = rf_model.predict_proba(packet_1)[0]
            # Assumes index 0 or 'BENIGN' class mapping. Let's do a generic check:
            # If Benign is class 0 in the RF model
            benign_idx = 0 
            if hasattr(rf_model, "classes_"):
                if "BENIGN" in rf_model.classes_:
                    benign_idx = list(rf_model.classes_).index("BENIGN")
                elif "Normal" in rf_model.classes_:
                    benign_idx = list(rf_model.classes_).index("Normal")
                    
            if rf_probs[benign_idx] > 0.95:
                # Fast Path exit
                return {
                    "label": "BENIGN",
                    "class_id": -1,
                    "confidence": float(rf_probs[benign_idx]),
                    "tier_used": "tier1_rf",
                    "probabilities": {"BENIGN": float(rf_probs[benign_idx])},
                    "action": "ALLOW"
                }
        except Exception as e:
            pass # ignore RF errors and fallback to T2

    if ort_session is None:
        raise RuntimeError("ONNX session not initialized")

    # 2. Tier-2 Deep Inspection
    seq_np = np.array(seq_list, dtype=np.float32).reshape(1, 20, 17)
    logits = ort_session.run(None, {onnx_input_name: seq_np})[0][0] # shape (5,)
    
    # Apply Temperature Scaling and Softmax
    scaled_logits = logits / TEMPERATURE
    exp_logits = np.exp(scaled_logits - np.max(scaled_logits)) # numeric stability
    probs = exp_logits / exp_logits.sum()
    
    class_id = int(np.argmax(probs))
    confidence = float(probs[class_id])
    label = CLASSES[class_id]

    action = "ALLOW"
    if confidence > 0.95:
        action = "BLOCK"
    elif confidence >= 0.80:
        action = "FLAG"

    return {
        "label": label,
        "class_id": class_id,
        "confidence": confidence,
        "tier_used": "tier2_cnn_gru",
        "probabilities": {CLASSES[i]: float(probs[i]) for i in range(5)},
        "action": action
    }

# ─── API ENDPOINTS ───────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model": "tier2_cnn_gru_r17",
        "classes": 5,
        "architecture": "Conv1D(17→64) → GRU(64→128,2L) → Linear(128→5)",
        "test_f1": 0.9840,
        "round": 17
    }


@app.post("/predict")
def predict(req: PredictRequest):
    try:
        start_time = time.perf_counter()
        
        result = process_single_sequence(req.sequence)
        
        # Update Stats
        stats["total_packets_inspected"] += 1
        if result["action"] == "BLOCK":
            stats["total_blocked"] += 1
            if result["label"] in stats["attacks_detected"]:
                stats["attacks_detected"][result["label"]] += 1
        elif result["action"] == "FLAG":
            stats["total_flagged"] += 1
        else:
            stats["total_allowed"] += 1
            
        return result
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
def predict_batch(req: BatchPredictRequest):
    start_time = time.perf_counter()
    results = []
    
    blk, flg, alw = 0, 0, 0
    
    for seq in req.sequences:
        try:
            res = process_single_sequence(seq)
            results.append({
                "label": res["label"],
                "confidence": res["confidence"],
                "action": res["action"]
            })
            
            # Update batch local counts
            if res["action"] == "BLOCK": blk += 1
            elif res["action"] == "FLAG": flg += 1
            else: alw += 1
            
            # Update global stats
            stats["total_packets_inspected"] += 1
            if res["action"] == "BLOCK":
                stats["total_blocked"] += 1
                if res["label"] in stats["attacks_detected"]:
                    stats["attacks_detected"][res["label"]] += 1
            elif res["action"] == "FLAG":
                stats["total_flagged"] += 1
            else:
                stats["total_allowed"] += 1
                
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve))
            
    latency_ms = (time.perf_counter() - start_time) * 1000
    return {
        "results": results,
        "total": len(req.sequences),
        "blocked": blk,
        "flagged": flg,
        "allowed": alw,
        "latency_ms": round(latency_ms, 2)
    }


@app.get("/stats")
def get_stats():
    total = stats["total_packets_inspected"]
    blk = stats["total_blocked"]
    rate = (blk / total * 100) if total > 0 else 0.0
    
    return {
        **stats,
        "block_rate_percent": round(rate, 2),
        "avg_latency_ms": 0.0, # Handled dynamically by load balancer or metrics, here we return 0.0 or we can track it
        "uptime_seconds": int(time.time() - APP_START_TIME)
    }
