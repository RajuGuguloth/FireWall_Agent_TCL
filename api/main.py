from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import sys
from pathlib import Path

# Add src to path to import HybridFirewall
sys.path.append(str(Path(__file__).parent.parent / "src"))
from inference.hybrid_pipeline import HybridFirewall

app = FastAPI(title="ML Firewall API", description="Inference API for Hybrid ML Firewall")

# Initialize firewall
# Note: In production, you might want to load this on startup
fw = HybridFirewall()

class PacketData(BaseModel):
    # Tier-1 needs 11 features:
    # ['packet_length', 'has_ip', 'has_tcp', 'has_udp', 'has_icmp', 'ip_version', 'ip_ttl', 'ip_proto', 'src_port', 'dst_port', 'tcp_flags']
    features: List[float]
    
class SequenceData(BaseModel):
    # Tier-2 needs a sequence of 10 packets, each with 25 enhanced features
    sequence: List[List[float]]

class InferenceRequest(BaseModel):
    rf_features: List[float]  # 11 features for T1
    seq_features: List[List[float]] # 10x25 features for T2

@app.get("/")
async def root():
    return {"status": "online", "model": "Hybrid RF + CNN-GRU V3"}

@app.post("/predict")
async def predict(request: InferenceRequest):
    try:
        rf_feats = np.array(request.rf_features)
        seq_feats = np.array(request.seq_features)
        
        if rf_feats.shape[0] != 11:
            raise HTTPException(status_code=400, detail="Tier-1 features must be length 11")
        if seq_feats.shape != (10, 25):
            raise HTTPException(status_code=400, detail="Sequence features must be 10x25")
            
        result = fw.classify(rf_feats, seq_feats, use_tier2=True)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
