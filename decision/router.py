import torch
import numpy as np
from tier2.train_tier2 import Tier2GRU
from utils.sliding_window import generate_sliding_windows

TIER1_THRESHOLD = 0.85
TIER2_THRESHOLD = 0.70
WINDOW_SIZE = 64
STRIDE = 32

DEVICE = torch.device("cpu")

# -------- Tier-2 loader --------

def load_tier2_model():
    model = Tier2GRU()
    model.load_state_dict(
        torch.load("models/checkpoints/tier2_gru.pt", map_location=DEVICE)
    )
    model.eval()
    return model

tier2_model = load_tier2_model()

# -------- Router --------

def route_decision(
    tier1_pred,
    tier1_conf,
    packet_features: np.ndarray
):
    """
    tier1_pred: int (0=benign, 1=attack)
    tier1_conf: float [0,1]
    packet_features: (N, F)
    """

    if tier1_conf >= TIER1_THRESHOLD:
        return tier1_pred, tier1_conf, "Tier1"

    windows = generate_sliding_windows(
        packet_features,
        WINDOW_SIZE,
        STRIDE
    )

    if len(windows) == 0:
        return tier1_pred, tier1_conf, "Tier1-Fallback"

    with torch.no_grad():
        x = torch.tensor(windows, dtype=torch.float32)
        logits = tier2_model(x)
        probs = torch.softmax(logits, dim=1)
        confs, preds = torch.max(probs, dim=1)

    best_idx = torch.argmax(confs)
    best_conf = confs[best_idx].item()
    best_pred = preds[best_idx].item()

    if best_conf >= TIER2_THRESHOLD:
        return best_pred, best_conf, "Tier2"

    return tier1_pred, max(tier1_conf, best_conf), "Uncertain"

