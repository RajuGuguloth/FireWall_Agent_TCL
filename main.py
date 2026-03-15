import os
import numpy as np
from decision.router import route_decision

# -------- Mock Tier-1 (placeholder) --------
# You already experimented with RF earlier.
# Here we simulate its output cleanly.

def mock_tier1(packet_features):
    """
    Returns (prediction, confidence)
    0 = benign, 1 = attack
    """
    # Simple heuristic for demo:
    # If too many packets, be uncertain
    if packet_features.shape[0] > 500:
        return 0, 0.60   # uncertain benign
    else:
        return 0, 0.92   # confident benign

# -------- Load one real sample --------

DATA_PATH = "data/processed/benign"
files = [f for f in os.listdir(DATA_PATH) if f.endswith(".npy")]
files.sort()

sample_path = os.path.join(DATA_PATH, files[0])
packet_features = np.load(sample_path)

print(f"Loaded sample: {sample_path}")
print(f"Packet feature shape: {packet_features.shape}")

# -------- Tier-1 decision --------

tier1_pred, tier1_conf = mock_tier1(packet_features)

print("\nTier-1 Output:")
print("Prediction:", "BENIGN" if tier1_pred == 0 else "ATTACK")
print("Confidence:", tier1_conf)

# -------- Final routing --------

final_pred, final_conf, source = route_decision(
    tier1_pred,
    tier1_conf,
    packet_features
)

print("\nFinal Decision:")
print("Prediction:", "BENIGN" if final_pred == 0 else "ATTACK")
print("Confidence:", round(final_conf, 4))
print("Decision Source:", source)

