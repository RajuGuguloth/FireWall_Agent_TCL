"""
Tier-2 Data Preparation (Round 8) - Flow Features Only
- Subset: BRUTE_FORCE, DDOS_HTTP_FLOOD, SLOW_HTTP
- Features: 7 (flow-level statistics)
- Output: 2D NumPy arrays (No sequences)
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import joblib

# ─── Settings ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE = os.path.join(BASE_DIR, "combined_dataset_v4_flow.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "splits", "v4_flow_only_subset")

TARGET_CLASSES = ["BRUTE_FORCE", "DDOS_HTTP_FLOOD", "SLOW_HTTP"]
FLOW_FEATURES = [
    "flow_packet_count",
    "flow_total_bytes",
    "flow_mean_pkt_len",
    "flow_std_pkt_len",
    "flow_mean_entropy",
    "flow_syn_ratio",
    "flow_ack_ratio"
]

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "models", "serialized"), exist_ok=True)

def main():
    print(f"[1/4] Loading dataset: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)
    
    print(f"[2/4] Filtering for target classes: {TARGET_CLASSES}")
    df = df[df["attack_type"].isin(TARGET_CLASSES)].copy()
    
    # Encode labels specifically for this subset
    le = LabelEncoder()
    df["label_idx"] = le.fit_transform(df["attack_type"])
    ENCODER_PATH = os.path.join(BASE_DIR, "models", "serialized", "hard_subset_encoder_v8.pkl")
    joblib.dump(le, ENCODER_PATH)
    print(f"      Labels encoded: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    
    X = df[FLOW_FEATURES].values
    y = df["label_idx"].values
    
    print(f"[3/4] Splitting and Scaling (7 features)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    SCALER_PATH = os.path.join(BASE_DIR, "models", "serialized", "hard_subset_scaler_v8.pkl")
    joblib.dump(scaler, SCALER_PATH)
    print(f"      Scaler saved to: {SCALER_PATH}")
    
    print(f"[4/4] Saving processed data to {OUTPUT_DIR}...")
    np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train.astype(np.float32))
    np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), y_train.astype(np.int64))
    np.save(os.path.join(OUTPUT_DIR, "X_test.npy"), X_test.astype(np.float32))
    np.save(os.path.join(OUTPUT_DIR, "y_test.npy"), y_test.astype(np.int64))
    
    print(f"      Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print("Done.")

if __name__ == "__main__":
    main()
