"""
Prepare sequences for Tier-2 CNN-GRU v4.
- Filters for hard classes: BRUTE_FORCE, DDOS_HTTP_FLOOD, SLOW_HTTP
- Groups by flow: (src_port, dst_port, ip_proto)
- Sliding window size: 20
- Saves to: data/splits/v4_sequences_hard_subset/
"""

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import joblib

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "combined_dataset_v4_flow.csv")
OUT_DIR = os.path.join(BASE_DIR, "data", "splits", "v4_sequences_hard_subset")
ENCODER_PATH = os.path.join(BASE_DIR, "models", "serialized", "hard_subset_encoder.pkl")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(ENCODER_PATH), exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
TARGET_CLASSES = ["BRUTE_FORCE", "DDOS_HTTP_FLOOD", "SLOW_HTTP"]
WINDOW_SIZE = 20
DROP_COLS = ["attack_type", "label"]

def main():
    print(f"[1/4] Loading dataset: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    
    # Filter for target classes
    print(f"[2/4] Filtering for target classes: {TARGET_CLASSES}")
    df = df[df["attack_type"].isin(TARGET_CLASSES)].copy()
    print(f"      Rows remaining: {len(df):,}")

    # Encode labels for this subset
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df["label_idx"] = le.fit_transform(df["attack_type"])
    joblib.dump(le, ENCODER_PATH)
    print(f"      Labels encoded: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    # ── Flow Grouping Logic ────────────────────────────────────────────────────
    # User specified: src_port + dst_port + ip_proto
    print("[3/4] Grouping by flow context ...")
    group_cols = ["src_port", "dst_port", "ip_proto"]
    
    # Extract features
    features = [c for c in df.columns if c not in DROP_COLS + ["label_idx"]]
    print(f"      Features for scaling ({len(features)}): {features}")

    # ── Scaling ──────────────────────────────────────────────────────────────
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    
    # Clip to handle extreme outliers that blow up RNN/GRU gradients
    print("      Clipping features to range [-5, 5]...")
    df[features] = df[features].clip(-5, 5)
    
    SCALER_PATH = os.path.join(BASE_DIR, "models", "serialized", "hard_subset_scaler.pkl")
    joblib.dump(scaler, SCALER_PATH)
    print(f"      Scaler saved to: {SCALER_PATH}")

    all_X = []
    all_y = []
    
    # Group and generate windows
    grouped = df.groupby(group_cols)
    print(f"      Found {len(grouped):,} flows. Generating windows (size={WINDOW_SIZE}) ...")
    
    for _, group in tqdm(grouped):
        if len(group) < WINDOW_SIZE:
            continue
            
        # Sort by index (presuming it preserves temporal order)
        group = group.sort_index()
        
        # Get feature values and label
        # (Assuming the whole flow has the same label for simplicity, 
        # or taking the last label if it varies, but usually it's per-packet)
        X_vals = group[features].values
        Y_vals = group["label_idx"].values
        
        # Sliding window sequence generation
        # We only generate sequences where all packets in the window have the same label 
        # to avoid ambiguous sequences, or we take the most common label.
        # Given these are attack labels, they usually stay consistent in a flow.
        
        for i in range(len(X_vals) - WINDOW_SIZE + 1):
            window = X_vals[i : i + WINDOW_SIZE]
            # Take label of the last packet in window
            label = Y_vals[i + WINDOW_SIZE - 1]
            
            all_X.append(window)
            all_y.append(label)

    if not all_X:
        print("!!! No sequences generated. Check window size or grouping.")
        return

    X = np.array(all_X, dtype=np.float32)
    y = np.array(all_y)
    
    print(f"\n[4/4] Saving sequences ({len(X):,} windows)...")
    # Split into train/test
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    np.save(os.path.join(OUT_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(OUT_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(OUT_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(OUT_DIR, "y_test.npy"), y_test)
    
    print(f"      Saved to: {OUT_DIR}")
    print(f"      Train distribution: {np.bincount(y_train)}")
    print("Done.")

if __name__ == "__main__":
    main()
