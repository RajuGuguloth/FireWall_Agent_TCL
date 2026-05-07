"""
Step 8: Generate Train/Test Sequences using GroupShuffleSplit
Step 9: Verify the dataset
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder

V5_FINAL = "combined_dataset_v5_final.csv"
OUT_DIR  = "data/splits/v5_sequences"
ENCODER_PATH = "models/serialized/v5_encoder.pkl"
SCALER_PATH  = "models/serialized/v5_scaler.pkl"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs("models/serialized", exist_ok=True)

WINDOW_SIZE = 20

FEATURES_17 = [
    "packet_length", "has_tcp", "has_udp", "has_icmp",
    "payload_length", "payload_entropy",
    "is_ack", "is_rst", "is_fin", "is_psh",
    "is_high_port_src", "ip_ttl", "ip_proto",
    "dst_port", "tcp_flags",
    "flow_total_bytes", "flow_mean_pkt_len"
]

print("=" * 60)
print("  STEP 8 — Sequence Generation + GroupShuffleSplit")
print("=" * 60)

print("  Loading v5_final.csv ...", end="", flush=True)
df = pd.read_csv(V5_FINAL)
print(f" {len(df):,} rows")

# Filter to 5 target attack classes (Tier-2 only classifies attacks)
TARGET_CLASSES = ["BRUTE_FORCE", "DDOS_HTTP_FLOOD", "SLOW_HTTP", "PORT_SCAN", "DNS_TUNNELING"]
df = df[df["label"].isin(TARGET_CLASSES)].copy()

# 1. Encode Labels
le = LabelEncoder()
df["label_idx"] = le.fit_transform(df["label"].astype(str))
joblib.dump(le, ENCODER_PATH)
label_map = dict(zip(le.classes_, le.transform(le.classes_)))
print(f"  Labels encoded: {label_map}")

# 2. Scale Features
scaler = StandardScaler()
df[FEATURES_17] = scaler.fit_transform(df[FEATURES_17])
df[FEATURES_17] = df[FEATURES_17].clip(-5, 5)
joblib.dump(scaler, SCALER_PATH)

# Since explicit src/dst IPs aren't in FEATURES_17, we group by contiguous blocks 
# to represent pseudo-flows, ensuring 0% temporal overlap (leakage) between Train/Test.
# Every 1000 consecutive packets of the same label = 1 flow block.
df["flow_group"] = (df.index // 1000).astype(str) + "_" + df["label"].astype(str)

all_X = []
all_y = []
all_groups = []

# Generate Sliding Windows
print(f"  Generating sequences (window={WINDOW_SIZE}) ...")
grouped = df.groupby("flow_group")
for flow_id, group in grouped:
    if len(group) < WINDOW_SIZE: continue
    
    Y_vals = group["label_idx"].values
    X_vals = group[FEATURES_17].values
    
    # We step by WINDOW_SIZE//2 for some overlap, but overlap is only within the SAME flow_group
    for i in range(0, len(X_vals) - WINDOW_SIZE + 1, WINDOW_SIZE//2):
        window = X_vals[i : i + WINDOW_SIZE]
        label = Y_vals[i + WINDOW_SIZE - 1]
        
        all_X.append(window)
        all_y.append(label)
        all_groups.append(flow_id)

X = np.array(all_X, dtype=np.float32)
y = np.array(all_y)
groups = np.array(all_groups)

print(f"  Total sequences generated: {len(X):,}")

# GroupShuffleSplit
print("  Splitting Train/Test (80/20) by flow group ...")
gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups))

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]
groups_train, groups_test = groups[train_idx], groups[test_idx]

# Check leakage
train_groups_set = set(groups_train)
test_groups_set = set(groups_test)
leakage = train_groups_set.intersection(test_groups_set)

np.save(os.path.join(OUT_DIR, "X_train.npy"), X_train)
np.save(os.path.join(OUT_DIR, "X_test.npy"), X_test)
np.save(os.path.join(OUT_DIR, "y_train.npy"), y_train)
np.save(os.path.join(OUT_DIR, "y_test.npy"), y_test)

print(f"\n  [Train] Total: {len(y_train):,}")
train_counts = np.bincount(y_train)
for cls, idx in label_map.items():
    cnt = train_counts[idx] if idx < len(train_counts) else 0
    print(f"    {cls:<20}: {cnt:,}")

print(f"\n  [Test] Total: {len(y_test):,}")
test_counts = np.bincount(y_test)
for cls, idx in label_map.items():
    cnt = test_counts[idx] if idx < len(test_counts) else 0
    print(f"    {cls:<20}: {cnt:,}")

print(f"\n  Leakage (overlapping groups): {len(leakage)}")
print(f"\n  Saved to → {OUT_DIR}")
print("=" * 60)

print("\n" + "=" * 60)
print("  STEP 9 — Verification Report")
print("=" * 60)
print(f"  Test set has {len(y_test):,} samples minimum 2000? -> {'YES' if len(y_test) >= 2000 else 'NO'}")
print(f"  Features present count: {X_train.shape[2]} == 17? -> {'YES' if X_train.shape[2] == 17 else 'NO'}")
print(f"  Any leaky overlapping groups? -> {'NO' if len(leakage)==0 else 'YES'}")

# Ensure min 400 per class in test
min_test_cls = test_counts.min() if len(test_counts) > 0 else 0
print(f"  Min samples per class in Test set: {min_test_cls:,} >= 400? -> {'YES' if min_test_cls >= 400 else 'NO'}")
print("=" * 60)
