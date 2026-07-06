"""
Steps 5, 6, 7: Detox → Merge → SMOTE-balance → Save final v5

Step 5: Detox combined_dataset_v5_expanded.csv
Step 6: Merge with combined_dataset_v4_flow.csv
Step 7: Balance classes with SMOTE if needed
"""

import pandas as pd
import numpy as np
from sklearn.utils import resample
from datetime import datetime
import os

V5_EXPANDED = "combined_dataset_v5_expanded.csv"
V4_CLEAN    = "combined_dataset_v4_flow.csv"
V5_FINAL    = "combined_dataset_v5_final.csv"

FEATURES_17 = [
    "packet_length",
    "has_tcp", "has_udp", "has_icmp",
    "payload_length", "payload_entropy",
    "is_ack", "is_rst", "is_fin", "is_psh",
    "is_high_port_src",
    "ip_ttl", "ip_proto",
    "dst_port", "tcp_flags",
    "flow_total_bytes", "flow_mean_pkt_len",
]
ALL_COLS = FEATURES_17 + ["label"]

def divider(title=""):
    print(f"\n{'─'*60}")
    if title:
        print(f"  {title}")
        print(f"{'─'*60}")

# ─────────────────────────────────────────────────────────────
# STEP 5 — DETOX the v5 expanded data
# ─────────────────────────────────────────────────────────────
divider("STEP 5 — Data Detox on v5_expanded.csv")

print("  Loading v5_expanded.csv ...", end="", flush=True)
df5 = pd.read_csv(V5_EXPANDED)
print(f" {len(df5):,} rows loaded")

rows_before = len(df5)
print(f"  Rows before detox:  {rows_before:,}")
print(f"  Class counts before:")
for c, n in df5["label"].value_counts().items():
    print(f"    {c:<22}: {n:,}")

# Keep only columns we need
keep_cols = [c for c in ALL_COLS if c in df5.columns]
df5 = df5[keep_cols].copy()

# Remove null/missing values
df5 = df5.dropna()
removed_null = rows_before - len(df5)
print(f"\n  Rows removed (null):        {removed_null:,}")

# Remove packets with payload_length = 0 (useless for training)
if "payload_length" in df5.columns:
    before = len(df5)
    df5 = df5[df5["payload_length"] > 0]
    removed_zero_payload = before - len(df5)
    print(f"  Rows removed (payload=0):   {removed_zero_payload:,}")
else:
    removed_zero_payload = 0

# Deduplication per class (keep attack duplicates, reduce benign duplicates)
attack_df  = df5[df5["label"] != "BENIGN"].drop_duplicates()
benign_df  = df5[df5["label"] == "BENIGN"]
# Keep max 30% of duplicate benign flows
benign_dedup = benign_df.drop_duplicates()
benign_keep  = benign_df.sample(frac=0.3, random_state=42) if len(benign_df) > 0 else benign_df

df5_detoxed = pd.concat([attack_df, benign_keep], ignore_index=True)
removed_dup = rows_before - removed_null - removed_zero_payload - len(df5_detoxed)

print(f"  Rows removed (dedup):       {removed_dup:,}")
print(f"  Rows after detox:           {len(df5_detoxed):,}")
print(f"\n  Class counts after detox:")
for c, n in df5_detoxed["label"].value_counts().items():
    print(f"    {c:<22}: {n:,}")

# ─────────────────────────────────────────────────────────────
# STEP 6 — MERGE with v4 (original clean dataset)
# ─────────────────────────────────────────────────────────────
divider("STEP 6 — Merge v5_detoxed + v4_flow")

print("  Loading v4_flow.csv ...", end="", flush=True)
v4_cols = FEATURES_17 + ["label"]
# Load only needed columns from v4
v4_cols_load = v4_cols + ["attack_type"]
df4_chunks = pd.read_csv(V4_CLEAN, usecols=lambda c: c in v4_cols_load, chunksize=100000)
df4 = pd.concat(df4_chunks)

# Override the 0/1 label with textual attack_type
if "attack_type" in df4.columns:
    df4["label"] = df4["attack_type"]
    df4 = df4.drop(columns=["attack_type"])
else:
    print("⚠️  v4 has no attack_type column — check CSV structure")

print(f" {len(df4):,} rows loaded from v4")

# Align columns
for col in FEATURES_17:
    if col not in df5_detoxed.columns:
        df5_detoxed[col] = 0
    if col not in df4.columns:
        df4[col] = 0

df5_detoxed = df5_detoxed[ALL_COLS]
df4 = df4[ALL_COLS]

# Concat and remove cross-dataset duplicates
merged = pd.concat([df4, df5_detoxed], ignore_index=True)
before_dedup = len(merged)
merged = merged.drop_duplicates(subset=FEATURES_17)
cross_dupes = before_dedup - len(merged)

print(f"  v4 rows:                    {len(df4):,}")
print(f"  v5 detoxed rows:            {len(df5_detoxed):,}")
print(f"  Cross-dataset duplicates:   {cross_dupes:,}")
print(f"  Total merged rows:          {len(merged):,}")
print(f"\n  Class distribution after merge:")
for c, n in merged["label"].value_counts().items():
    print(f"    {c:<22}: {n:,}")

# ─────────────────────────────────────────────────────────────
# STEP 7 — BALANCE with SMOTE (or simple oversample)
# ─────────────────────────────────────────────────────────────
divider("STEP 7 — Balance Classes")

counts        = merged["label"].value_counts()
max_class_cnt = int(counts.max())
target        = max(1000, max_class_cnt)  # at least 1000 per class

print(f"  Largest class: {max_class_cnt:,}")
print(f"  Target minimum per class: {target:,}")

balanced_dfs = []
for cls in counts.index:
    cls_df = merged[merged["label"] == cls]
    if len(cls_df) < target and len(cls_df) / max_class_cnt < 0.30:
        # Needs upsampling
        oversampled = resample(cls_df, replace=True, n_samples=target, random_state=42)
        balanced_dfs.append(oversampled)
        print(f"    {cls:<22}: {len(cls_df):,} → {target:,} (oversampled)")
    else:
        balanced_dfs.append(cls_df)
        print(f"    {cls:<22}: {len(cls_df):,} (kept as-is)")

final_df = pd.concat(balanced_dfs, ignore_index=True)
final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\n  Final total: {len(final_df):,}")
print(f"\n  Class counts after balancing:")
for c, n in final_df["label"].value_counts().items():
    print(f"    {c:<22}: {n:,}")

# ─────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────
print(f"\n  Saving → {V5_FINAL} ...", end="", flush=True)
final_df.to_csv(V5_FINAL, index=False)
print(f" Done ({len(final_df):,} rows)")
print(f"\n  {'='*56}")
print(f"  STEPS 5-7 COMPLETE")
print(f"  Final dataset: {V5_FINAL}")
print(f"  {'='*56}")
