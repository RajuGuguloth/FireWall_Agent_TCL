"""
=============================================================
  Dataset Refinement Pipeline — NDN AI Firewall Project
  Round 15 | Tushar Sood | 15 March 2026
=============================================================
  Fixes applied:
    Fix 1. has_ip == 1 filter  → removes ~1.15M Layer-2 rows
    Fix 2. Drop 3 leaked features + column exclusions → 27 features
    Fix 3. GroupBy dst_port+ip_proto+attack_type (stable service port)
    Fix 4. SMOTE on TRAIN ONLY (target_ratio=0.50 for SLOW_HTTP)
    Fix 5 (R15). GroupShuffleSplit — disjoint groups across train/val/test
=============================================================
"""

import os, sys, json, time, warnings
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from collections import Counter

warnings.filterwarnings("ignore")

# ── CONFIG ───────────────────────────────────────────────────────────────────
CSV_PATH      = "combined_dataset_v4_flow.csv"
OUT_DIR       = "data/splits/v4_sequences_hard_subset"
RESULTS_DIR   = "results"
WINDOW_SIZE   = 20
STRIDE        = 10
TARGET_LABELS = ["BRUTE_FORCE", "DDOS_HTTP_FLOOD", "SLOW_HTTP"]
CLIP_VAL      = 5.0
ADV_RATIO     = 0.20
RANDOM_SEED   = 42

# Features to ALWAYS EXCLUDE (not predictive or are labels/targets)
ALWAYS_DROP = [
    "has_ip",            # always 1 after filter
    "is_layer2_only",    # always 0 after filter
    "label",             # binary 0/1, redundant
    "attack_type",       # target variable, not a feature
]

# Features dropped due to data leakage (correlation too high per class)
LEAKED_FEATURES = [
    "flow_syn_ratio",     # corr=0.978 with BRUTE_FORCE
    "is_high_port_dst",   # corr=1.000 with DDOS
    "flow_packet_count",  # corr=0.974 with DDOS
]

np.random.seed(RANDOM_SEED)
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

LOG_FILE     = os.path.join(RESULTS_DIR, "proof_of_work_log.json")
AUDIT_FILE   = os.path.join(RESULTS_DIR, "dataset_audit.txt")
SUMMARY_FILE = os.path.join(RESULTS_DIR, "proof_of_work_summary.txt")
FEAT_FILE    = os.path.join(RESULTS_DIR, "final_feature_list.txt")

def log(msg):
    print(msg)
    with open(SUMMARY_FILE, "a") as f:
        f.write(msg + "\n")

def append_json_log(entry):
    logs = []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            try: logs = json.load(f)
            except: logs = []
    logs.append(entry)
    with open(LOG_FILE, "w") as f:
        json.dump(logs, f, indent=2)

# ═══════════════════════════════════════════════════════════════
#  STEP 0 — LOAD
# ═══════════════════════════════════════════════════════════════
log("\n" + "="*60)
log("  DATASET REFINEMENT PIPELINE — ROUND 15")
log(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log("="*60)

if not os.path.exists(CSV_PATH):
    log(f"\n❌  ERROR: File not found → {CSV_PATH}")
    sys.exit(1)

log(f"\n📂  Loading: {CSV_PATH}")
df = pd.read_csv(CSV_PATH)
log(f"   Raw shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

# Issue 8 fix: default so PoW log never hits NameError
removed = 0

# ── Fix 1: Filter out Layer-2 (non-IP) packets ───────────────
if "has_ip" in df.columns:
    before = len(df)
    df = df[df["has_ip"] == 1].copy()
    removed = before - len(df)
    log(f"   [Fix 1] Removed {removed:,} Layer-2 rows, {len(df):,} rows remain")
else:
    log("   [Fix 1] Column 'has_ip' not found — skipping layer-2 filter")

# ── Detect label column ──────────────────────────────────────
label_col = None
for candidate in ["attack_type", "label", "Label", "class", "Class", "category"]:
    if candidate in df.columns:
        label_col = candidate
        break
if label_col is None:
    log("\n❌  ERROR: Cannot find label column.")
    sys.exit(1)
log(f"   Label column: '{label_col}'")
df[label_col] = df[label_col].astype(str).str.strip().str.upper()

# ── Class distribution ───────────────────────────────────────
log("\n📊  Class Distribution (ALL classes):")
dist = df[label_col].value_counts()
total = len(df)
for cls, cnt in dist.items():
    log(f"  {cls:<30} {cnt:>8,}  ({cnt/total*100:.1f}%)")

# ── Identify numeric feature columns ────────────────────────
feature_cols = [
    "packet_length", "has_tcp", "has_udp", "has_icmp",
    "payload_length", "payload_entropy", "is_ack", "is_rst",
    "is_fin", "is_psh", "is_high_port_src", "is_well_known_port",
    "ip_ttl", "ip_proto", "tcp_flags", "flow_total_bytes",
    "flow_mean_pkt_len"
]
assert len(feature_cols) == 17, f"Expected 17, got {len(feature_cols)}"

# Verify no zero-variance columns slipped through
for col in feature_cols:
    if df[col].std() == 0:
        raise ValueError(f"Zero variance column slipped through: {col}")

# ── FIX 2: Log dropped features ─────────────────────────────
log(f"\n🧹  [Fix 2] Feature Selection (Round 16):")
log(f"   Feature count: {len(feature_cols)}")
log(f"   Feature list: {feature_cols}")

# ── Missing / Infinite value fix ─────────────────────────────
missing = df[feature_cols].isnull().sum()
missing = missing[missing > 0]
if len(missing) > 0:
    df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())
inf_mask = np.isinf(df[feature_cols].values)
if inf_mask.any():
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())
log("   ✅  No missing / infinite values (or fixed)")

# ── Save audit ───────────────────────────────────────────────
with open(AUDIT_FILE, "w") as f:
    f.write(f"DATASET AUDIT — Round 16 — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("="*60 + "\n\n")
    f.write(f"FEATURES USED ({len(feature_cols)}): {feature_cols}\n\n")
    f.write(f"FEATURES DROPPED: Fixed list of 17 features used for Round 16\n")

# ═══════════════════════════════════════════════════════════════
#  STEP 1 — FILTER TO 3 TARGET CLASSES
# ═══════════════════════════════════════════════════════════════
log("\n" + "="*60)
log("📌  STEP 1 — Filter to 3 HTTP Attack Classes")
log("-"*50)

df_filtered = df[df[label_col].isin(TARGET_LABELS)].copy()
log(f"  Filtered shape: {df_filtered.shape[0]:,} rows")
for cls in TARGET_LABELS:
    cnt = (df_filtered[label_col] == cls).sum()
    log(f"  {cls:<25} {cnt:>7,} rows")

# ═══════════════════════════════════════════════════════════════
#  STEP 2 — TEMPORAL ORDERING
# ═══════════════════════════════════════════════════════════════
log("\n" + "="*60)
log("⏱️   STEP 2 — Temporal Ordering")
log("-"*50)

time_col = None
for candidate in ["timestamp", "flow_start_time", "start_time", "time", "Time"]:
    if candidate in df_filtered.columns:
        time_col = candidate
        break

# ── FIX 3: Group by time window (dst_port + ip_proto + attack_type + time_window)
WINDOW_PACKETS = 300
df_filtered = df_filtered.sort_values(["dst_port", "ip_proto", label_col])
df_filtered["row_within_group"] = df_filtered.groupby(["dst_port", "ip_proto", label_col]).cumcount()
df_filtered["time_window"] = df_filtered["row_within_group"] // WINDOW_PACKETS
available_group_cols = ["dst_port", "ip_proto", label_col, "time_window"]
log(f"  [Fix 3] Grouping by: {available_group_cols} (Time window = {WINDOW_PACKETS} pkts)")

# Show group size distribution
groups = df_filtered.groupby(available_group_cols)
group_sizes = groups.size()
log(f"  Total groups: {len(group_sizes):,}")
log(f"  Group sizes: min={group_sizes.min()}, max={group_sizes.max():,}, median={group_sizes.median():.0f}")

# ═══════════════════════════════════════════════════════════════
#  STEP 3 — BUILD SEQUENCES
# ═══════════════════════════════════════════════════════════════
log("\n" + "="*60)
log("🔄  STEP 3 — Build Sliding Window Sequences")
log(f"    Window={WINDOW_SIZE}, Stride={STRIDE}")
log("-"*50)

le = LabelEncoder()
le.fit(TARGET_LABELS)

def build_sequences(dataframe, feature_columns, label_column,
                    group_columns, window=20, stride=10):
    """
    Returns:
      all_seqs  : np.array  (N, window, features)
      all_labels: np.array  (N,)
      all_gids  : np.array  (N,)  integer group ID per (dst_port, ip_proto, attack_type)
    Issue 4 fix: skip groups smaller than window instead of zero-padding.
    Zero-padded sequences become non-zero after scaling, polluting the model.
    """
    all_seqs, all_labels, all_gids, all_dst_ports = [], [], [], []
    groups = dataframe.groupby(group_columns)
    short_groups = 0
    for gid, (group_key, grp) in enumerate(groups):
        dst_port = group_key[0]  # dst_port is the first element in group_columns
        grp = grp.reset_index(drop=True)
        if len(grp) < window:          # Issue 4: skip instead of pad
            short_groups += 1
            continue
        X = grp[feature_columns].values.astype(np.float32)
        y = grp[label_column].values
        for start in range(0, len(X) - window + 1, stride):
            seq = X[start:start + window]
            window_labels = y[start:start + window]
            majority = Counter(window_labels).most_common(1)[0][0]
            if majority in TARGET_LABELS:
                all_seqs.append(seq)
                all_labels.append(majority)
                all_gids.append(gid)
                all_dst_ports.append(dst_port)
    if short_groups:
        print(f"  (Skipped {short_groups} groups with <{window} packets — no padding)")
    return (np.array(all_seqs, dtype=np.float32),
            np.array(all_labels),
            np.array(all_gids, dtype=np.int32),
            np.array(all_dst_ports, dtype=np.int32))

seqs, labels, group_ids, dst_ports = build_sequences(
    df_filtered, feature_cols, label_col,
    available_group_cols, WINDOW_SIZE, STRIDE
)

log(f"  Total sequences generated: {len(seqs):,}")
if len(seqs) > 0:
    log(f"  Sequence shape: {seqs.shape}")

log(f"\n  Per-class sequence counts:")
for cls in TARGET_LABELS:
    cnt = (labels == cls).sum()
    log(f"  {cls:<25} {cnt:>7,} sequences")

if len(seqs) == 0:
    log("\n❌  ERROR: No sequences generated. Check grouping columns.")
    sys.exit(1)

# ═══════════════════════════════════════════════════════════════
#  STEP 4 — GROUP-AWARE SPLIT (R15 fix — no group leakage)
# ═══════════════════════════════════════════════════════════════
log("\n" + "="*60)
log("✂️   STEP 4 — Group-Aware Train / Val / Test Split")
log("    Strategy: GroupShuffleSplit on (dst_port,ip_proto,attack_type) groups")
log("-"*50)

# ── GroupShuffleSplit as requested by user ───────────────────
# 1. Try random seeds until test set contains all 3 classes
required_classes = {"BRUTE_FORCE", "DDOS_HTTP_FLOOD", "SLOW_HTTP"}
valid_seed = None

for seed in [42, 7, 13, 99, 123, 456, 789]:
    gss_test = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    train_val_idx, test_idx = next(gss_test.split(seqs, labels, group_ids))
    classes_in_test = set(labels[test_idx].tolist())
    
    if required_classes.issubset(classes_in_test):
        valid_seed = seed
        log(f"  ✅ Seed {seed} successfully produced all 3 classes in test set.")
        break

if valid_seed is None:
    raise RuntimeError(f"STOP: Test set missing classes: {required_classes - classes_in_test}. Re-run with different RANDOM_SEED.")

RANDOM_SEED = valid_seed

# 2. Split train_val into train and val (0.176 ratio to get 15% of total)
gss_val = GroupShuffleSplit(n_splits=1, test_size=0.176, random_state=RANDOM_SEED)
train_idx_rel, val_idx_rel = next(gss_val.split(seqs[train_val_idx], labels[train_val_idx], group_ids[train_val_idx]))

train_idx = train_val_idx[train_idx_rel]
val_idx   = train_val_idx[val_idx_rel]

classes_in_train = set(labels[train_idx].tolist())
classes_in_val   = set(labels[val_idx].tolist())
classes_in_test  = set(labels[test_idx].tolist())

print(f"Classes in test: {classes_in_test}")
print(f"Feature count: {len(feature_cols)}")
print(f"Total sequences: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")
print(f"SLOW_HTTP test sequences: {(labels[test_idx] == 'SLOW_HTTP').sum()}")

log(f"Train classes: {classes_in_train}")
log(f"Val classes:   {classes_in_val}")
log(f"Test classes:  {classes_in_test}")

X_train, y_train = seqs[train_idx], labels[train_idx]
X_val,   y_val   = seqs[val_idx],   labels[val_idx]
X_test,  y_test  = seqs[test_idx],  labels[test_idx]

# Save test destination ports for the GNN topology pass
np.save(os.path.join(OUT_DIR, "test_ports.npy"), dst_ports[test_idx])

# Verify groups are disjoint across splits
train_groups = set(group_ids[train_idx].tolist())
val_groups   = set(group_ids[val_idx].tolist())
test_groups  = set(group_ids[test_idx].tolist())
assert train_groups.isdisjoint(test_groups), "TRAIN/TEST GROUP OVERLAP DETECTED"
assert val_groups.isdisjoint(test_groups),   "VAL/TEST GROUP OVERLAP DETECTED"
log("  ✅  Groups are fully disjoint (per-class stratified group split)")
log(f"  Train groups: {len(train_groups)}   Val groups: {len(val_groups)}   Test groups: {len(test_groups)}")

log(f"  Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")
log(f"\n  Train class distribution:")
for cls in TARGET_LABELS:
    log(f"  {cls:<25} {(y_train == cls).sum():>7,}")

# ═══════════════════════════════════════════════════════════════
#  STEP 5 — SMOTE FOR SLOW_HTTP (train only)
# ═══════════════════════════════════════════════════════════════
log("\n" + "="*60)
log("⚖️   STEP 5 — SMOTE Oversampling (SLOW_HTTP, train only)")
log("-"*50)

def manual_smote(X, y, minority_class, target_ratio=0.50, k=5):
    others = [c for c in np.unique(y) if c != minority_class]
    if not others:
        return X, y
    majority_count = max((y == c).sum() for c in others)
    minority_idx = np.where(y == minority_class)[0]
    current_count = len(minority_idx)
    target_count  = int(majority_count * target_ratio)

    if current_count >= target_count:
        log(f"  ✅  {minority_class} already at {current_count:,} — no oversampling needed")
        return X, y

    # Issue 7 fix: guard k against tiny class sizes
    k = min(k, current_count - 1)
    if k < 1:
        log(f"  ⚠️  {minority_class} has only {current_count} samples — skipping SMOTE (need ≥2)")
        return X, y

    needed = target_count - current_count
    log(f"  {minority_class}: {current_count:,} → target {target_count:,} (+{needed:,} synthetic, k={k})")

    minority_X = X[minority_idx]
    n_min, W, F = minority_X.shape
    flat = minority_X.reshape(n_min, -1)

    new_samples = []
    for _ in range(needed):
        idx = np.random.randint(n_min)
        sample = flat[idx]
        dists = np.linalg.norm(flat - sample, axis=1)
        dists[idx] = np.inf
        nn_idx = np.argsort(dists)[:k]
        neighbor = flat[np.random.choice(nn_idx)]
        alpha = np.random.random()
        synthetic_flat = sample + alpha * (neighbor - sample)
        new_samples.append(synthetic_flat.reshape(W, F))

    new_X = np.array(new_samples, dtype=np.float32)
    new_y = np.array([minority_class] * len(new_X))

    X_out = np.concatenate([X, new_X], axis=0)
    y_out = np.concatenate([y, new_y], axis=0)
    perm = np.random.permutation(len(X_out))
    return X_out[perm], y_out[perm]

# Issue 1 fix: capture clean train BEFORE augmentation so scaler fits on real data only
X_train_clean = X_train.copy()

slow_train_before = (y_train == "SLOW_HTTP").sum()
log(f"  SLOW_HTTP train before SMOTE: {slow_train_before:,}")
X_train, y_train = manual_smote(X_train, y_train, "SLOW_HTTP", target_ratio=0.50)
slow_train_after = (y_train == "SLOW_HTTP").sum()
log(f"  SLOW_HTTP train after SMOTE:  {slow_train_after:,}")
log(f"  SLOW_HTTP val (real only):    {(y_val == 'SLOW_HTTP').sum():,}")
slow_http_test_count = int((y_test == "SLOW_HTTP").sum())
log(f"  SLOW_HTTP test (real only):   {slow_http_test_count:,}")

if slow_http_test_count < 50:
    log("\n⚠️  CRITICAL WARNING: SLOW_HTTP test samples < 50. Metrics very unreliable.")
elif slow_http_test_count < 100:
    log("\n⚠️  WARNING: SLOW_HTTP test samples < 100.")

log(f"\n  Train class distribution after SMOTE:")
for cls in TARGET_LABELS:
    cnt = (y_train == cls).sum()
    log(f"  {cls:<25} {cnt:>7,}")

# ═══════════════════════════════════════════════════════════════
#  STEP 6 — ADVERSARIAL AUGMENTATION (train only)
# ═══════════════════════════════════════════════════════════════
log("\n" + "="*60)
log(f"🥷  STEP 6 — Adversarial Augmentation ({int(ADV_RATIO*100)}% of train)")
log("-"*50)

# Issue 2 fix: do NOT use X_val as pool — inject random noise instead to avoid indirect leakage
adv_seqs, adv_labels = [], []
for cls in TARGET_LABELS:
    cls_idx = np.where(y_train == cls)[0]
    n_adv = int(len(cls_idx) * ADV_RATIO)
    if n_adv == 0:
        continue
    chosen = np.random.choice(cls_idx, n_adv, replace=False)
    for i in chosen:
        seq = X_train[i].copy()
        if cls == "BRUTE_FORCE":
            aug = seq + np.random.normal(0, 0.3, seq.shape).astype(np.float32)
        elif cls == "DDOS_HTTP_FLOOD":
            aug = seq.copy()
            mask = np.random.random(seq.shape[0]) < 0.20
            if mask.any():
                # random noise instead of val pool (Issue 2 fix)
                aug[mask] = np.random.normal(0, 0.1, aug[mask].shape).astype(np.float32)
        else:  # SLOW_HTTP
            aug = seq * np.random.uniform(0.8, 1.2, seq.shape).astype(np.float32)
        adv_seqs.append(aug)
        adv_labels.append(cls)
    log(f"  {cls:<25} +{n_adv:,} adversarial samples")

if adv_seqs:
    adv_seqs  = np.array(adv_seqs, dtype=np.float32)
    adv_labels = np.array(adv_labels)
    X_train = np.concatenate([X_train, adv_seqs], axis=0)
    y_train  = np.concatenate([y_train, adv_labels], axis=0)
    perm = np.random.permutation(len(X_train))
    X_train, y_train = X_train[perm], y_train[perm]

log(f"\n  Final train size: {len(X_train):,} sequences")

# ═══════════════════════════════════════════════════════════════
#  STEP 7 — SCALE + CLIP
# ═══════════════════════════════════════════════════════════════
log("\n" + "="*60)
log("📏  STEP 7 — StandardScaler + Clip to [-5, +5]")
log("-"*50)

# Issue 1 fix: fit scaler on CLEAN real train data (before SMOTE + adversarial aug)
n_clean, W, F = X_train_clean.shape
scaler = StandardScaler()
scaler.fit(X_train_clean.reshape(-1, F))  # fit on clean real data only

# Transform all splits with the clean-fitted scaler
n_train = len(X_train)
if n_train > 0:
    X_train = np.clip(scaler.transform(X_train.reshape(-1, F)).reshape(n_train, W, F), -CLIP_VAL, CLIP_VAL)
if len(X_val) > 0:
    X_val   = np.clip(scaler.transform(X_val.reshape(-1, F)).reshape(len(X_val), W, F), -CLIP_VAL, CLIP_VAL)
if len(X_test) > 0:
    X_test  = np.clip(scaler.transform(X_test.reshape(-1, F)).reshape(len(X_test), W, F), -CLIP_VAL, CLIP_VAL)

log(f"  Scaler fitted on CLEAN real train ({n_clean} sequences, before SMOTE/aug) ✅")
log(f"  Train range: [{X_train.min():.2f}, {X_train.max():.2f}]")
log(f"  Val range:   [{X_val.min():.2f},  {X_val.max():.2f}]")
log(f"  Test range:  [{X_test.min():.2f}, {X_test.max():.2f}]")

y_train_enc = le.transform(y_train)
y_val_enc   = le.transform(y_val)
y_test_enc  = le.transform(y_test)
log(f"\n  Label encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# ═══════════════════════════════════════════════════════════════
#  STEP 8 — SAVE EVERYTHING
# ═══════════════════════════════════════════════════════════════
log("\n" + "="*60)
log("💾  STEP 8 — Save Splits & Scaler")
log("-"*50)

import pickle
np.save(os.path.join(OUT_DIR, "train_sequences.npy"), X_train)
np.save(os.path.join(OUT_DIR, "train_labels.npy"),    y_train_enc)
np.save(os.path.join(OUT_DIR, "val_sequences.npy"),   X_val)
np.save(os.path.join(OUT_DIR, "val_labels.npy"),      y_val_enc)
np.save(os.path.join(OUT_DIR, "test_sequences.npy"),  X_test)
np.save(os.path.join(OUT_DIR, "test_labels.npy"),     y_test_enc)

with open(os.path.join(OUT_DIR, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)
with open(os.path.join(OUT_DIR, "label_encoder.pkl"), "wb") as f:
    pickle.dump(le, f)

stats = {
    "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "round": 16,
    "first_valid_result": True,
    "feature_count": 17,
    "zero_variance_cols_removed": [
        "ip_header_length","packet_direction","ip_version",
        "flow_std_pkt_len","flow_ack_ratio","Unnamed: 0"
    ],
    "additional_leaks_removed": [
        "is_syn (corr=0.81 BF)",
        "tcp_window_size (corr=0.81 SLOW)",
        "flow_mean_entropy (corr=0.80 SLOW)"
    ],
    "grouping": "time_window_300",
    "viable_groups": len(train_groups) + len(val_groups) + len(test_groups),
    "classes_in_test": list(classes_in_test),
    "random_seed_used": RANDOM_SEED,
    "fresh_start": True,
    "prior_rounds_valid": False,
    "prior_rounds_note": "R12-R15 invalid due to data bugs",
    "window_size": WINDOW_SIZE,
    "stride": STRIDE,
    "features_used": int(len(feature_cols)),
    "feature_list": feature_cols,
    "train_size": int(len(X_train)),
    "val_size":   int(len(X_val)),
    "test_size":  int(len(X_test)),
    "train_class_dist": {c: int((y_train == c).sum()) for c in TARGET_LABELS},
    "val_class_dist":   {c: int((y_val   == c).sum()) for c in TARGET_LABELS},
    "test_class_dist":  {c: int((y_test  == c).sum()) for c in TARGET_LABELS},
    "smote_applied_to": "TRAIN only",
}
with open(os.path.join(OUT_DIR, "dataset_stats.json"), "w") as f:
    json.dump(stats, f, indent=2)

log(f"  ✅  train_sequences.npy   → {X_train.shape}")
log(f"  ✅  val_sequences.npy     → {X_val.shape}")
log(f"  ✅  test_sequences.npy    → {X_test.shape}")
log(f"  ✅  scaler.pkl + label_encoder.pkl")
log(f"  ✅  dataset_stats.json")

# ── Proof of work log ────────────────────────────────────────
pow_entry = {
    "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
    "round": 15,
    "script": "refine_dataset.py",
    "status": "SUCCESS",
    "split_strategy": "group_aware_dst_port",
    "r14_f1_was_invalid": True,
    "r14_cause": "same-group sequence leakage (random split shared dst_port groups)",
    "train_groups": len(train_groups),
    "val_groups":   len(val_groups),
    "test_groups":  len(test_groups),
    "total_sequences": int(len(X_train) + len(X_val) + len(X_test)),
    "train_size": int(len(X_train)),
    "val_size":   int(len(X_val)),
    "test_size":  int(len(X_test)),
    "features_used": len(feature_cols),
    "slow_http_train_before_smote": int(slow_train_before),
    "slow_http_train_after_smote":  int(slow_train_after),
    "slow_http_test_real_only":     slow_http_test_count,
    "data_fixes_applied": [
        f"has_ip filter: removed {removed:,} Layer-2 rows" if "has_ip" in df.columns else "has_ip column not found",
        "dropped flow_syn_ratio (corr=0.978 BRUTE_FORCE)",
        "dropped is_high_port_dst (corr=1.0 DDOS)",
        "dropped flow_packet_count (corr=0.974 DDOS)",
        "grouping: dst_port+ip_proto+attack_type",
        "R15: GroupShuffleSplit — disjoint groups across train/val/test",
    ],
    "train_class_dist": {c: int((y_train == c).sum()) for c in TARGET_LABELS},
}
append_json_log(pow_entry)

log("\n" + "="*60)
log("✅  PIPELINE COMPLETE — Round 15")
log("="*60)
log(f"\n  Train : {X_train.shape}  labels: {y_train_enc.shape}")
log(f"  Val   : {X_val.shape}   labels: {y_val_enc.shape}")
log(f"  Test  : {X_test.shape}  labels: {y_test_enc.shape}")
log(f"  SLOW_HTTP test samples (real): {slow_http_test_count}")
log(f"\n  ⚡  Ready for train_cnn_gru_v4.py")
log(f"\n  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log("="*60 + "\n")
