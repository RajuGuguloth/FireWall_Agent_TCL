import pandas as pd
import numpy as np
from scipy.stats import pointbiserialr
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

CSV = "combined_dataset_v4_flow.csv"
TARGET = ["BRUTE_FORCE", "DDOS_HTTP_FLOOD", "SLOW_HTTP"]
WINDOW = 20

print("Loading...")
df = pd.read_csv(CSV)
print(f"Raw shape: {df.shape[0]:,} rows x {df.shape[1]} cols")

# ── SECTION 1: COLUMN INVENTORY ──────────────────────
print("\n" + "="*60)
print("SECTION 1 — COLUMN INVENTORY")
print("="*60)
for col in df.columns:
    dtype = str(df[col].dtype)
    nuniq = df[col].nunique()
    sample = df[col].dropna().iloc[:3].tolist()
    print(f"  {col:<30} dtype={dtype:<10} unique={nuniq:<8} sample={sample}")

# ── SECTION 2: CLASS DISTRIBUTION (raw) ──────────────
print("\n" + "="*60)
print("SECTION 2 — CLASS DISTRIBUTION (before any filter)")
print("="*60)
label_col = "attack_type" if "attack_type" in df.columns else "label"
print(f"Using label column: '{label_col}'")
dist = df[label_col].value_counts()
for cls, cnt in dist.items():
    bar = "█" * int(cnt/len(df)*40)
    print(f"  {cls:<25} {cnt:>8,}  ({cnt/len(df)*100:.2f}%)  {bar}")

# ── SECTION 3: LAYER-2 CONTAMINATION ─────────────────
print("\n" + "="*60)
print("SECTION 3 — LAYER-2 CONTAMINATION")
print("="*60)
if "has_ip" in df.columns:
    no_ip = (df["has_ip"] == 0).sum()
    has_ip = (df["has_ip"] == 1).sum()
    print(f"  has_ip=0 (Layer-2 noise): {no_ip:,}  ({no_ip/len(df)*100:.1f}%)")
    print(f"  has_ip=1 (valid IP):      {has_ip:,}  ({has_ip/len(df)*100:.1f}%)")
    df_clean = df[df["has_ip"] == 1].copy()
    print(f"  After filter: {len(df_clean):,} rows remain")
else:
    print("  has_ip column NOT FOUND")
    df_clean = df.copy()

# ── SECTION 4: DATA QUALITY ───────────────────────────
print("\n" + "="*60)
print("SECTION 4 — DATA QUALITY")
print("="*60)
num_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
missing = df_clean[num_cols].isnull().sum()
missing_cols = missing[missing > 0]
inf_count = np.isinf(df_clean[num_cols].values).sum()
print(f"  Numeric columns:   {len(num_cols)}")
print(f"  Missing values:    {missing.sum():,}  {'✅' if missing.sum()==0 else '❌'}")
if len(missing_cols) > 0:
    for col, cnt in missing_cols.items():
        print(f"    {col}: {cnt:,} missing")
print(f"  Infinite values:   {inf_count:,}  {'✅' if inf_count==0 else '❌'}")

# Per-column stats
print("\n  Feature Statistics (numeric cols):")
print(f"  {'Column':<30} {'Min':>10} {'Max':>10} {'Mean':>10} {'Std':>10} {'Zeros%':>8}")
for col in num_cols:
    if col in [label_col, "label", "attack_type"]:
        continue
    col_data = df_clean[col].dropna()
    zeros_pct = (col_data == 0).sum() / len(col_data) * 100
    print(f"  {col:<30} {col_data.min():>10.3f} {col_data.max():>10.3f} "
          f"{col_data.mean():>10.3f} {col_data.std():>10.3f} {zeros_pct:>7.1f}%")

# ── SECTION 5: LEAKAGE CHECK ──────────────────────────
print("\n" + "="*60)
print("SECTION 5 — FEATURE LEAKAGE CHECK (|corr| > 0.70)")
print("="*60)
df3 = df_clean[df_clean[label_col].isin(TARGET)].copy()
df3[label_col] = df3[label_col].str.upper().str.strip()
exclude = ["has_ip","is_layer2_only","label","attack_type"]
feat_cols = [c for c in num_cols if c not in exclude]

leakage_found = []
for cls in TARGET:
    binary = (df3[label_col] == cls).astype(int)
    for col in feat_cols:
        try:
            corr, _ = pointbiserialr(df3[col].fillna(0), binary)
            if abs(corr) > 0.70:
                leakage_found.append((cls, col, round(corr, 4)))
        except: pass

if leakage_found:
    print(f"  ⚠️  {len(leakage_found)} potentially leaked features:")
    for cls, col, corr in sorted(leakage_found, key=lambda x: abs(x[2]), reverse=True):
        severity = "🔴 CRITICAL" if abs(corr) > 0.85 else "🟠 HIGH"
        print(f"  {severity}  {cls:<25} ← {col:<30} corr={corr}")
else:
    print("  ✅ No leakage detected above 0.70 threshold")

# ── SECTION 6: GROUPING ANALYSIS ─────────────────────
print("\n" + "="*60)
print("SECTION 6 — GROUPING ANALYSIS")
print("="*60)

group_configs = [
    ["src_port", "dst_port", "ip_proto"],
    ["dst_port", "ip_proto", label_col],
    ["dst_port", "ip_proto", label_col, "time_window_300"],
]

# Add time window column
df3 = df3.sort_values(["dst_port","ip_proto",label_col])
df3["row_within_group"] = df3.groupby(
    ["dst_port","ip_proto",label_col]).cumcount()
df3["time_window_300"] = df3["row_within_group"] // 300

for gcols in group_configs:
    available = [c for c in gcols if c in df3.columns]
    if len(available) != len(gcols):
        print(f"\n  Config {gcols} — skipped (missing cols)")
        continue
    g = df3.groupby(available).size()
    viable = (g >= WINDOW).sum()
    print(f"\n  Grouping: {available}")
    print(f"  Total groups:          {len(g):,}")
    print(f"  Groups with >= {WINDOW} pkts: {viable:,}")
    print(f"  Max group size:        {g.max():,}")
    print(f"  Median group size:     {g.median():.0f}")
    if viable > 0:
        total_seqs = sum((s - WINDOW)//10 + 1 for s in g if s >= WINDOW)
        print(f"  Estimated sequences:   ~{total_seqs:,}")
        print(f"  Group details (viable only):")
        for name, size in g[g >= WINDOW].items():
            seqs = (size - WINDOW)//10 + 1
            print(f"    {str(name):<50} {size:>6} pkts → ~{seqs:>5} seqs")

# ── SECTION 7: CLASS BALANCE IN 3-CLASS SUBSET ───────
print("\n" + "="*60)
print("SECTION 7 — 3-CLASS SUBSET ANALYSIS")
print("="*60)
for cls in TARGET:
    cnt = (df3[label_col] == cls).sum()
    print(f"  {cls:<25} {cnt:>8,} rows ({cnt/len(df3)*100:.1f}%)")

# ── SECTION 8: FEATURE CORRELATION MATRIX ────────────
print("\n" + "="*60)
print("SECTION 8 — INTER-FEATURE CORRELATION (top 10 highest pairs)")
print("="*60)
use_feats = [c for c in feat_cols if c in df3.columns][:20]
corr_matrix = df3[use_feats].corr().abs()
np.fill_diagonal(corr_matrix.values, 0)
pairs = []
for i in range(len(corr_matrix)):
    for j in range(i+1, len(corr_matrix)):
        pairs.append((corr_matrix.columns[i], 
                      corr_matrix.columns[j],
                      round(corr_matrix.iloc[i,j], 4)))
pairs.sort(key=lambda x: x[2], reverse=True)
for a, b, c in pairs[:10]:
    flag = "⚠️ " if c > 0.90 else "   "
    print(f"  {flag}{a:<30} ↔ {b:<30} corr={c}")

# ── SECTION 9: ENTROPY ANALYSIS ──────────────────────
print("\n" + "="*60)
print("SECTION 9 — PAYLOAD ENTROPY BY CLASS")
print("="*60)
if "payload_entropy" in df3.columns:
    for cls in TARGET:
        subset = df3[df3[label_col] == cls]["payload_entropy"]
        print(f"  {cls:<25} mean={subset.mean():.4f}  "
              f"std={subset.std():.4f}  "
              f"min={subset.min():.4f}  "
              f"max={subset.max():.4f}")

# ── SECTION 10: FINAL VERDICT ─────────────────────────
print("\n" + "="*60)
print("SECTION 10 — FINAL VERDICT")
print("="*60)

viable_groups_tw = df3.groupby(
    ["dst_port","ip_proto",label_col,"time_window_300"]).filter(
    lambda x: len(x) >= WINDOW).groupby(
    ["dst_port","ip_proto",label_col,"time_window_300"]).ngroups \
    if "time_window_300" in df3.columns else 0

critical_leaks = [x for x in leakage_found if abs(x[2]) > 0.85]
high_leaks = [x for x in leakage_found if 0.70 < abs(x[2]) <= 0.85]

print(f"  Total rows (raw):         {len(df):,}")
print(f"  Valid IP rows:            {len(df_clean):,}")
print(f"  3-class subset:           {len(df3):,}")
print(f"  Missing/Inf values:       {'✅ None' if missing.sum()+inf_count==0 else '❌ Present'}")
print(f"  Critical leaks (>0.85):   {'✅ None' if not critical_leaks else f'❌ {len(critical_leaks)}: {[x[1] for x in critical_leaks]}'}")
print(f"  High leaks (0.70-0.85):   {'✅ None' if not high_leaks else f'⚠️  {len(high_leaks)}: {[x[1] for x in high_leaks]}'}")
print(f"  Groups (dst+proto+class): {len(df3.groupby(['dst_port','ip_proto',label_col]))}")
print(f"  Groups with time windows: ~{viable_groups_tw} viable groups")
print(f"  Ready for R16 training:   {'✅ YES' if viable_groups_tw >= 15 else '❌ Need more groups — run time-window fix'}")
print("\n" + "="*60)
print("AUDIT COMPLETE — paste full output to Claude")
print("="*60)
