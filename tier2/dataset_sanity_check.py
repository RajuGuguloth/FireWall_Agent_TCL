import pandas as pd
import numpy as np

print("Loading v4...")
df = pd.read_csv("combined_dataset_v4_flow.csv")

print("\n1. CLASS DISTRIBUTION")
print(df['attack_type'].value_counts())

print("\n2. FLOW_PACKET_COUNT per class (reveals granularity mismatch)")
for cls in df['attack_type'].unique():
    subset = df[df['attack_type']==cls]['flow_packet_count']
    print(f"  {cls:<20} min={subset.min():.0f}  max={subset.max():.0f}  "
          f"mean={subset.mean():.1f}  all_same={(subset.nunique()==1)}")

print("\n3. HAS_IP per class")
for cls in df['attack_type'].unique():
    subset = df[df['attack_type']==cls]
    no_ip = (subset['has_ip']==0).sum()
    has_ip = (subset['has_ip']==1).sum()
    print(f"  {cls:<20} has_ip=1: {has_ip:>8,}  has_ip=0: {no_ip:>8,}")

print("\n4. WHAT SCRIPTS CREATED THIS FILE?")
import os
scripts = ['extract_flow_features.py','extract_large.py',
           'merge_and_extract.py','create_mixed.py',
           'generate_synthetic_attacks.py']
for s in scripts:
    if os.path.exists(s):
        mtime = os.path.getmtime(s)
        import datetime
        dt = datetime.datetime.fromtimestamp(mtime)
        print(f"  {s:<40} modified: {dt.strftime('%Y-%m-%d %H:%M:%S')}")

print("\n5. CHECK V3 CLASSES")
df3 = pd.read_csv("combined_dataset_v3.csv", nrows=100000)
print("  v3 attack_type values:", df3['attack_type'].unique())
print("  v3 shape (sample):", df3.shape)
