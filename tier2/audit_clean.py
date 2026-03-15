import pandas as pd
import numpy as np

print("Loading v4...")
df = pd.read_csv("combined_dataset_v4_flow.csv")

# Apply IP filter
df_clean = df[df['has_ip'] == 1].copy()
print(f"After has_ip filter: {len(df_clean):,} rows")
print("\nClean class distribution:")
print(df_clean['attack_type'].value_counts())

# Check if enough for training
print("\nGrouping viability on clean data:")
targets = ['BRUTE_FORCE','DDOS_HTTP_FLOOD','SLOW_HTTP']
df3 = df_clean[df_clean['attack_type'].isin(targets)].copy()

# Time window grouping
df3 = df3.sort_values(['dst_port','ip_proto','attack_type'])
df3['row_within_group'] = df3.groupby(
    ['dst_port','ip_proto','attack_type']).cumcount()
df3['time_window'] = df3['row_within_group'] // 300

groups = df3.groupby(['dst_port','ip_proto','attack_type','time_window']).size()
viable = groups[groups >= 20]
print(f"Total groups: {len(groups)}")
print(f"Viable groups (>=20 pkts): {len(viable)}")

# Per-class viable groups
for cls in targets:
    cls_viable = [(k,v) for k,v in viable.items() if k[2]==cls]
    total_seqs = sum((v-20)//10+1 for _,v in cls_viable)
    print(f"  {cls:<25} {len(cls_viable):>3} groups  ~{total_seqs:>4} sequences")

# Check dst_port diversity
print("\nDst port diversity (clean IP rows):")
for cls in targets:
    ports = df3[df3['attack_type']==cls]['dst_port'].value_counts()
    print(f"  {cls:<25} {len(ports)} unique dst_ports: {ports.index.tolist()[:5]}")

print("\nDecision:")
total_viable = len(viable)
if total_viable >= 15:
    print(f"  ✅ {total_viable} viable groups — enough for R16 training")
else:
    print(f"  ❌ Only {total_viable} viable groups — need more captures")
    per_class = {}
    for cls in targets:
        cls_viable = [(k,v) for k,v in viable.items() if k[2]==cls]
        per_class[cls] = len(cls_viable)
        if len(cls_viable) < 5:
            print(f"  ⚠️  {cls} has only {len(cls_viable)} groups — needs more Docker captures")
