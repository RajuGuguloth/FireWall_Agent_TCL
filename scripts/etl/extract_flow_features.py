import pandas as pd
import numpy as np

print("Loading dataset...")
df = pd.read_csv("combined_dataset_v3.csv")

print("Computing flow features...")

# ── Group by src_port + dst_port + ip_proto (flow key) ──
df = df.sort_values(["src_port", "dst_port", "ip_proto"]).reset_index(drop=True)
grp = df.groupby(["src_port", "dst_port", "ip_proto"])

# Flow packet count
df["flow_packet_count"] = grp["packet_length"].transform("count")

# Flow byte rate (total bytes in flow)
df["flow_total_bytes"] = grp["packet_length"].transform("sum")

# Mean and std of packet length within flow
df["flow_mean_pkt_len"] = grp["packet_length"].transform("mean").round(4)
df["flow_std_pkt_len"]  = grp["packet_length"].transform("std").fillna(0).round(4)

# Mean payload entropy within flow
df["flow_mean_entropy"] = grp["payload_entropy"].transform("mean").round(4)

# SYN ratio within flow
df["flow_syn_ratio"] = grp["is_syn"].transform("mean").round(4)

# ACK ratio within flow
df["flow_ack_ratio"] = grp["is_ack"].transform("mean").round(4)

print(f"Done. New shape: {df.shape}")
print(f"New columns added: flow_packet_count, flow_total_bytes, flow_mean_pkt_len,")
print(f"                   flow_std_pkt_len, flow_mean_entropy, flow_syn_ratio, flow_ack_ratio")

df.to_csv("combined_dataset_v4_flow.csv", index=False)
print(f"\n✅ Saved → combined_dataset_v4_flow.csv")
print(f"Total features now: {len(df.columns) - 2} (excl. attack_type + label)")
