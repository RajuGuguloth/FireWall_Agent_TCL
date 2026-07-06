"""
Step 3 + 4: Extract 17-feature dataset from ALL attack PCAPs
- Port Scan  → PORT_SCAN
- DNS Tunnel → DNS_TUNNELING
- BruteForce → BRUTE_FORCE
- Slow HTTP  → SLOW_HTTP
- HTTP Flood → DDOS_HTTP_FLOOD

Saves: combined_dataset_v5_expanded.csv
Uses EXACTLY the 17 features from Round 16 (no leaking columns).
"""

import os, sys
import pandas as pd
import numpy as np
from scapy.all import PcapReader, IP, TCP, UDP, ICMP, Raw

# ── The 17 exact features used in Round 16 ──────────────────
# Derived from final_feature_list.txt minus:
#   zero-variance:  ip_header_length, packet_direction, ip_version,
#                   flow_std_pkt_len, flow_ack_ratio, is_layer2_only, has_ip
#   leaking:        is_well_known_port, is_syn, tcp_window_size, flow_mean_entropy
#   dropped R16:    src_port (not in model input, used for grouping only)
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

ATTACK_MAP = {
    "bruteforce": "BRUTE_FORCE",
    "dnstunnel":  "DNS_TUNNELING",
    "portscan":   "PORT_SCAN",
    "slowhttp":   "SLOW_HTTP",
    "httpflood":  "DDOS_HTTP_FLOOD",
}

ATTACK_DIR = "data/raw/attacks"
OUT_CSV    = "combined_dataset_v5_expanded.csv"
MAX_PKTS_PER_FILE = 10000  # cap per file to avoid huge CSVs

def entropy(data: bytes) -> float:
    if not data:
        return 0.0
    freq = np.bincount(list(data), minlength=256) / len(data)
    freq = freq[freq > 0]
    return float(-np.sum(freq * np.log2(freq)))

def extract_packet_features(pkt, label, attack_type):
    row = {f: 0 for f in FEATURES_17}
    row["label"] = label
    row["attack_type"] = attack_type

    row["packet_length"] = len(pkt)

    if pkt.haslayer(IP):
        ip = pkt[IP]
        row["ip_ttl"]   = ip.ttl
        row["ip_proto"] = ip.proto
        row["has_tcp"]  = int(pkt.haslayer(TCP))
        row["has_udp"]  = int(pkt.haslayer(UDP))
        row["has_icmp"] = int(pkt.haslayer(ICMP))

        if pkt.haslayer(TCP):
            tcp = pkt[TCP]
            flags = int(tcp.flags)
            row["is_ack"] = int(bool(flags & 0x10))
            row["is_rst"] = int(bool(flags & 0x04))
            row["is_fin"] = int(bool(flags & 0x01))
            row["is_psh"] = int(bool(flags & 0x08))
            row["tcp_flags"]        = flags
            row["dst_port"]         = tcp.dport
            row["is_high_port_src"] = int(tcp.sport > 1024)
        elif pkt.haslayer(UDP):
            udp = pkt[UDP]
            row["dst_port"]         = udp.dport
            row["is_high_port_src"] = int(udp.sport > 1024)

    payload = bytes(pkt[Raw]) if pkt.haslayer(Raw) else b""
    row["payload_length"]  = len(payload)
    row["payload_entropy"] = entropy(payload)
    return row

def extract_pcap(fpath, label, attack_type, max_pkts=MAX_PKTS_PER_FILE):
    rows = []
    total_bytes = 0
    try:
        with PcapReader(fpath) as pcap:
            for i, pkt in enumerate(pcap):
                if i >= max_pkts:
                    break
                try:
                    row = extract_packet_features(pkt, label, attack_type)
                    total_bytes += row["packet_length"]
                    rows.append(row)
                except Exception:
                    pass
    except Exception as e:
        print(f"  ⚠️  Error reading {fpath}: {e}")
        return []

    # Add flow-level aggregates over the whole file batch
    if rows:
        pkt_lens = [r["packet_length"] for r in rows]
        mean_len = float(np.mean(pkt_lens))
        for r in rows:
            r["flow_total_bytes"]   = total_bytes
            r["flow_mean_pkt_len"]  = mean_len
    return rows

def main():
    print("=" * 60)
    print("  STEP 3+4: Feature Extraction → combined_dataset_v5_expanded.csv")
    print("=" * 60)

    all_rows = []
    class_counts = {}

    pcap_files = sorted(os.listdir(ATTACK_DIR))
    for fname in pcap_files:
        if not fname.endswith(".pcap"):
            continue
        prefix = fname.split("_")[0].lower().replace("dnstunnel", "dnstunnel")
        # match prefix
        label = None
        for key, lbl in ATTACK_MAP.items():
            if fname.lower().startswith(key):
                label = lbl
                break
        if label is None:
            print(f"  ⚠️  Skipped (unknown type): {fname}")
            continue

        fpath = os.path.join(ATTACK_DIR, fname)
        print(f"  [{label}] {fname} ...", end="", flush=True)
        rows = extract_pcap(fpath, label, label)
        all_rows.extend(rows)
        class_counts[label] = class_counts.get(label, 0) + len(rows)
        print(f" {len(rows):,} rows")

    df = pd.DataFrame(all_rows, columns=FEATURES_17 + ["label", "attack_type"])
    df.to_csv(OUT_CSV, index=False)

    print(f"\n{'─'*60}")
    print(f"  STEP 3+4 COMPLETE")
    print(f"  Total rows extracted: {len(df):,}")
    print(f"  Class breakdown:")
    for cls, cnt in sorted(class_counts.items()):
        print(f"    {cls:<22}: {cnt:,}")
    print(f"  Saved → {OUT_CSV}")
    print(f"{'─'*60}")

if __name__ == "__main__":
    main()
