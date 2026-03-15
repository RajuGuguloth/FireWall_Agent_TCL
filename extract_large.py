from scapy.all import rdpcap, IP, TCP, UDP, ICMP
import pandas as pd, math, os, glob

WELL_KNOWN = {20,21,22,23,25,53,80,110,143,443,445,3306,3389,8080}

PATTERN_LABELS = {
    "bruteforce": "BRUTE_FORCE",
    "dnstunnel":  "DNS_TUNNELING",
    "portscan":   "PORT_SCAN",
    "slowhttp":   "SLOW_HTTP",
    "httpflood":  "DDOS_HTTP_FLOOD",
}

def entropy(payload):
    if not payload: return 0.0
    counts = {}
    for b in payload: counts[b] = counts.get(b, 0) + 1
    t = len(payload)
    return -sum((c/t)*math.log2(c/t) for c in counts.values())

rows = []
files = sorted(glob.glob("data/raw/attacks/*.pcap"))
total_files = len(files)

for idx, pcap_file in enumerate(files, 1):
    basename = os.path.basename(pcap_file)
    attack_type = next((v for k, v in PATTERN_LABELS.items() if k in basename), None)
    if not attack_type:
        continue
    print(f"[{idx}/{total_files}] {basename} → {attack_type}", flush=True)
    packets = rdpcap(pcap_file)
    for pkt in packets:
        try:
            has_ip   = int(pkt.haslayer(IP))
            has_tcp  = int(pkt.haslayer(TCP))
            has_udp  = int(pkt.haslayer(UDP))
            has_icmp = int(pkt.haslayer(ICMP))
            payload  = bytes(pkt[TCP].payload) if has_tcp else \
                       bytes(pkt[UDP].payload) if has_udp else b""
            ip  = pkt[IP]  if has_ip  else None
            tcp = pkt[TCP] if has_tcp else None
            udp = pkt[UDP] if has_udp else None
            src_port = tcp.sport if tcp else (udp.sport if udp else 0)
            dst_port = tcp.dport if tcp else (udp.dport if udp else 0)
            flags    = int(tcp.flags) if tcp else 0
            rows.append({
                "attack_type":        attack_type,
                "label":              1,
                "packet_length":      len(pkt),
                "has_ip":             has_ip,
                "has_tcp":            has_tcp,
                "has_udp":            has_udp,
                "has_icmp":           has_icmp,
                "payload_length":     len(payload),
                "payload_entropy":    round(entropy(payload), 4),
                "is_syn":             int(bool(flags & 0x02)),
                "is_ack":             int(bool(flags & 0x10)),
                "is_rst":             int(bool(flags & 0x04)),
                "is_fin":             int(bool(flags & 0x01)),
                "is_psh":             int(bool(flags & 0x08)),
                "is_high_port_src":   int(src_port > 1024),
                "is_high_port_dst":   int(dst_port > 1024),
                "is_well_known_port": int(src_port in WELL_KNOWN or dst_port in WELL_KNOWN),
                "ip_header_length":   ip.ihl * 4 if ip else 0,
                "tcp_window_size":    tcp.window if tcp else 0,
                "packet_direction":   1,
                "ip_version":         ip.version if ip else 0,
                "ip_ttl":             ip.ttl if ip else 0,
                "ip_proto":           ip.proto if ip else 0,
                "src_port":           src_port,
                "dst_port":           dst_port,
                "tcp_flags":          flags,
                "is_layer2_only":     int(not has_ip),
            })
        except Exception:
            continue

print(f"\nExtracted {len(rows):,} rows from attack pcaps")
print("Loading existing combined_dataset_v2.csv...")
new_df      = pd.DataFrame(rows)
existing_df = pd.read_csv("combined_dataset_v2.csv")
final_df    = pd.concat([existing_df, new_df], ignore_index=True)
final_df    = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
final_df.to_csv("combined_dataset_v3.csv", index=False)

print(f"\n✅ Done!")
print(f"Previous : {len(existing_df):,}")
print(f"New rows : {len(new_df):,}")
print(f"Final    : {len(final_df):,}")
print("\nClass breakdown:")
print(final_df["attack_type"].value_counts().to_string())
