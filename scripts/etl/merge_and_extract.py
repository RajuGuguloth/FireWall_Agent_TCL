from scapy.all import rdpcap, IP, TCP, UDP, ICMP
import pandas as pd, math, os

WELL_KNOWN = {20,21,22,23,25,53,80,110,143,443,445,3306,3389,8080}

PCAP_LABELS = {
    "data/raw/attacks/portscan_attack.pcap":      ("PORT_SCAN",      1),
    "data/raw/attacks/portscan_syn.pcap":         ("PORT_SCAN",      1),
    "data/raw/attacks/portscan_connect.pcap":     ("PORT_SCAN",      1),
    "data/raw/attacks/portscan_aggressive.pcap":  ("PORT_SCAN",      1),
    "data/raw/attacks/portscan_fast.pcap":        ("PORT_SCAN",      1),
    "data/raw/attacks/portscan_udp.pcap":         ("PORT_SCAN",      1),
    "data/raw/attacks/httpflood_attack.pcap":     ("DDOS_HTTP_FLOOD",1),
    "data/raw/attacks/bruteforce_attack.pcap":    ("BRUTE_FORCE",    1),
    "data/raw/attacks/slowhttp_attack.pcap":      ("SLOW_HTTP",      1),
    "data/raw/attacks/dnstunnel_attack.pcap":     ("DNS_TUNNELING",  1),
    "data/raw/attacks/portscan_v2.pcap":          ("PORT_SCAN",      1),
    "data/raw/attacks/httpflood_v2.pcap":         ("DDOS_HTTP_FLOOD",1),
    "data/raw/attacks/bruteforce_v2.pcap":        ("BRUTE_FORCE",    1),
    "data/raw/attacks/slowhttp_v2.pcap":          ("SLOW_HTTP",      1),
    "data/raw/attacks/dnstunnel_v2.pcap":         ("DNS_TUNNELING",  1),
    "data/raw/benign/baseline_20260215_094524.pcap": ("BENIGN",      0),
    "data/raw/benign/baseline_20260215_101846.pcap": ("BENIGN",      0),
    "data/raw/benign/dns_only.pcap":              ("BENIGN",         0),
    "data/raw/benign/https_only.pcap":            ("BENIGN",         0),
}

def entropy(payload):
    if not payload: return 0.0
    counts = {}
    for b in payload: counts[b] = counts.get(b, 0) + 1
    t = len(payload)
    return -sum((c/t)*math.log2(c/t) for c in counts.values())

rows = []
for pcap_file, (attack_type, label) in PCAP_LABELS.items():
    if not os.path.exists(pcap_file):
        print(f"⚠️  Skipping: {pcap_file}")
        continue
    packets = rdpcap(pcap_file)
    print(f"✅ {attack_type:20s} ← {os.path.basename(pcap_file)} ({len(packets):,} pkts)")
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
                "label":              label,
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

real_df      = pd.DataFrame(rows)
synthetic_df = pd.read_csv("enhanced_synthetic_attacks.csv")
combined     = pd.concat([synthetic_df, real_df], ignore_index=True)
combined     = combined.sample(frac=1, random_state=42).reset_index(drop=True)
combined.to_csv("combined_dataset_v2.csv", index=False)

print(f"\nSynthetic : {len(synthetic_df):>10,}")
print(f"Real      : {len(real_df):>10,}")
print(f"Combined  : {len(combined):>10,}")
print("\nClass breakdown:")
print(combined["attack_type"].value_counts().to_string())
