from scapy.all import rdpcap, TCP, UDP, IP, IPv6
import numpy as np
import os
import sys

# -------- helpers --------

def port_bucket(port: int) -> int:
    if port < 1024:
        return 0
    elif port < 49152:
        return 1
    else:
        return 2


def extract_packet_features(packets):
    features = []
    last_time = None

    for pkt in packets:
        if IP in pkt:
            ip_layer = pkt[IP]
        elif IPv6 in pkt:
            ip_layer = pkt[IPv6]
        else:
            continue

        pkt_len = len(pkt)
        now = float(pkt.time)
        iat = 0.0 if last_time is None else now - last_time
        last_time = now

        proto = [0, 0, 0]   # TCP, UDP, ICMP
        flags = [0] * 6     # FIN SYN RST PSH ACK URG
        src_port = 0
        dst_port = 0

        if TCP in pkt:
            proto[0] = 1
            src_port = pkt[TCP].sport
            dst_port = pkt[TCP].dport
            flags = [
                int(pkt[TCP].flags.F),
                int(pkt[TCP].flags.S),
                int(pkt[TCP].flags.R),
                int(pkt[TCP].flags.P),
                int(pkt[TCP].flags.A),
                int(pkt[TCP].flags.U),
            ]
        elif UDP in pkt:
            proto[1] = 1
            src_port = pkt[UDP].sport
            dst_port = pkt[UDP].dport
        else:
            proto[2] = 1

        feat = [
            pkt_len,
            iat,
            *proto,
            port_bucket(src_port),
            port_bucket(dst_port),
            *flags
        ]

        features.append(feat)

    return np.array(features, dtype=np.float32)

# -------- main runner --------

def process_pcap(pcap_path, out_dir):
    packets = rdpcap(pcap_path)
    feats = extract_packet_features(packets)

    if len(feats) == 0:
        print(f"[WARN] No usable packets in {pcap_path}")
        return

    base = os.path.splitext(os.path.basename(pcap_path))[0]
    out_path = os.path.join(out_dir, base + ".npy")
    np.save(out_path, feats)

    print(f"[OK] {pcap_path} → {out_path} | shape={feats.shape}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python extract_features.py <pcap_file> <output_dir>")
        sys.exit(1)

    pcap_file = sys.argv[1]
    output_dir = sys.argv[2]
    os.makedirs(output_dir, exist_ok=True)

    process_pcap(pcap_file, output_dir)

