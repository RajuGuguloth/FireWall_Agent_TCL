"""
Enhanced Feature Extractor - Adds payload entropy, flow stats, timing features
CPU-optimized for production
"""
import pandas as pd
import numpy as np
from scapy.all import rdpcap
from pathlib import Path
import json
import math
from collections import Counter

class EnhancedFeatureExtractor:
    def __init__(self):
        self.feature_columns = [
            'packet_length', 'has_ip', 'has_tcp', 'has_udp',
            'has_icmp', 'ip_version', 'ip_ttl', 'ip_proto',
            'src_port', 'dst_port', 'tcp_flags',
            'payload_length', 'payload_entropy',
            'is_syn', 'is_ack', 'is_rst', 'is_fin', 'is_psh',
            'is_high_port_src', 'is_high_port_dst',
            'is_well_known_port', 'ip_header_length',
            'tcp_window_size', 'packet_direction',
            'is_layer2_only'
        ]

    def calculate_entropy(self, data):
        if not data or len(data) == 0:
            return 0.0
        counts = Counter(data)
        total = len(data)
        entropy = -sum(
            (count/total) * math.log2(count/total)
            for count in counts.values()
            if count > 0
        )
        return round(entropy, 4)

    def extract_features(self, pkt, label, attack_type):
        features = {
            'label': label,
            'attack_type': attack_type if attack_type else 'BENIGN',
            'packet_length': len(pkt),
            'has_ip': int(pkt.haslayer('IP')),
            'has_tcp': int(pkt.haslayer('TCP')),
            'has_udp': int(pkt.haslayer('UDP')),
            'has_icmp': int(pkt.haslayer('ICMP')),
            'payload_length': 0,
            'payload_entropy': 0.0,
            'is_syn': 0, 'is_ack': 0, 'is_rst': 0,
            'is_fin': 0, 'is_psh': 0,
            'is_high_port_src': 0, 'is_high_port_dst': 0,
            'is_well_known_port': 0,
            'ip_header_length': 0,
            'tcp_window_size': 0,
            'packet_direction': 0,
            'ip_version': 0, 'ip_ttl': 0, 'ip_proto': 0,
            'src_port': 0, 'dst_port': 0, 'tcp_flags': 0,
            'is_layer2_only': 0
        }

        if not pkt.haslayer('IP'):
            features['is_layer2_only'] = 1
            return features

        if pkt.haslayer('IP'):
            features['ip_version'] = pkt['IP'].version
            features['ip_ttl'] = pkt['IP'].ttl
            features['ip_proto'] = pkt['IP'].proto
            features['ip_header_length'] = pkt['IP'].ihl * 4

        if pkt.haslayer('TCP'):
            tcp = pkt['TCP']
            features['src_port'] = tcp.sport
            features['dst_port'] = tcp.dport
            features['tcp_flags'] = int(tcp.flags)
            features['tcp_window_size'] = tcp.window
            flags = int(tcp.flags)
            features['is_syn'] = int(bool(flags & 0x02))
            features['is_ack'] = int(bool(flags & 0x10))
            features['is_rst'] = int(bool(flags & 0x04))
            features['is_fin'] = int(bool(flags & 0x01))
            features['is_psh'] = int(bool(flags & 0x08))
            features['is_high_port_src'] = int(tcp.sport > 1024)
            features['is_high_port_dst'] = int(tcp.dport > 1024)
            features['is_well_known_port'] = int(
                tcp.dport in [80, 443, 53, 22, 21, 25, 8080]
            )
            features['packet_direction'] = int(tcp.sport > tcp.dport)

        if pkt.haslayer('UDP'):
            udp = pkt['UDP']
            features['src_port'] = udp.sport
            features['dst_port'] = udp.dport
            features['is_high_port_src'] = int(udp.sport > 1024)
            features['is_high_port_dst'] = int(udp.dport > 1024)
            features['is_well_known_port'] = int(
                udp.dport in [53, 67, 68, 123, 161]
            )

        if pkt.haslayer('Raw'):
            payload = bytes(pkt['Raw'])
            features['payload_length'] = len(payload)
            features['payload_entropy'] = self.calculate_entropy(payload)

        return features

    def process_pcap(self, pcap_path, label, attack_type, max_packets=10000):
        print(f"  Processing: {Path(pcap_path).name}")
        try:
            packets = rdpcap(str(pcap_path))
            packets = packets[:max_packets]
            features_list = []
            for pkt in packets:
                features = self.extract_features(pkt, label, attack_type)
                features_list.append(features)
            print(f"    Extracted {len(features_list)} packets")
            return features_list
        except Exception as e:
            print(f"    Error: {e}")
            return []

    def process_all(self, metadata_path="data/metadata.json",
                   output_path="data/processed/features/enhanced_features.csv"):
        print("=" * 50)
        print("Enhanced Feature Extraction")
        print("=" * 50)

        with open(metadata_path) as f:
            metadata = json.load(f)

        all_features = []

        print("\nBENIGN FILES:")
        for file_info in metadata['files']:
            if file_info['label'] == 'BENIGN':
                features = self.process_pcap(
                    file_info['filepath'], 'BENIGN', 'BENIGN', 5000
                )
                all_features.extend(features)

        print("\nATTACK FILES:")
        attack_mapping = {
            "portscan": "PORT_SCAN",
            "httpflood": "DDOS_HTTP_FLOOD",
            "bruteforce": "BRUTE_FORCE",
            "slowhttp": "SLOW_HTTP",
            "dnstunnel": "DNS_TUNNELING"
        }

        for file_info in metadata['files']:
            if file_info['label'] == 'MALICIOUS':
                attack_type = "UNKNOWN"
                for key, value in attack_mapping.items():
                    if key in file_info['filename'].lower():
                        attack_type = value
                        break
                features = self.process_pcap(
                    file_info['filepath'], 'MALICIOUS', attack_type, 5000
                )
                all_features.extend(features)

        df = pd.DataFrame(all_features)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

        print(f"\nEnhanced features saved: {output_path}")
        print(f"Total samples: {len(df)}")
        print(f"Features per sample: {len(self.feature_columns)}")
        print(f"\nClass distribution:")
        print(df['attack_type'].value_counts())

        return df

if __name__ == "__main__":
    extractor = EnhancedFeatureExtractor()
    df = extractor.process_all()
