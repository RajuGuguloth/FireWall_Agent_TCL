"""
Parse PCAP files and extract packet-level information
"""
import json
from pathlib import Path
from scapy.all import rdpcap
from tqdm import tqdm
import pandas as pd

class PCAPParser:
    def __init__(self, metadata_path="data/metadata.json"):
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
    
    def parse_packet(self, pkt, label, attack_type):
        """Extract features from a single packet"""
        features = {
            'label': label,
            'attack_type': attack_type if attack_type else 'BENIGN',
            'packet_length': len(pkt),
            'has_ip': int(pkt.haslayer('IP')),
            'has_tcp': int(pkt.haslayer('TCP')),
            'has_udp': int(pkt.haslayer('UDP')),
            'has_icmp': int(pkt.haslayer('ICMP')),
        }
        
        # IP layer features
        if pkt.haslayer('IP'):
            features['ip_version'] = pkt['IP'].version
            features['ip_ttl'] = pkt['IP'].ttl
            features['ip_proto'] = pkt['IP'].proto
        else:
            features['ip_version'] = 0
            features['ip_ttl'] = 0
            features['ip_proto'] = 0
        
        # TCP features
        if pkt.haslayer('TCP'):
            features['src_port'] = pkt['TCP'].sport
            features['dst_port'] = pkt['TCP'].dport
            features['tcp_flags'] = int(pkt['TCP'].flags)
        else:
            features['src_port'] = 0
            features['dst_port'] = 0
            features['tcp_flags'] = 0
        
        # UDP features
        if pkt.haslayer('UDP'):
            features['src_port'] = pkt['UDP'].sport
            features['dst_port'] = pkt['UDP'].dport
        
        return features
    
    def parse_all_pcaps(self, output_csv="data/processed/features/all_packets.csv"):
        """Parse all PCAP files and save features"""
        print("=" * 50)
        print("Parsing all PCAP files...")
        print("=" * 50)
        
        all_features = []
        
        for file_info in tqdm(self.metadata['files'], desc="Processing files"):
            filepath = file_info['filepath']
            label = file_info['label']
            attack_type = file_info.get('attack_type')
            
            print(f"\nParsing: {Path(filepath).name}")
            
            try:
                packets = rdpcap(filepath)
                
                for pkt in tqdm(packets[:10000], desc="  Packets", leave=False):  # Limit for speed
                    features = self.parse_packet(pkt, label, attack_type)
                    all_features.append(features)
                    
            except Exception as e:
                print(f"Error parsing {filepath}: {e}")
        
        # Convert to DataFrame
        df = pd.DataFrame(all_features)
        
        # Save
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        
        print(f"\n✅ Features saved to: {output_csv}")
        print(f"   Total packets: {len(df)}")
        print(f"   Benign: {len(df[df['label'] == 'BENIGN'])}")
        print(f"   Malicious: {len(df[df['label'] == 'MALICIOUS'])}")
        
        return df

if __name__ == "__main__":
    parser = PCAPParser()
    df = parser.parse_all_pcaps()
    
    # Print dataset summary
    print("\n" + "=" * 50)
    print("DATASET SUMMARY")
    print("=" * 50)
    print(df.groupby(['label', 'attack_type']).size())
