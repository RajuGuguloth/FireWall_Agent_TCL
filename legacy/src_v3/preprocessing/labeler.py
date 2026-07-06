"""
Label PCAP files with attack types and create metadata
"""
import os
import json
from pathlib import Path
from scapy.all import rdpcap, PcapReader
from datetime import datetime

class PCAPLabeler:
    def __init__(self, data_dir="data/raw"):
        self.data_dir = Path(data_dir)
        self.benign_dir = self.data_dir / "benign"
        self.attack_dir = self.data_dir / "attacks"
        
    def get_pcap_stats(self, pcap_path):
        """Extract basic statistics from PCAP file"""
        try:
            packets = rdpcap(str(pcap_path))
            
            stats = {
                "filename": pcap_path.name,
                "filepath": str(pcap_path),
                "total_packets": len(packets),
                "file_size_mb": pcap_path.stat().st_size / (1024 * 1024),
                "protocols": {},
                "created_date": datetime.fromtimestamp(pcap_path.stat().st_ctime).isoformat()
            }
            
            # Count protocols
            for pkt in packets:
                proto = pkt.sprintf("%IP.proto%") if pkt.haslayer("IP") else "Other"
                stats["protocols"][proto] = stats["protocols"].get(proto, 0) + 1
            
            return stats
        except Exception as e:
            print(f"Error processing {pcap_path}: {e}")
            return None
    
    def label_benign_files(self):
        """Label all benign traffic files"""
        benign_metadata = []
        
        for pcap_file in self.benign_dir.glob("*.pcap"):
            print(f"Processing benign: {pcap_file.name}")
            stats = self.get_pcap_stats(pcap_file)
            
            if stats:
                stats["label"] = "BENIGN"
                stats["attack_type"] = None
                benign_metadata.append(stats)
        
        return benign_metadata
    
    def label_attack_files(self):
        """Label all attack traffic files"""
        attack_metadata = []
        
        # Attack type mapping based on filename
        attack_mapping = {
            "portscan": "PORT_SCAN",
            "httpflood": "DDOS_HTTP_FLOOD",
            "bruteforce": "BRUTE_FORCE",
            "slowhttp": "SLOW_HTTP",
            "dnstunnel": "DNS_TUNNELING"
        }
        
        for pcap_file in self.attack_dir.glob("*.pcap"):
            print(f"Processing attack: {pcap_file.name}")
            stats = self.get_pcap_stats(pcap_file)
            
            if stats:
                stats["label"] = "MALICIOUS"
                
                # Determine attack type from filename
                attack_type = "UNKNOWN"
                for key, value in attack_mapping.items():
                    if key in pcap_file.name.lower():
                        attack_type = value
                        break
                
                stats["attack_type"] = attack_type
                attack_metadata.append(stats)
        
        return attack_metadata
    
    def create_metadata_file(self, output_path="data/metadata.json"):
        """Create complete metadata file"""
        print("=" * 50)
        print("Creating metadata for all PCAP files...")
        print("=" * 50)
        
        benign_data = self.label_benign_files()
        attack_data = self.label_attack_files()
        
        metadata = {
            "created_at": datetime.now().isoformat(),
            "total_files": len(benign_data) + len(attack_data),
            "benign_files": len(benign_data),
            "attack_files": len(attack_data),
            "files": benign_data + attack_data
        }
        
        # Save metadata
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✅ Metadata saved to: {output_path}")
        print(f"   Total files: {metadata['total_files']}")
        print(f"   Benign: {metadata['benign_files']}")
        print(f"   Attacks: {metadata['attack_files']}")
        
        return metadata

if __name__ == "__main__":
    labeler = PCAPLabeler()
    metadata = labeler.create_metadata_file()
