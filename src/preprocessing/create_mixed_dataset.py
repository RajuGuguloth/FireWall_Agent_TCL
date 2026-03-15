"""
Create realistic mixed traffic (90% benign + 10% attacks)
"""
from scapy.all import rdpcap, wrpcap
import random
from pathlib import Path

def create_mixed_pcap():
    print("=" * 50)
    print("Creating Mixed Production Dataset")
    print("=" * 50)
    
    # Load benign traffic (main dataset)
    print("\nLoading benign traffic...")
    benign = rdpcap("data/raw/benign/baseline_20260215_101846.pcap")
    print(f"  Loaded {len(benign)} benign packets")
    
    # Load attack samples
    print("\nLoading attack traffic...")
    attacks = []
    
    attack_files = [
        "data/raw/attacks/httpflood_attack.pcap",
        "data/raw/attacks/bruteforce_attack.pcap",
        "data/raw/attacks/slowhttp_attack.pcap",
        "data/raw/attacks/dnstunnel_attack.pcap",
        "data/raw/attacks/portscan_aggressive.pcap"
    ]
    
    for f in attack_files:
        if Path(f).exists():
            pkts = rdpcap(f)
            attacks.extend(pkts[:50])  # Take 50 packets from each
            print(f"  Added {min(50, len(pkts))} from {Path(f).name}")
    
    print(f"\nTotal attack packets: {len(attacks)}")
    
    # Mix: Insert attacks at random positions
    print("\nMixing traffic...")
    mixed = list(benign[:10000])  # Take 10K benign packets
    
    # Shuffle attack positions
    attack_positions = random.sample(range(len(mixed)), len(attacks))
    
    for i, pos in enumerate(sorted(attack_positions)):
        mixed.insert(pos + i, attacks[i])
    
    print(f"  Total mixed packets: {len(mixed)}")
    print(f"  Benign: {len(mixed) - len(attacks)} ({100*(len(mixed)-len(attacks))/len(mixed):.1f}%)")
    print(f"  Attacks: {len(attacks)} ({100*len(attacks)/len(mixed):.1f}%)")
    
    # Save
    output_path = "data/raw/mixed_production.pcap"
    wrpcap(output_path, mixed)
    print(f"\n✅ Saved: {output_path}")
    
    return len(mixed), len(attacks)

if __name__ == "__main__":
    create_mixed_pcap()
