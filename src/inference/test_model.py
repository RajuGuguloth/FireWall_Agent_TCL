"""
Test trained model on new PCAP files
"""
import joblib
import pandas as pd
from scapy.all import rdpcap
from pathlib import Path
import json
import sys

class ModelTester:
    def __init__(self, model_dir="models/serialized"):
        """Load trained model and encoder"""
        print("=" * 50)
        print("Loading trained model...")
        print("=" * 50)
        
        self.model = joblib.load(f"{model_dir}/tier1_rf.pkl")
        self.label_encoder = joblib.load(f"{model_dir}/label_encoder.pkl")
        
        with open(f"{model_dir}/feature_names.json", 'r') as f:
            self.feature_columns = json.load(f)
        
        print(f"✅ Model loaded: Random Forest")
        print(f"✅ Classes: {list(self.label_encoder.classes_)}")
    
    def extract_features(self, pkt):
        """Extract features from a single packet"""
        features = {
            'packet_length': len(pkt),
            'has_ip': int(pkt.haslayer('IP')),
            'has_tcp': int(pkt.haslayer('TCP')),
            'has_udp': int(pkt.haslayer('UDP')),
            'has_icmp': int(pkt.haslayer('ICMP')),
        }
        
        # IP features
        if pkt.haslayer('IP'):
            features['ip_version'] = pkt['IP'].version
            features['ip_ttl'] = pkt['IP'].ttl
            features['ip_proto'] = pkt['IP'].proto
        else:
            features['ip_version'] = 0
            features['ip_ttl'] = 0
            features['ip_proto'] = 0
        
        # TCP/UDP features
        if pkt.haslayer('TCP'):
            features['src_port'] = pkt['TCP'].sport
            features['dst_port'] = pkt['TCP'].dport
            features['tcp_flags'] = int(pkt['TCP'].flags)
        elif pkt.haslayer('UDP'):
            features['src_port'] = pkt['UDP'].sport
            features['dst_port'] = pkt['UDP'].dport
            features['tcp_flags'] = 0
        else:
            features['src_port'] = 0
            features['dst_port'] = 0
            features['tcp_flags'] = 0
        
        return features
    
    def test_pcap(self, pcap_path, max_packets=1000):
        """Test model on a PCAP file"""
        print("\n" + "=" * 50)
        print(f"Testing: {Path(pcap_path).name}")
        print("=" * 50)
        
        try:
            packets = rdpcap(pcap_path)
            print(f"Total packets in file: {len(packets)}")
            
            # Limit for speed
            test_packets = packets[:max_packets]
            print(f"Testing first {len(test_packets)} packets...")
            
            # Extract features
            features_list = []
            for pkt in test_packets:
                features = self.extract_features(pkt)
                features_list.append(features)
            
            # Create DataFrame
            df = pd.DataFrame(features_list)
            X = df[self.feature_columns].values
            
            # Predict
            predictions = self.model.predict(X)
            probabilities = self.model.predict_proba(X)
            
            # Decode predictions
            predicted_labels = self.label_encoder.inverse_transform(predictions)
            
            # Get confidence scores
            max_probas = probabilities.max(axis=1)
            
            # Statistics
            unique, counts = pd.Series(predicted_labels).value_counts().items(), pd.Series(predicted_labels).value_counts().values
            
            print("\n📊 PREDICTION SUMMARY:")
            print("-" * 50)
            for label, count in zip(pd.Series(predicted_labels).value_counts().index, 
                                   pd.Series(predicted_labels).value_counts().values):
                percentage = (count / len(predicted_labels)) * 100
                print(f"   {label:20s}: {count:5d} packets ({percentage:5.1f}%)")
            
            # Show some example predictions
            print("\n🔍 SAMPLE PREDICTIONS (first 10):")
            print("-" * 50)
            for i in range(min(10, len(predicted_labels))):
                print(f"   Packet {i+1}: {predicted_labels[i]:20s} (confidence: {max_probas[i]:.2%})")
            
            # Flag suspicious packets
            malicious_mask = predicted_labels != 'BENIGN'
            num_malicious = malicious_mask.sum()
            
            if num_malicious > 0:
                print(f"\n⚠️  ALERT: {num_malicious} potentially malicious packets detected!")
                print("\nTop 5 suspicious packets:")
                suspicious_indices = probabilities[malicious_mask].max(axis=1).argsort()[-5:][::-1]
                for idx in suspicious_indices:
                    orig_idx = malicious_mask.nonzero()[0][idx]
                    print(f"   Packet #{orig_idx+1}: {predicted_labels[orig_idx]} "
                          f"(confidence: {max_probas[orig_idx]:.2%})")
            else:
                print("\n✅ All packets classified as BENIGN")
            
            return {
                'total_tested': len(test_packets),
                'predictions': predicted_labels,
                'probabilities': max_probas,
                'summary': dict(zip(pd.Series(predicted_labels).value_counts().index,
                                  pd.Series(predicted_labels).value_counts().values))
            }
            
        except Exception as e:
            print(f"❌ Error testing {pcap_path}: {e}")
            return None

if __name__ == "__main__":
    tester = ModelTester()
    
    # Test on existing attack files
    attack_files = [
        "data/raw/attacks/portscan_attack.pcap",
        "data/raw/attacks/httpflood_attack.pcap",
        "data/raw/attacks/bruteforce_attack.pcap",
        "data/raw/attacks/slowhttp_attack.pcap",
        "data/raw/attacks/dnstunnel_attack.pcap"
    ]
    
    # Test on benign file
    benign_file = "data/raw/benign/baseline_20260215_094524.pcap"
    
    print("\n" + "=" * 50)
    print("🧪 TESTING ON BENIGN TRAFFIC")
    print("=" * 50)
    tester.test_pcap(benign_file, max_packets=500)
    
    print("\n" + "=" * 50)
    print("🧪 TESTING ON ATTACK TRAFFIC")
    print("=" * 50)
    
    for attack_file in attack_files:
        if Path(attack_file).exists():
            tester.test_pcap(attack_file, max_packets=200)
