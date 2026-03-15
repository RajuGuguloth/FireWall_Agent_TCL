"""
Extract packet sequences for Tier-2 LSTM model
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json

class SequenceExtractor:
    def __init__(self, sequence_length=10):
        self.sequence_length = sequence_length
        
    def create_sequences(self, features_csv="data/processed/features/all_packets.csv",
                        output_dir="data/processed/sequences"):
        """Create sequences from packet features"""
        print("=" * 50)
        print("Creating packet sequences...")
        print("=" * 50)
        
        # Load features
        df = pd.read_csv(features_csv)
        print(f"Total packets: {len(df)}")
        
        # Group by attack type
        sequences = []
        labels = []
        
        for attack_type in df['attack_type'].unique():
            attack_df = df[df['attack_type'] == attack_type]
            print(f"\nProcessing {attack_type}: {len(attack_df)} packets")
            
            # Create sliding windows
            for i in range(len(attack_df) - self.sequence_length + 1):
                seq = attack_df.iloc[i:i+self.sequence_length]
                
                # Extract numeric features only
                feature_cols = ['packet_length', 'has_ip', 'has_tcp', 'has_udp', 
                               'has_icmp', 'ip_version', 'ip_ttl', 'ip_proto', 
                               'src_port', 'dst_port', 'tcp_flags']
                
                seq_features = seq[feature_cols].values
                sequences.append(seq_features)
                labels.append(attack_type)
        
        # Convert to numpy arrays
        X = np.array(sequences)
        y = np.array(labels)
        
        print(f"\n✅ Created {len(X)} sequences")
        print(f"   Sequence shape: {X.shape}")
        
        # Save
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        np.save(f"{output_dir}/sequences.npy", X)
        np.save(f"{output_dir}/labels.npy", y)
        
        print(f"   Saved to {output_dir}/")
        
        return X, y

if __name__ == "__main__":
    extractor = SequenceExtractor(sequence_length=10)
    X, y = extractor.create_sequences()
