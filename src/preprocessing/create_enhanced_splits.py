"""
Create train/val/test splits for enhanced features
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from pathlib import Path
import joblib

def create_enhanced_splits():
    print("=" * 50)
    print("Creating Enhanced Splits...")
    print("=" * 50)

    df = pd.read_csv("data/processed/features/enhanced_features.csv")
    print(f"Total samples: {len(df)}")

    feature_cols = [
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

    X = df[feature_cols].values
    y = df['attack_type'].values

    # Encode labels
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_scaled, y_encoded,
        test_size=0.3,
        random_state=42,
        stratify=y_encoded
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.5,
        random_state=42,
        stratify=y_temp
    )

    # Save splits
    splits_dir = Path("data/splits/enhanced")
    splits_dir.mkdir(parents=True, exist_ok=True)

    np.save(splits_dir / "X_train.npy", X_train)
    np.save(splits_dir / "X_val.npy", X_val)
    np.save(splits_dir / "X_test.npy", X_test)
    np.save(splits_dir / "y_train.npy", y_train)
    np.save(splits_dir / "y_val.npy", y_val)
    np.save(splits_dir / "y_test.npy", y_test)

    # Save scaler and encoder
    joblib.dump(scaler, "models/serialized/feature_scaler.pkl")
    joblib.dump(encoder, "models/serialized/enhanced_label_encoder.pkl")

    print(f"\nSplits created:")
    print(f"  Train: {len(X_train)}")
    print(f"  Val:   {len(X_val)}")
    print(f"  Test:  {len(X_test)}")
    print(f"\nClasses: {list(encoder.classes_)}")
    print(f"Features: {len(feature_cols)}")

    return X_train, X_val, X_test, y_train, y_val, y_test, encoder

if __name__ == "__main__":
    create_enhanced_splits()
