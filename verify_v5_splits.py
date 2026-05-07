"""
Step 1: Verify Splits
Loads data/splits/v5_sequences/ .npy files and confirms shapes, classes, and lack of NaNs.
"""
import numpy as np
import os
import joblib

SPLIT_DIR = "data/splits/v5_sequences"
ENCODER_PATH = "models/serialized/v5_encoder.pkl"

def verify():
    print("=" * 60)
    print("  STEP 1 — VERIFY SPLITS")
    print("=" * 60)

    try:
        X_train = np.load(os.path.join(SPLIT_DIR, "X_train.npy"))
        X_test  = np.load(os.path.join(SPLIT_DIR, "X_test.npy"))
        y_train = np.load(os.path.join(SPLIT_DIR, "y_train.npy"))
        y_test  = np.load(os.path.join(SPLIT_DIR, "y_test.npy"))
        le = joblib.load(ENCODER_PATH)
        label_map = dict(zip(le.transform(le.classes_), le.classes_))
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    print("Shapes:")
    print(f"  X_train: {X_train.shape}  — Expected: (N, 20, 17)")
    print(f"  X_test : {X_test.shape}  — Expected: (M, 20, 17)")
    print(f"  y_train: {y_train.shape}      — Expected: (N,)")
    print(f"  y_test : {y_test.shape}      — Expected: (M,)")
    
    print("\nNaN / Inf Check:")
    print(f"  X_train has NaN: {np.isnan(X_train).any()}")
    print(f"  X_train has Inf: {np.isinf(X_train).any()}")
    print(f"  X_test  has NaN: {np.isnan(X_test).any()}")
    print(f"  X_test  has Inf: {np.isinf(X_test).any()}")
    
    print("\nClass Distribution (y_train):")
    train_counts = np.bincount(y_train)
    for idx, count in enumerate(train_counts):
        if count > 0:
            print(f"  {idx} = {label_map.get(idx, 'UNKNOWN'):<20}: {count:,}")
            
    print("\nClass Distribution (y_test):")
    test_counts = np.bincount(y_test)
    for idx, count in enumerate(test_counts):
        if count > 0:
            print(f"  {idx} = {label_map.get(idx, 'UNKNOWN'):<20}: {count:,}")

    print("=" * 60)

if __name__ == "__main__":
    verify()
