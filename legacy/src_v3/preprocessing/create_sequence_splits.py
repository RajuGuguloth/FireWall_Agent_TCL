"""
Create train/val/test splits for sequences
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

# Load sequences
X = np.load("data/processed/sequences/sequences.npy")
y = np.load("data/processed/sequences/labels.npy")

print("=" * 50)
print("Creating sequence splits...")
print("=" * 50)
print(f"Total sequences: {len(X)}")

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Split: 70% train, 15% val, 15% test
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# Save
output_dir = Path("data/splits/sequences")
output_dir.mkdir(parents=True, exist_ok=True)

np.save(output_dir / "X_train.npy", X_train)
np.save(output_dir / "X_val.npy", X_val)
np.save(output_dir / "X_test.npy", X_test)
np.save(output_dir / "y_train.npy", y_train)
np.save(output_dir / "y_val.npy", y_val)
np.save(output_dir / "y_test.npy", y_test)

# Save encoder
import joblib
joblib.dump(encoder, "models/serialized/sequence_label_encoder.pkl")

print(f"\n✅ Splits created:")
print(f"   Train: {len(X_train)}")
print(f"   Val:   {len(X_val)}")
print(f"   Test:  {len(X_test)}")
print(f"\nClasses: {list(encoder.classes_)}")
