"""
Tier-1 Random Forest Classifier
Target: attack_type (multi-class)
Features: 25 columns (all except attack_type and label)
Binary label column is dropped; attack_type is the training target.
"""

import os
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, "combined_dataset_v3.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "tier1_rf_v3.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
print(f"[1/5] Loading dataset: {DATA_PATH}")
t0 = time.time()
df = pd.read_csv(DATA_PATH)
print(f"      Loaded {len(df):,} rows × {len(df.columns)} columns  ({time.time()-t0:.1f}s)")

# ── Sanity check ──────────────────────────────────────────────────────────────
required_cols = {"attack_type", "label"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing expected columns: {missing}")

print(f"\n[2/5] Class distribution (attack_type):")
print(df["attack_type"].value_counts().to_string())

# ── Feature / target split ────────────────────────────────────────────────────
TARGET   = "attack_type"
DROP_COLS = [TARGET, "label"]             # drop target + binary label
FEATURES  = [c for c in df.columns if c not in DROP_COLS]

assert len(FEATURES) == 25, f"Expected 25 features, got {len(FEATURES)}: {FEATURES}"

X = df[FEATURES].values
y = df[TARGET].values

print(f"\n[3/5] Features ({len(FEATURES)}): {FEATURES}")

# ── Train / test split ────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"\n[4/5] Split → train: {len(X_train):,}  test: {len(X_test):,}")

# ── Train Random Forest ───────────────────────────────────────────────────────
print("\n[5/5] Training Random Forest (class_weight='balanced') …")
t1 = time.time()

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,          # fully grown trees
    min_samples_split=5,
    min_samples_leaf=2,
    max_features="sqrt",
    class_weight="balanced",
    n_jobs=-1,
    random_state=42,
    verbose=1,
)
rf.fit(X_train, y_train)
train_time = time.time() - t1
print(f"      Training complete in {train_time:.1f}s")

# ── Evaluate ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("EVALUATION RESULTS")
print("=" * 60)

y_pred = rf.predict(X_test)
labels = sorted(rf.classes_)

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
recall    = recall_score(y_test, y_pred, average="weighted", zero_division=0)

print(f"\nAccuracy  : {accuracy:.4f}  ({accuracy*100:.2f}%)")
print(f"Precision : {precision:.4f}  (weighted avg)")
print(f"Recall    : {recall:.4f}  (weighted avg)")

print("\n── Confusion Matrix (rows=actual, cols=predicted) ──")
cm = confusion_matrix(y_test, y_pred, labels=labels)
cm_df = pd.DataFrame(cm, index=labels, columns=labels)
print(cm_df.to_string())

print("\n── Per-Class Classification Report ──")
print(classification_report(y_test, y_pred, target_names=labels, zero_division=0))

# ── Feature importance ────────────────────────────────────────────────────────
print("\n── Top-10 Feature Importances ──")
importances = pd.Series(rf.feature_importances_, index=FEATURES)
print(importances.sort_values(ascending=False).head(10).to_string())

# ── Save model ────────────────────────────────────────────────────────────────
joblib.dump(rf, MODEL_PATH)
print(f"\nModel saved → {MODEL_PATH}")
print("Done.")
