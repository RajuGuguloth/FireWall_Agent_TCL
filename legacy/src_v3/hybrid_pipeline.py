"""
Hybrid Tier-1 (RF) + Tier-2 (CNN+GRU v3) Pipeline
Tier-1: Random Forest — fast first-pass on every packet (11 or 25 features)
Tier-2: CNN+GRU v3 — deep analysis on suspicious sequences (25 features, seq_len=10)
"""
import sys
import torch
import torch.nn as nn
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# ──────────────────────────────────────────────────────────
# CNN+GRU Classifier  (copy-pasted from train_cnn_gru_v3.py
#  so this file is self-contained; no import from training/)
# ──────────────────────────────────────────────────────────
class CNNGRUClassifier(nn.Module):
    def __init__(self, input_size=25, hidden_size=128,
                 num_classes=6, sequence_length=10):
        super(CNNGRUClassifier, self).__init__()
        self.conv1      = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv2      = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.relu       = nn.ReLU()
        self.dropout    = nn.Dropout(0.3)
        self.batch_norm1 = nn.BatchNorm1d(32)
        self.batch_norm2 = nn.BatchNorm1d(64)
        self.gru = nn.GRU(
            input_size=64, hidden_size=hidden_size,
            num_layers=2, batch_first=True,
            dropout=0.2, bidirectional=True
        )
        self.attention = nn.Linear(hidden_size * 2, 1)
        self.fc1       = nn.Linear(hidden_size * 2, 64)
        self.fc2       = nn.Linear(64, num_classes)

    def attention_weights(self, gru_output):
        weights = torch.softmax(self.attention(gru_output), dim=1)
        return (gru_output * weights).sum(dim=1)

    def forward(self, x):
        batch_size = x.size(0)
        x_cnn = x.view(batch_size * x.size(1), 1, -1)
        x_cnn = self.relu(self.batch_norm1(self.conv1(x_cnn)))
        x_cnn = self.relu(self.batch_norm2(self.conv2(x_cnn)))
        x_cnn = x_cnn.mean(dim=2)
        x_gru = x_cnn.view(batch_size, -1, 64)
        gru_out, _ = self.gru(x_gru)
        attended   = self.attention_weights(gru_out)
        out        = self.relu(self.fc1(attended))
        out        = self.dropout(out)
        return self.fc2(out)


# ──────────────────────────────────────────────────────────
# Hybrid Firewall
# ──────────────────────────────────────────────────────────
class HybridFirewall:
    CHECKPOINT = "models/serialized/tier2_cnn_gru.pth"
    RF_MODEL   = "models/serialized/tier1_rf.pkl"
    RF_ENCODER = "models/serialized/label_encoder.pkl"
    ENH_ENCODER = "models/serialized/enhanced_label_encoder.pkl"

    def __init__(self):
        print("=" * 55)
        print("Loading Hybrid Firewall  (RF + CNN+GRU v3)")
        print("=" * 55)

        # ── Tier-1: Random Forest ──────────────────────────
        self.rf_model   = joblib.load(self.RF_MODEL)
        self.rf_encoder = joblib.load(self.RF_ENCODER)
        print(f"✅ Tier-1 RF loaded   | classes: {list(self.rf_encoder.classes_)}")

        # ── Tier-2: CNN+GRU v3 ────────────────────────────
        self.enh_encoder = joblib.load(self.ENH_ENCODER)
        num_classes      = len(self.enh_encoder.classes_)

        checkpoint = torch.load(self.CHECKPOINT, map_location="cpu")

        # Read saved hyper-parameters so the architecture always matches
        input_size      = checkpoint.get("input_size", 25)
        sequence_length = checkpoint.get("sequence_length", 10)

        self.cnn_gru = CNNGRUClassifier(
            input_size=input_size,
            hidden_size=128,
            num_classes=num_classes,
            sequence_length=sequence_length
        )
        self.cnn_gru.load_state_dict(checkpoint["model_state_dict"])
        self.cnn_gru.eval()

        self.seq_len = sequence_length
        self.device  = torch.device("cpu")
        self.cnn_gru.to(self.device)

        print(f"✅ Tier-2 CNN+GRU loaded | input={input_size} features"
              f" | seq_len={sequence_length}"
              f" | classes: {list(self.enh_encoder.classes_)}")

    # ── Tier-1 inference ──────────────────────────────────
    def tier1_classify(self, features: np.ndarray):
        """features: 1-D array of 11 basic RF features."""
        proba      = self.rf_model.predict_proba([features])[0]
        pred_idx   = proba.argmax()
        confidence = proba[pred_idx]
        label      = self.rf_encoder.inverse_transform([pred_idx])[0]
        return label, float(confidence)

    # ── Tier-2 inference ──────────────────────────────────
    def tier2_classify(self, sequence: np.ndarray):
        """sequence: (seq_len, 25) array of enhanced features."""
        if sequence.shape[0] != self.seq_len:
            raise ValueError(
                f"sequence must have {self.seq_len} timesteps, "
                f"got {sequence.shape[0]}"
            )
        seq_t = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.cnn_gru(seq_t)
            proba  = torch.softmax(logits, dim=1)
            conf, pred = torch.max(proba, 1)
        label = self.enh_encoder.inverse_transform([pred.item()])[0]
        return label, float(conf.item())

    # ── Hybrid decision ───────────────────────────────────
    def classify(self, rf_features: np.ndarray,
                 sequence: np.ndarray = None,
                 use_tier2: bool = True,
                 tier1_threshold: float = 0.95):
        """
        rf_features : 1-D array (11 features) for Tier-1 RF
        sequence    : (seq_len, 25) array for Tier-2 CNN+GRU
        tier1_threshold : skip Tier-2 if Tier-1 says BENIGN with this confidence
        """
        tier1_label, tier1_conf = self.tier1_classify(rf_features)

        # Fast-pass: high-confidence BENIGN → no need for deep model
        if tier1_label == "BENIGN" and tier1_conf >= tier1_threshold:
            return {
                "verdict":          "SAFE",
                "tier1_label":      tier1_label,
                "tier1_confidence": tier1_conf,
                "tier2_used":       False,
            }

        # Deep pass: suspicious packet → Tier-2
        if use_tier2 and sequence is not None:
            tier2_label, tier2_conf = self.tier2_classify(sequence)
            return {
                "verdict":          "MALICIOUS" if tier2_label != "BENIGN" else "SAFE",
                "attack_type":      tier2_label,
                "tier1_label":      tier1_label,
                "tier1_confidence": tier1_conf,
                "tier2_label":      tier2_label,
                "tier2_confidence": tier2_conf,
                "tier2_used":       True,
            }

        # Tier-2 unavailable → fall back to Tier-1 verdict
        return {
            "verdict":          "MALICIOUS" if tier1_label != "BENIGN" else "SAFE",
            "attack_type":      tier1_label,
            "tier1_label":      tier1_label,
            "tier1_confidence": tier1_conf,
            "tier2_used":       False,
        }


# ──────────────────────────────────────────────────────────
# Quick smoke-test using saved .npy splits
# ──────────────────────────────────────────────────────────
def test_hybrid():
    import json
    from sklearn.metrics import accuracy_score

    print("\n" + "=" * 55)
    print("Testing Hybrid Pipeline (RF + CNN+GRU v3)")
    print("=" * 55)

    fw = HybridFirewall()

    SEQ_LEN = fw.seq_len

    # ── 1. Tier-1 accuracy on its own CSV test set (unscaled 11 features) ──
    rf_df = pd.read_csv("data/splits/test/features.csv")
    with open("models/serialized/feature_names.json") as _f:
        rf_cols = json.load(_f)
    X_rf   = rf_df[rf_cols].values
    y_rf   = rf_df["attack_type"].values   # string labels

    rf_preds = fw.rf_model.predict(X_rf)
    rf_labels = fw.rf_encoder.inverse_transform(rf_preds)
    t1_acc = accuracy_score(y_rf, rf_labels)
    print(f"\n✅ Tier-1 (RF) Accuracy on CSV test set : {t1_acc:.2%}  ({len(X_rf)} samples)")

    # ── 2. Tier-2 accuracy on its own enhanced .npy test set (scaled 25 features) ──
    X_enh = np.load("data/splits/enhanced/X_test.npy")
    y_enh = np.load("data/splits/enhanced/y_test.npy")

    n_seq  = len(X_enh) - SEQ_LEN + 1
    t2_correct = 0
    for i in range(n_seq):
        seq       = X_enh[i: i + SEQ_LEN]
        true_idx  = y_enh[i + SEQ_LEN - 1]
        t2_label, _ = fw.tier2_classify(seq)
        if t2_label == fw.enh_encoder.classes_[true_idx]:
            t2_correct += 1
    t2_acc = t2_correct / n_seq
    print(f"✅ Tier-2 (CNN+GRU) Accuracy on .npy test  : {t2_acc:.2%}  ({n_seq} sequences)")

    # ── 3. Sample hybrid predictions (first 5) using .npy data for both tiers ──
    print(f"\n{'─'*55}")
    print("Sample Hybrid Predictions (Tier-1 fed scaled cols 0-10):")
    print(f"{'─'*55}")
    for i in range(min(5, n_seq)):
        seq        = X_enh[i: i + SEQ_LEN]          # (10, 25) scaled
        rf_feats   = X_enh[i + SEQ_LEN - 1, :11]    # first 11 scaled cols (demo only)
        true_label = fw.enh_encoder.classes_[y_enh[i + SEQ_LEN - 1]]

        result = fw.classify(rf_feats, seq, use_tier2=True)
        print(f"\n  Sample {i+1}: true={true_label}")
        print(f"    Tier-1 → {result['tier1_label']} ({result['tier1_confidence']:.2%})")
        if result["tier2_used"]:
            print(f"    Tier-2 → {result['tier2_label']} ({result['tier2_confidence']:.2%})")
        print(f"    Final  → {result['verdict']}")

    print("\n" + "=" * 55)
    print("Pipeline End-to-End Test Complete ✅")
    print("=" * 55)


if __name__ == "__main__":
    test_hybrid()
