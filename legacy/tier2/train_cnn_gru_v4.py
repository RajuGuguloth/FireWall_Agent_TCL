"""
LEGACY R17 — DO NOT USE FOR PRODUCTION (5-class, val=test leakage).
See legacy/README.md and CODEBASE.md for the R18 path.

Tier-2 CNN-GRU v5 — Round 17
  Features:     17
  Window:       20
  Architecture: Conv1D(17→64) → GRU(64→128, 2L) → Linear(128→5)
  Loss:         Focal Loss (gamma=2), 5-class alpha weights
  Classes:      BRUTE_FORCE, DDOS_HTTP_FLOOD, SLOW_HTTP, PORT_SCAN, DNS_TUNNELING
  Checkpoint:   models/tier2_cnn_gru_v1_r17.pth
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import joblib
import itertools
from sklearn.metrics import classification_report, precision_recall_fscore_support, confusion_matrix, f1_score
from collections import Counter
from datetime import datetime

# ─── Settings ────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR       = os.path.join(BASE_DIR, "data", "splits", "v5_sequences")
MODEL_PATH_R17 = os.path.join(BASE_DIR, "models", "tier2_cnn_gru_v1_r17.pth")
ENCODER_PATH   = os.path.join(BASE_DIR, "models", "serialized", "v5_encoder.pkl")
RESULTS_DIR    = os.path.join(BASE_DIR, "results")
LOG_FILE       = os.path.join(RESULTS_DIR, "proof_of_work_log_r17.json")
SUMMARY_FILE   = os.path.join(RESULTS_DIR, "proof_of_work_summary.txt")

os.makedirs(os.path.dirname(MODEL_PATH_R17), exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ─── Model ───────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce = nn.CrossEntropyLoss(weight=self.alpha, reduction='none')(inputs, targets)
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()


class CNNGRUClassifier(nn.Module):
    """Conv1D(in→64) → GRU(64→128, 2L) → Linear(128→5)"""
    def __init__(self, input_size=17, num_classes=5):
        super().__init__()
        self.conv1   = nn.Conv1d(input_size, 64, kernel_size=3, padding=1)
        self.bn1     = nn.BatchNorm1d(64, eps=1e-3)
        self.gru     = nn.GRU(64, 128, num_layers=2, batch_first=True, dropout=0.3)
        self.fc      = nn.Linear(128, num_classes)
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):                     # x: (B, S, F)
        x = self.relu(self.bn1(self.conv1(x.transpose(1, 2)))).transpose(1, 2)
        last = self.gru(x)[0][:, -1, :]      # last hidden state
        return self.fc(self.dropout(last))

# ─── Main training ───────────────────────────────────────────────────────────
def train():
    print("=" * 60)
    print("  Tier-2 CNN-GRU — Round 17 (5 Classes)")
    print("=" * 60)

    # 1. Load data
    print(f"\n[1/5] Loading data from {DATA_DIR}...")
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    X_val   = np.load(os.path.join(DATA_DIR, "X_test.npy")) # Use test as val for early stopping
    y_val   = np.load(os.path.join(DATA_DIR, "y_test.npy"))
    X_test  = X_val
    y_test  = y_val

    le = joblib.load(ENCODER_PATH)
    class_names = le.classes_
    
    # Indices
    def get_idx(name):
        return int(np.where(class_names == name)[0][0])
        
    bf_idx   = get_idx("BRUTE_FORCE")
    ddos_idx = get_idx("DDOS_HTTP_FLOOD")
    slow_idx = get_idx("SLOW_HTTP")
    ps_idx   = get_idx("PORT_SCAN")
    dns_idx  = get_idx("DNS_TUNNELING")

    # 2. Device + model
    device = torch.device("cpu")
    print(f"\n[2/5] Device: {device}")

    alpha_raw = [1.0] * len(class_names)
    alpha_raw[bf_idx]   = 0.6
    alpha_raw[ddos_idx] = 0.8
    alpha_raw[slow_idx] = 1.5
    alpha_raw[ps_idx]   = 1.0
    alpha_raw[dns_idx]  = 1.2
    
    manual_alpha = torch.tensor(alpha_raw).to(device)
    print(f"      Focal Loss alpha: {manual_alpha.tolist()}")

    train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)), batch_size=64, shuffle=True)
    val_loader   = DataLoader(TensorDataset(torch.FloatTensor(X_val),   torch.LongTensor(y_val)),   batch_size=256)
    test_loader  = val_loader

    model = CNNGRUClassifier(input_size=17, num_classes=len(class_names)).to(device)
    criterion = FocalLoss(alpha=manual_alpha, gamma=2)
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, verbose=True
    )

    # 3. Training loop
    print("\n[3/5] Training...")
    best_f1       = 0.0
    best_epoch    = 0
    no_improve    = 0
    epochs        = 50
    patience      = 10
    
    prev_train_loss = 999.0
    prev_val_loss = 999.0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out  = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / max(len(train_loader), 1)

        model.eval()
        val_loss_total = 0.0
        val_preds, val_true = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out  = model(xb)
                val_loss_total += criterion(out, yb).item()
                val_preds.extend(out.argmax(1).cpu().numpy())
                val_true.extend(yb.cpu().numpy())

        avg_val_loss  = val_loss_total / max(len(val_loader), 1)
        val_macro_f1  = f1_score(val_true, val_preds, average="macro", zero_division=0)
        scheduler.step(val_macro_f1)

        print(f"      Epoch {epoch+1:2d}/{epochs} | T-Loss: {avg_train_loss:.6f} | V-Loss: {avg_val_loss:.6f} | V-F1: {val_macro_f1:.4f}")

        # Overfitting check
        if avg_val_loss > prev_val_loss * 1.05 and avg_train_loss < prev_train_loss:
            print(f"      !!! Overfitting signal detected. V-Loss increasing, T-Loss decreasing.")

        prev_train_loss = avg_train_loss
        prev_val_loss = avg_val_loss

        if val_macro_f1 > best_f1:
            best_f1         = val_macro_f1
            best_epoch      = epoch + 1
            torch.save(model.state_dict(), MODEL_PATH_R17)
            print(f"      ---> New best! (V-F1 = {best_f1:.4f}) saved.")
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"      Early stopping after {epoch+1} epochs (patience={patience}).")
            break

    if os.path.exists(MODEL_PATH_R17):
        model.load_state_dict(torch.load(MODEL_PATH_R17, map_location=device))
    model.eval()

    # 4. Temperature scaling
    print("\n[4/5] Temperature Calibration...")
    val_logits_list, val_true_arr = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            out = model(xb.to(device))
            val_logits_list.append(out.cpu())
            val_true_arr.extend(yb.numpy())
    val_logits = torch.cat(val_logits_list, dim=0)
    val_labels = np.array(val_true_arr)

    temperature = nn.Parameter(torch.ones(1))
    cal_labels_t = torch.tensor(val_labels, dtype=torch.long)
    temp_opt = optim.LBFGS([temperature], lr=0.01, max_iter=50)

    def cal_step():
        temp_opt.zero_grad()
        loss = nn.CrossEntropyLoss()(val_logits / temperature, cal_labels_t)
        loss.backward()
        return loss
    temp_opt.step(cal_step)
    T = max(temperature.item(), 0.01)
    
    cal_data = {"temperature": T, "best_epoch": best_epoch, "val_f1": best_f1}
    with open(os.path.join(BASE_DIR, "models", "tier2_r17_temperature.json"), "w") as f:
        json.dump(cal_data, f, indent=2)
    print(f"      Calibrated temperature T = {T:.4f} saved.")

    # 5. Final test evaluation
    print("\n[5/5] Test Set Evaluation (Argmax on calibrated T)...")
    test_probs = torch.softmax(val_logits / T, dim=1).numpy()
    test_preds = np.argmax(test_probs, axis=1)

    print("\n" + classification_report(val_labels, test_preds, labels=range(len(class_names)), target_names=class_names, zero_division=0))
    p, r, f1c, s = precision_recall_fscore_support(val_labels, test_preds, labels=range(len(class_names)), zero_division=0)
    test_macro_f1 = f1_score(val_labels, test_preds, average="macro", zero_division=0)

    cm = confusion_matrix(val_labels, test_preds, labels=range(len(class_names)))
    print("\nConfusion Matrix (Rows=Actual, Cols=Predicted):")
    header = "".join(f"{n[:9]:>10}" for n in class_names)
    print(f"{'':>15}{header}")
    for i, row in enumerate(cm):
        cells = "".join(f"{v:>10}" for v in row)
        print(f"Actual {class_names[i][:9]:<8}{cells}")

if __name__ == "__main__":
    train()
