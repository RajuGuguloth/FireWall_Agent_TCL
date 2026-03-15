"""
Tier-2 CNN-GRU v4 — Round 14
  Features:     23 (leaked features removed)
  Window:       20
  Architecture: Conv1D(23→64) → GRU(64→128, 2L) → Linear(128→3)
  Loss:         Focal Loss, alpha=[0.6, 0.8, 1.5]
  Scheduler:    ReduceLROnPlateau(mode='max') on val_macro_f1
  Calibration:  Temperature Scaling on val set before threshold tuning
  Threshold:    Coarse-to-fine search per class on val set
  Checkpoint:   models/tier2_cnn_gru_v1_r14.pth (keep r13)
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pickle
import itertools
from sklearn.metrics import classification_report, precision_recall_fscore_support, confusion_matrix, f1_score
from collections import Counter
from datetime import datetime

# ─── Settings ────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR       = os.path.join(BASE_DIR, "data", "splits", "v4_sequences_hard_subset")
MODEL_PATH_R16 = os.path.join(BASE_DIR, "models", "tier2_cnn_gru_v1_r16.pth")
ENCODER_PATH   = os.path.join(DATA_DIR, "label_encoder.pkl")
STATS_PATH     = os.path.join(DATA_DIR, "dataset_stats.json")
RESULTS_DIR    = os.path.join(BASE_DIR, "results")
LOG_FILE       = os.path.join(RESULTS_DIR, "proof_of_work_log.json")
SUMMARY_FILE   = os.path.join(RESULTS_DIR, "proof_of_work_summary.txt")

R16_HONEST_BASELINE = None  # no valid prior baseline exists

os.makedirs(os.path.dirname(MODEL_PATH_R16), exist_ok=True)
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
    """Conv1D(in→64) → GRU(64→128, 2L) → Linear(128→num_classes)"""
    def __init__(self, input_size, num_classes=3):
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

    def extract_features(self, x):
        """Return 128-dim GRU hidden state (for GNN edge features)."""
        with torch.no_grad():
            x = self.relu(self.bn1(self.conv1(x.transpose(1, 2)))).transpose(1, 2)
            return self.gru(x)[0][:, -1, :]

# ─── Helpers ─────────────────────────────────────────────────────────────────
def append_json_log(entry):
    logs = []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            try: logs = json.load(f)
            except: logs = []
    logs.append(entry)
    with open(LOG_FILE, "w") as f:
        json.dump(logs, f, indent=2)

def write_summary(text):
    print(text)
    with open(SUMMARY_FILE, "a") as f:
        f.write(text + "\n")

# ─── Main training ───────────────────────────────────────────────────────────
def train():
    print("=" * 60)
    print("  Tier-2 CNN-GRU — Round 16")
    print("=" * 60)

    # 1. Load data
    print(f"\n[1/5] Loading data from {DATA_DIR}...")
    X_train = np.load(os.path.join(DATA_DIR, "train_sequences.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "train_labels.npy"))
    X_val   = np.load(os.path.join(DATA_DIR, "val_sequences.npy"))
    y_val   = np.load(os.path.join(DATA_DIR, "val_labels.npy"))
    X_test  = np.load(os.path.join(DATA_DIR, "test_sequences.npy"))
    y_test  = np.load(os.path.join(DATA_DIR, "test_labels.npy"))

    with open(ENCODER_PATH, "rb") as f:
        le = pickle.load(f)
    with open(STATS_PATH, "r") as f:
        stats = json.load(f)
        
    class_names = le.classes_
    name = {i: n for i, n in enumerate(class_names)}
    bf_idx   = int(np.where(class_names == "BRUTE_FORCE")[0][0])
    ddos_idx = int(np.where(class_names == "DDOS_HTTP_FLOOD")[0][0])
    slow_idx = int(np.where(class_names == "SLOW_HTTP")[0][0])

    print(f"Classes in test: {{{', '.join(stats.get('classes_in_test', []))}}}")
    print(f"Feature count: {stats.get('feature_count', 17)}")
    print(f"Total sequences: train={len(X_train)} val={len(X_val)} test={len(X_test)}")
    
    test_freq = Counter(y_test)
    slow_support = test_freq[slow_idx]
    print(f"SLOW_HTTP test sequences: {slow_support}")
    
    print(f"      Train distribution: {dict(Counter(y_train))}")

    # 2. Device + model
    device = torch.device("cpu")
    print(f"\n[2/5] Device: {device}")

    n_features = X_train.shape[2]
    print(f"      Feature dimension: {n_features}")
    assert n_features == 17, f"ERROR: Expected 17 features, got {n_features}"

    alpha_raw = [1.0] * len(class_names)
    alpha_raw[bf_idx]   = 0.6
    alpha_raw[ddos_idx] = 0.8
    alpha_raw[slow_idx] = 1.5
    manual_alpha = torch.tensor(alpha_raw).to(device)
    print(f"      Focal Loss alpha: {manual_alpha.tolist()}")

    train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)), batch_size=64, shuffle=True)
    val_loader   = DataLoader(TensorDataset(torch.FloatTensor(X_val),   torch.LongTensor(y_val)),   batch_size=256)
    test_loader  = DataLoader(TensorDataset(torch.FloatTensor(X_test),  torch.LongTensor(y_test)),  batch_size=256)

    model = CNNGRUClassifier(input_size=17, num_classes=len(class_names)).to(device)

    FRESH_START = True
    print("      FRESH_START = True — Training from random initialization (no warm-start).")

    criterion = FocalLoss(alpha=manual_alpha, gamma=2)
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
    # LR Scheduler: maximize val macro F1
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, verbose=True
    )

    # 3. Training loop
    print("\n[3/5] Training...")
    best_f1       = 0.0
    best_epoch    = 0
    best_train_loss = 0.0
    best_val_loss   = 0.0
    no_improve    = 0
    epochs        = 50
    patience      = 7

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out  = model(xb)
            loss = criterion(out, yb)
            if torch.isnan(loss):
                print(f"      !!! NaN loss at epoch {epoch+1} — reducing LR")
                optimizer.param_groups[0]["lr"] = 1e-5
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / max(len(train_loader), 1)

        # Validation
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

        if val_macro_f1 > best_f1:
            best_f1         = val_macro_f1
            best_epoch      = epoch + 1
            best_train_loss = avg_train_loss
            best_val_loss   = avg_val_loss
            torch.save(model.state_dict(), MODEL_PATH_R16)
            print(f"      ---> New best! (V-F1 = {best_f1:.4f}) saved to {MODEL_PATH_R16}")
            no_improve = 0
        else:
            no_improve += 1

        # Periodic checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            epoch_path = MODEL_PATH_R16.replace(".pth", f"_epoch{epoch+1}.pth")
            torch.save(model.state_dict(), epoch_path)

        if no_improve >= patience:
            print(f"      Early stopping after {epoch+1} epochs (patience={patience}).")
            break

    # Load best weights for evaluation
    if os.path.exists(MODEL_PATH_R16):
        model.load_state_dict(torch.load(MODEL_PATH_R16, map_location=device))
    model.eval()

    # 4. Temperature scaling + threshold tuning
    print("\n[4/5] Temperature Scaling + Threshold Tuning (on val set)...")

    # Collect raw val logits
    val_logits_list, val_true_arr = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            out = model(xb.to(device))
            val_logits_list.append(out.cpu())
            val_true_arr.extend(yb.numpy())
    val_logits = torch.cat(val_logits_list, dim=0)
    val_true_arr = np.array(val_true_arr)

    # Split val set for independent tuning (Issue 11 fix)
    cal_size = len(val_true_arr) // 2
    cal_logits  = val_logits[:cal_size]    # for temperature
    cal_labels  = val_true_arr[:cal_size]
    thr_logits  = val_logits[cal_size:]    # for threshold tuning
    thr_labels  = val_true_arr[cal_size:]

    # Learn temperature on calibration set
    temperature = nn.Parameter(torch.ones(1))
    cal_labels_t = torch.tensor(cal_labels, dtype=torch.long)
    temp_opt = optim.LBFGS([temperature], lr=0.01, max_iter=50)

    def cal_step():
        temp_opt.zero_grad()
        loss = nn.CrossEntropyLoss()(cal_logits / temperature, cal_labels_t)
        loss.backward()
        return loss
    temp_opt.step(cal_step)
    T = max(temperature.item(), 0.01)    # guard against <=0
    print(f"      Calibrated temperature T = {T:.4f}")

    thr_probs = torch.softmax(thr_logits / T, dim=1).numpy()

    # Correct threshold logic: predict class i if prob[i] > threshold[i]
    # Fall back to argmax if no class exceeds its threshold (Issue 9 fix)
    def apply_thresholds(probs, thresholds):
        preds = []
        for row in probs:
            above = [i for i in range(len(row)) if row[i] > thresholds[i]]
            if above:
                preds.append(max(above, key=lambda i: row[i]))
            else:
                preds.append(int(np.argmax(row)))
        return np.array(preds)

    # Coarse search (5³ = 125 combos)
    coarse = [0.30, 0.40, 0.50, 0.60, 0.70]
    best_coarse, best_sum = [0.5, 0.5, 0.5], 0.0
    for t0, t1, t2 in itertools.product(coarse, coarse, coarse):
        preds = apply_thresholds(thr_probs, [t0, t1, t2])
        _, _, f, _ = precision_recall_fscore_support(thr_labels, preds, zero_division=0)
        if sum(f) > best_sum:
            best_sum, best_coarse = sum(f), [t0, t1, t2]
    print(f"      Coarse best: {best_coarse}")

    # Fine search (±0.05 in 0.01 steps → ~27 combos)
    best_t, best_fsum = list(best_coarse), best_sum
    fine_grids = [
        np.arange(max(0.01, t - 0.05), min(1.0, t + 0.05) + 0.001, 0.01).tolist()
        for t in best_coarse
    ]
    for t0, t1, t2 in itertools.product(*fine_grids):
        preds = apply_thresholds(thr_probs, [t0, t1, t2])
        _, _, f, _ = precision_recall_fscore_support(thr_labels, preds, zero_division=0)
        if sum(f) > best_fsum:
            best_fsum, best_t = sum(f), [t0, t1, t2]
    best_t = [round(t, 2) for t in best_t]
    print(f"      Fine best:   BF={best_t[bf_idx]}  DDOS={best_t[ddos_idx]}  SLOW={best_t[slow_idx]}")

    # 5. Final test evaluation with calibrated probs + tuned thresholds
    print("\n[5/5] Test Set Evaluation...")
    test_logits_list, test_true_arr = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            out = model(xb.to(device))
            test_logits_list.append(out.cpu())
            test_true_arr.extend(yb.numpy())
    test_logits   = torch.cat(test_logits_list, dim=0)
    test_true_arr = np.array(test_true_arr)
    test_probs    = torch.softmax(test_logits / T, dim=1).numpy()

    test_preds = apply_thresholds(test_probs, best_t)

    print("\n" + classification_report(test_true_arr, test_preds, labels=[0, 1, 2], target_names=class_names, zero_division=0))
    p, r, f1c, s = precision_recall_fscore_support(test_true_arr, test_preds, labels=[0, 1, 2], zero_division=0)
    test_macro_f1 = f1_score(test_true_arr, test_preds, average="macro", zero_division=0)

    cm = confusion_matrix(test_true_arr, test_preds, labels=[0, 1, 2])
    print("\nConfusion Matrix (Rows=Actual, Cols=Predicted):")
    header = "".join(f"{class_names[i]:>18}" for i in range(len(class_names)))
    print(f"{'':>22}{header}")
    for i, row in enumerate(cm):
        cells = "".join(f"{v:>18}" for v in row)
        print(f"Actual {class_names[i]:<15}{cells}")

    slow_support = int((test_true_arr == slow_idx).sum())
    print(f"\n      SLOW_HTTP test samples (real): {slow_support}")
    if slow_support < 50:
        print("      ⚠️  CRITICAL WARNING: SLOW_HTTP test < 50. Metrics unreliable.")
    elif slow_support < 100:
        print("      ⚠️  WARNING: SLOW_HTTP test < 100.")

    print(f"\n      Thresholds used: BF={best_t[bf_idx]}  DDOS={best_t[ddos_idx]}  SLOW={best_t[slow_idx]}")
    print(f"      Temperature T: {T:.4f}")

    print("\n      R16 is first valid result — no prior baseline")

    # ─── Proof of Work ───────────────────────────────────────
    pow_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "round": 16,
        "first_valid_result": True,
        "feature_count": 17,
        "zero_variance_cols_removed": [
            "ip_header_length","packet_direction","ip_version",
            "flow_std_pkt_len","flow_ack_ratio","Unnamed: 0"
        ],
        "additional_leaks_removed": [
            "is_syn (corr=0.81 BF)",
            "tcp_window_size (corr=0.81 SLOW)",
            "flow_mean_entropy (corr=0.80 SLOW)"
        ],
        "grouping": "time_window_300",
        "viable_groups": stats.get('viable_groups', 53),
        "classes_in_test": stats.get('classes_in_test', []),
        "random_seed_used": stats.get('random_seed_used', 42),
        "fresh_start": True,
        "prior_rounds_valid": False,
        "prior_rounds_note": "R12-R15 invalid due to data bugs",
        "script": "train_cnn_gru_v4.py",
        "epoch_best": best_epoch,
        "train_loss": round(best_train_loss, 4),
        "val_loss": round(best_val_loss, 4),
        "val_f1_macro": round(best_f1, 4),
        "test_f1_macro": round(test_macro_f1, 4),
        "temperature": round(T, 4),
        "thresholds_used": {
            class_names[bf_idx]:   best_t[bf_idx],
            class_names[ddos_idx]: best_t[ddos_idx],
            class_names[slow_idx]: best_t[slow_idx],
        },
        "per_class_metrics": {
            class_names[i]: {
                "precision": round(float(p[i]), 4),
                "recall":    round(float(r[i]), 4),
                "f1":        round(float(f1c[i]), 4),
                "support":   int(s[i]),
            }
            for i in range(len(class_names))
        },
        "confusion_matrix": cm.tolist(),
        "slow_http_test_samples": slow_support,
    }
    logs = []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            try: logs = json.load(f)
            except: logs = []
    logs.append(pow_entry)
    with open(LOG_FILE, "w") as f:
        json.dump(logs, f, indent=2)

    with open(SUMMARY_FILE, "a") as f:
        f.write("\n──────────────────────────────────────────────────\n")
        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ROUND 16 — train_cnn_gru_v4.py\n")
        f.write(f"Val F1 Macro:           {best_f1:.4f}\n")
        f.write(f"Test F1 Macro:          {test_macro_f1:.4f}\n")
        f.write(f"BRUTE_FORCE Precision:  {p[bf_idx]:.4f}\n")
        f.write(f"DDOS Recall:            {r[ddos_idx]:.4f}\n")
        f.write(f"SLOW_HTTP Precision:    {p[slow_idx]:.4f}\n")
        f.write(f"SLOW_HTTP test samples: {slow_support}\n")
        f.write(f"Thresholds: BF={best_t[bf_idx]}  DDOS={best_t[ddos_idx]}  SLOW={best_t[slow_idx]}\n")
        f.write(f"Temperature T: {T:.4f}\n")
        f.write("R16 First Valid Evaluation Completed.\n")
        f.write("──────────────────────────────────────────────────\n")

    print(f"\nModel saved to: {MODEL_PATH_R16}")
    print("Round 16 complete.")

if __name__ == "__main__":
    train()
