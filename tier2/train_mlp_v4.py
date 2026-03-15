"""
Tier-2 MLP v1 - Flow Features Only (Round 8)
- Features: 7 (flow-level statistics)
- Architecture: Simple 3-layer MLP
- Balancing: Class Weights
- Saves: models/tier2_mlp_v1.pth
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import joblib
from sklearn.metrics import classification_report
from collections import Counter

# ─── Settings ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "splits", "v4_flow_only_subset")
MODEL_PATH = os.path.join(BASE_DIR, "models", "tier2_mlp_v1.pth")
ENCODER_PATH = os.path.join(BASE_DIR, "models", "serialized", "hard_subset_encoder_v8.pkl")

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# ─── MLP Architecture ──────────────────────────────────────────────────────────

class SimpleMLP(nn.Module):
    def __init__(self, input_size=7, num_classes=3):
        super(SimpleMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# ─── Training logic ──────────────────────────────────────────────────────────

def train():
    print("=" * 60)
    print("Tier-2 Simple MLP v1 Training (Flow Features - Round 8)")
    print("=" * 60)

    # Load data
    print(f"\n[1/4] Loading data from {DATA_DIR}...")
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))
    
    le = joblib.load(ENCODER_PATH)
    class_names = le.classes_
    
    print(f"      Train shape: {X_train.shape}, Distribution: {dict(Counter(y_train))}")

    # ─── Prepare Loaders ──────────────────────────────────────────────────────
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n[2/4] Using device: {device}")

    # Calculate Class Weights
    counts = Counter(y_train)
    total = sum(counts.values())
    w = [total / (len(counts) * counts[i]) for i in range(len(counts))]
    class_weights = torch.FloatTensor(w).to(device)
    print(f"      Class Weights: {w}")

    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    test_ds  = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))

    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)
    test_loader  = DataLoader(test_ds, batch_size=512, shuffle=False)

    # ─── Initialize Model ─────────────────────────────────────────────────────
    model = SimpleMLP(input_size=X_train.shape[1], num_classes=len(class_names)).to(device)
    print(f"      Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5, verbose=True)

    # ─── Training Loop ────────────────────────────────────────────────────────
    print("\n[3/4] Starting training...")
    best_acc = 0
    epochs = 30
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += yb.size(0)
            correct += (predicted == yb).sum().item()
            
        train_acc = 100 * correct / total
        avg_loss = total_loss / len(train_loader)
        
        # Evaluate
        model.eval()
        test_correct = 0
        test_total = 0
        all_test_preds = []
        all_test_labels = []
        
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                outputs = model(xb)
                _, predicted = torch.max(outputs, 1)
                test_total += yb.size(0)
                test_correct += (predicted == yb).sum().item()
                
                if epoch % 10 == 0 or epoch == epochs - 1:
                    all_test_preds.extend(predicted.cpu().numpy())
                    all_test_labels.extend(yb.cpu().numpy())
        
        test_acc = 100 * test_correct / test_total
        scheduler.step(avg_loss)
        
        print(f"      Epoch {epoch+1:2d}/{epochs} | Loss: {avg_loss:.6f} | Train Acc: {train_acc:6.2f}% | Test Acc: {test_acc:6.2f}%")
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"      ---> New Best Model! ({best_acc:.2f}%)")
            
        if (epoch % 10 == 0 or epoch == epochs - 1) and len(all_test_preds) > 0:
            print(f"\n      Snapshot Report (Epoch {epoch+1}):")
            print(classification_report(all_test_labels, all_test_preds, target_names=class_names, zero_division=0))

    # ─── Final Evaluation ─────────────────────────────────────────────────────
    print("\n[4/4] Final Evaluation Loop...")
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for xb, yb in test_loader:
            outputs = model(xb.to(device))
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(yb.numpy())
            
    print("\n" + classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))
    print(f"Model saved to: {MODEL_PATH}")
    print("Optimization complete.")

if __name__ == "__main__":
    train()
