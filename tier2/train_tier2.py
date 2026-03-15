import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from utils.sliding_window import generate_sliding_windows

DATA_ROOT = "data/processed"
WINDOW_SIZE = 64
STRIDE = 32
BATCH_SIZE = 16
EPOCHS = 3
LR = 1e-3

DEVICE = torch.device("cpu")

# -------- Model --------

class Tier2GRU(nn.Module):
    def __init__(self, feature_dim=13, hidden_dim=64):
        super().__init__()
        self.gru = nn.GRU(feature_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        _, h = self.gru(x)
        return self.fc(h[-1])

# -------- Dataset --------

class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# -------- Load data --------

def load_sequences(label_dir, label):
    all_windows = []
    all_labels = []

    for fname in os.listdir(label_dir):
        if not fname.endswith(".npy"):
            continue

        feats = np.load(os.path.join(label_dir, fname))
        windows = generate_sliding_windows(feats, WINDOW_SIZE, STRIDE)

        if len(windows) == 0:
            continue

        all_windows.append(windows)
        all_labels.extend([label] * len(windows))

    if not all_windows:
        return None, None

    X = np.vstack(all_windows)
    y = np.array(all_labels)
    return X, y

# -------- Main training --------

def main():
    X_b, y_b = load_sequences(os.path.join(DATA_ROOT, "benign"), 0)
    X_a, y_a = load_sequences(os.path.join(DATA_ROOT, "attacks"), 1)

    X = np.vstack([X_b, X_a])
    y = np.concatenate([y_b, y_a])

    dataset = SequenceDataset(X, y)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = Tier2GRU().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        total_loss = 0.0
        for xb, yb in loader:
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} | Loss {total_loss/len(loader):.4f}")

    os.makedirs("models/checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "models/checkpoints/tier2_gru.pt")
    print("Tier-2 GRU training complete and model saved")

if __name__ == "__main__":
    main()

