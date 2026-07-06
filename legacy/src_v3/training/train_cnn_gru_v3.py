"""
Tier-2: CNN+GRU V3 - Fixed double weighting issue
Uses Focal Loss for class imbalance
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import classification_report
from collections import Counter

class FocalLoss(nn.Module):
    """Focal Loss - focuses on hard examples"""
    def __init__(self, gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(weight=self.weight)(inputs, targets)
        pt = torch.exp(-ce_loss)
        return ((1 - pt) ** self.gamma) * ce_loss

class CNNGRUClassifier(nn.Module):
    def __init__(self, input_size=25, hidden_size=128,
                 num_classes=6, sequence_length=10):
        super(CNNGRUClassifier, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.batch_norm1 = nn.BatchNorm1d(32)
        self.batch_norm2 = nn.BatchNorm1d(64)
        self.gru = nn.GRU(
            input_size=64, hidden_size=hidden_size,
            num_layers=2, batch_first=True,
            dropout=0.2, bidirectional=True
        )
        self.attention = nn.Linear(hidden_size * 2, 1)
        self.fc1 = nn.Linear(hidden_size * 2, 64)
        self.fc2 = nn.Linear(64, num_classes)

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
        attended = self.attention_weights(gru_out)
        out = self.relu(self.fc1(attended))
        out = self.dropout(out)
        return self.fc2(out)

def create_sequences_per_class(X, y, sequence_length=10, oversample_min=200):
    """Create sequences per class with controlled oversampling"""
    sequences = []
    labels = []
    unique_classes = np.unique(y)

    for cls in unique_classes:
        cls_indices = np.where(y == cls)[0]
        cls_X = X[cls_indices]

        # Create natural sequences
        cls_seqs = []
        for i in range(len(cls_X) - sequence_length + 1):
            cls_seqs.append(cls_X[i:i+sequence_length])

        # Controlled oversampling for rare classes only
        if len(cls_seqs) < oversample_min and len(cls_X) >= sequence_length:
            while len(cls_seqs) < oversample_min:
                start = np.random.randint(0, max(1, len(cls_X) - sequence_length))
                cls_seqs.append(cls_X[start:start+sequence_length])

        sequences.extend(cls_seqs)
        labels.extend([cls] * len(cls_seqs))

    return np.array(sequences), np.array(labels)

def train_cnn_gru_v3():
    print("=" * 50)
    print("CNN+GRU V3 - Focal Loss + Controlled Oversampling")
    print("=" * 50)

    # Load data
    X_train = np.load("data/splits/enhanced/X_train.npy")
    X_val = np.load("data/splits/enhanced/X_val.npy")
    X_test = np.load("data/splits/enhanced/X_test.npy")
    y_train = np.load("data/splits/enhanced/y_train.npy")
    y_val = np.load("data/splits/enhanced/y_val.npy")
    y_test = np.load("data/splits/enhanced/y_test.npy")

    encoder = joblib.load("models/serialized/enhanced_label_encoder.pkl")
    
    # Create sequences first (from imbalanced packets)
    print("\nCreating initial sequences...")
    SEQ_LEN = 10
    X_train_seq_raw, y_train_seq_raw = create_sequences_per_class(
        X_train, y_train, SEQ_LEN, oversample_min=1 # No simple oversampling here
    )
    
    # ─── SEQUENCE-LEVEL SMOTE ───
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(random_state=42)
    
    # Flatten: (N, 10, 25) -> (N, 250)
    N, S, F = X_train_seq_raw.shape
    X_train_flat = X_train_seq_raw.reshape(N, -1)
    
    print(f"Applying SMOTE to {N} sequences (flattened to {S*F} features)...")
    X_train_res_flat, y_train_seq = smote.fit_resample(X_train_flat, y_train_seq_raw)
    
    # Reshape back: (N_new, 250) -> (N_new, 10, 25)
    X_train_seq = X_train_res_flat.reshape(-1, S, F)
    
    # ─── Validation/Test: Natural sequences only ───
    X_val_seq, y_val_seq = create_sequences_per_class(X_val, y_val, SEQ_LEN, 1)
    X_test_seq, y_test_seq = create_sequences_per_class(X_test, y_test, SEQ_LEN, 1)

    print(f"\nFinal Train sequences: {len(X_train_seq)}")
    print(f"Val sequences:         {len(X_val_seq)}")
    print(f"Test sequences:        {len(X_test_seq)}")

    print("\nBalanced Train class distribution:")
    for cls, count in sorted(Counter(y_train_seq).items()):
        print(f"  {encoder.classes_[cls]:20s}: {count}")

    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train_seq)
    X_val_t = torch.FloatTensor(X_val_seq)
    X_test_t = torch.FloatTensor(X_test_seq)
    y_train_t = torch.LongTensor(y_train_seq)
    y_val_t = torch.LongTensor(y_val_seq)
    y_test_t = torch.LongTensor(y_test_seq)

    # Simple shuffle (NO weighted sampler)
    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=128, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val_t, y_val_t),
        batch_size=128
    )
    test_loader = DataLoader(
        TensorDataset(X_test_t, y_test_t),
        batch_size=128
    )

    # Model
    device = torch.device('cpu')
    model = CNNGRUClassifier(
        input_size=X_train.shape[1],
        hidden_size=128,
        num_classes=len(encoder.classes_),
        sequence_length=SEQ_LEN
    ).to(device)

    print(f"\nParameters: {sum(p.numel() for p in model.parameters()):,}")

    # Focal loss (handles imbalance better than weighted CE)
    criterion = FocalLoss(gamma=2)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5
    )

    # Training
    print("\n" + "=" * 50)
    print("Training...")
    print("=" * 50)

    best_val_acc = 0
    patience_counter = 0
    patience_limit = 7

    for epoch in range(30):
        # Train
        model.train()
        train_correct = 0
        train_total = 0
        train_loss = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += y_batch.size(0)
            train_correct += (predicted == y_batch).sum().item()

        # Validate
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                val_loss += criterion(outputs, y_batch).item()
                _, predicted = torch.max(outputs, 1)
                val_total += y_batch.size(0)
                val_correct += (predicted == y_batch).sum().item()

        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        scheduler.step(val_loss)

        print(f"Epoch [{epoch+1:2d}/30] "
              f"Train: {train_acc:.2f}% "
              f"Val: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_accuracy': val_acc,
                'epoch': epoch,
                'input_size': X_train.shape[1],
                'sequence_length': SEQ_LEN,
                'num_classes': len(encoder.classes_)
            }, 'models/serialized/tier2_cnn_gru.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    # Test
    print("\n" + "=" * 50)
    print("Final Test Evaluation")
    print("=" * 50)

    checkpoint = torch.load('models/serialized/tier2_cnn_gru.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch.to(device))
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.numpy())

    test_acc = 100 * sum(
        p == l for p, l in zip(all_preds, all_labels)
    ) / len(all_labels)

    print(f"\nTest Accuracy:     {test_acc:.2f}%")
    print(f"Best Val Accuracy: {best_val_acc:.2f}%")
    print("\nDetailed Results:")
    print(classification_report(
        all_labels, all_preds,
        target_names=encoder.classes_,
        zero_division=0
    ))
    print("=" * 50)
    print("TIER-2 CNN+GRU V3 COMPLETE!")
    print("=" * 50)

if __name__ == "__main__":
    train_cnn_gru_v3()
