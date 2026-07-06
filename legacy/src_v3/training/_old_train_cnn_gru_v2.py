"""
Tier-2: CNN+GRU V2 - Fixed class imbalance
Uses oversampling for rare classes
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import classification_report
from collections import Counter

class CNNGRUClassifier(nn.Module):
    def __init__(self, input_size=25, hidden_size=128,
                 num_classes=6, sequence_length=10):
        super(CNNGRUClassifier, self).__init__()

        # CNN layers
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.batch_norm1 = nn.BatchNorm1d(32)
        self.batch_norm2 = nn.BatchNorm1d(64)

        # GRU layer
        self.gru = nn.GRU(
            input_size=64,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
            bidirectional=True
        )

        # Attention
        self.attention = nn.Linear(hidden_size * 2, 1)

        # Classification head
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

def create_sequences_per_class(X, y, sequence_length=10):
    """Create sequences maintaining class labels"""
    sequences = []
    labels = []
    
    unique_classes = np.unique(y)
    
    for cls in unique_classes:
        cls_indices = np.where(y == cls)[0]
        cls_X = X[cls_indices]
        
        # Create sequences within same class
        for i in range(len(cls_X) - sequence_length + 1):
            seq = cls_X[i:i+sequence_length]
            sequences.append(seq)
            labels.append(cls)
        
        # Oversample rare classes
        if len(cls_indices) < 100:
            needed = 100 - len(cls_indices)
            for _ in range(needed):
                start = np.random.randint(0, max(1, len(cls_X) - sequence_length))
                seq = cls_X[start:start+sequence_length]
                if len(seq) == sequence_length:
                    sequences.append(seq)
                    labels.append(cls)
    
    return np.array(sequences), np.array(labels)

def get_weighted_sampler(y):
    """Create weighted sampler for class balance"""
    class_counts = Counter(y)
    weights = [1.0 / class_counts[label] for label in y]
    return WeightedRandomSampler(weights, len(weights))

def train_cnn_gru_v2():
    print("=" * 50)
    print("CNN+GRU V2 - Fixed Class Imbalance")
    print("=" * 50)

    # Load data
    X_train = np.load("data/splits/enhanced/X_train.npy")
    X_val = np.load("data/splits/enhanced/X_val.npy")
    X_test = np.load("data/splits/enhanced/X_test.npy")
    y_train = np.load("data/splits/enhanced/y_train.npy")
    y_val = np.load("data/splits/enhanced/y_val.npy")
    y_test = np.load("data/splits/enhanced/y_test.npy")

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Load encoder
    encoder = joblib.load("models/serialized/enhanced_label_encoder.pkl")
    print(f"Classes: {list(encoder.classes_)}")

    # Create class-aware sequences
    print("\nCreating class-aware sequences...")
    SEQ_LEN = 10

    X_train_seq, y_train_seq = create_sequences_per_class(X_train, y_train, SEQ_LEN)
    X_val_seq, y_val_seq = create_sequences_per_class(X_val, y_val, SEQ_LEN)
    X_test_seq, y_test_seq = create_sequences_per_class(X_test, y_test, SEQ_LEN)

    print(f"Train sequences: {len(X_train_seq)}")
    print(f"Val sequences: {len(X_val_seq)}")
    print(f"Test sequences: {len(X_test_seq)}")

    print("\nTrain class distribution:")
    for cls, count in Counter(y_train_seq).items():
        print(f"  {encoder.classes_[cls]}: {count}")

    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train_seq)
    X_val_t = torch.FloatTensor(X_val_seq)
    X_test_t = torch.FloatTensor(X_test_seq)
    y_train_t = torch.LongTensor(y_train_seq)
    y_val_t = torch.LongTensor(y_val_seq)
    y_test_t = torch.LongTensor(y_test_seq)

    # Weighted sampler for balanced training
    sampler = get_weighted_sampler(y_train_seq)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=128, sampler=sampler
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

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {total_params:,}")

    # Class weights for loss
    class_counts = Counter(y_train_seq)
    total = sum(class_counts.values())
    class_weights = torch.FloatTensor([
        total / (len(class_counts) * class_counts[i])
        for i in range(len(encoder.classes_))
    ])

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5, verbose=True
    )

    # Training
    print("\n" + "=" * 50)
    print("Training CNN+GRU V2...")
    print("=" * 50)

    num_epochs = 30
    best_val_acc = 0
    patience_counter = 0
    patience_limit = 7

    for epoch in range(num_epochs):
        # Train
        model.train()
        train_correct = 0
        train_total = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
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

        print(f"Epoch [{epoch+1:2d}/{num_epochs}] "
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

    # Test evaluation
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
    print("TIER-2 CNN+GRU V2 COMPLETE!")
    print("=" * 50)

if __name__ == "__main__":
    train_cnn_gru_v2()
