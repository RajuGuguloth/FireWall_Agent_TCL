"""
Tier-2: CNN+GRU Model - CPU Optimized
CNN: Detects local patterns in packet sequences
GRU: Understands temporal flow (lighter than LSTM)
Inspired by Mamba's selective state space approach
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import classification_report

class CNNGRUClassifier(nn.Module):
    def __init__(self, input_size=25, hidden_size=128, 
                 num_classes=6, sequence_length=10):
        super(CNNGRUClassifier, self).__init__()
        
        # CNN layers - detect local patterns
        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            padding=1
        )
        self.conv2 = nn.Conv1d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.batch_norm1 = nn.BatchNorm1d(32)
        self.batch_norm2 = nn.BatchNorm1d(64)
        
        # GRU layer - temporal patterns
        self.gru = nn.GRU(
            input_size=64,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
            bidirectional=True
        )
        
        # Attention mechanism (Mamba-inspired)
        self.attention = nn.Linear(hidden_size * 2, 1)
        
        # Classification head
        self.fc1 = nn.Linear(hidden_size * 2, 64)
        self.fc2 = nn.Linear(64, num_classes)
        
    def attention_weights(self, gru_output):
        """Calculate attention weights over sequence"""
        weights = self.attention(gru_output)
        weights = torch.softmax(weights, dim=1)
        weighted = gru_output * weights
        return weighted.sum(dim=1)
    
    def forward(self, x):
        # x shape: (batch, sequence_len, features)
        batch_size = x.size(0)
        
        # Reshape for CNN: (batch * seq_len, 1, features)
        x_cnn = x.view(batch_size * x.size(1), 1, -1)
        
        # CNN feature extraction
        x_cnn = self.relu(self.batch_norm1(self.conv1(x_cnn)))
        x_cnn = self.relu(self.batch_norm2(self.conv2(x_cnn)))
        
        # Global average pooling
        x_cnn = x_cnn.mean(dim=2)
        
        # Reshape back for GRU: (batch, seq_len, cnn_features)
        x_gru = x_cnn.view(batch_size, -1, 64)
        
        # GRU temporal analysis
        gru_out, _ = self.gru(x_gru)
        
        # Attention pooling (Mamba-inspired selective state)
        attended = self.attention_weights(gru_out)
        
        # Classification
        out = self.relu(self.fc1(attended))
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out

def create_sequences_from_features(X, sequence_length=10):
    """Create sequences from flat feature vectors"""
    sequences = []
    for i in range(len(X) - sequence_length + 1):
        seq = X[i:i+sequence_length]
        sequences.append(seq)
    return np.array(sequences)

def train_cnn_gru():
    print("=" * 50)
    print("Loading Enhanced Data...")
    print("=" * 50)
    
    # Load data
    X_train = np.load("data/splits/enhanced/X_train.npy")
    X_val = np.load("data/splits/enhanced/X_val.npy")
    X_test = np.load("data/splits/enhanced/X_test.npy")
    y_train = np.load("data/splits/enhanced/y_train.npy")
    y_val = np.load("data/splits/enhanced/y_val.npy")
    y_test = np.load("data/splits/enhanced/y_test.npy")
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print(f"Features: {X_train.shape[1]}")
    
    # Create sequences
    print("\nCreating sequences...")
    SEQ_LEN = 10
    
    X_train_seq = create_sequences_from_features(X_train, SEQ_LEN)
    y_train_seq = y_train[SEQ_LEN-1:]
    
    X_val_seq = create_sequences_from_features(X_val, SEQ_LEN)
    y_val_seq = y_val[SEQ_LEN-1:]
    
    X_test_seq = create_sequences_from_features(X_test, SEQ_LEN)
    y_test_seq = y_test[SEQ_LEN-1:]
    
    print(f"Train sequences: {len(X_train_seq)}")
    print(f"Val sequences: {len(X_val_seq)}")
    print(f"Test sequences: {len(X_test_seq)}")
    
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train_seq)
    X_val_t = torch.FloatTensor(X_val_seq)
    X_test_t = torch.FloatTensor(X_test_seq)
    y_train_t = torch.LongTensor(y_train_seq)
    y_val_t = torch.LongTensor(y_val_seq)
    y_test_t = torch.LongTensor(y_test_seq)
    
    # Data loaders
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
    
    # Model setup
    device = torch.device('cpu')
    print(f"\nDevice: {device}")
    
    model = CNNGRUClassifier(
        input_size=X_train.shape[1],
        hidden_size=128,
        num_classes=6,
        sequence_length=SEQ_LEN
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5
    )
    
    # Training
    print("\n" + "=" * 50)
    print("Training CNN+GRU...")
    print("=" * 50)
    
    num_epochs = 30
    best_val_acc = 0
    patience_counter = 0
    patience_limit = 7
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += y_batch.size(0)
            train_correct += (predicted == y_batch).sum().item()
        
        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += y_batch.size(0)
                val_correct += (predicted == y_batch).sum().item()
        
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        
        scheduler.step(val_loss)
        
        print(f"Epoch [{epoch+1:2d}/{num_epochs}] "
              f"Train: {train_acc:.2f}% "
              f"Val: {val_acc:.2f}% "
              f"LR: {optimizer.param_groups[0]['lr']:.5f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_accuracy': val_acc,
                'epoch': epoch,
                'input_size': X_train.shape[1],
                'sequence_length': SEQ_LEN
            }, 'models/serialized/tier2_cnn_gru.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    # Final test evaluation
    print("\n" + "=" * 50)
    print("Final Test Evaluation...")
    print("=" * 50)
    
    # Load best model
    checkpoint = torch.load('models/serialized/tier2_cnn_gru.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.numpy())
    
    # Load encoder for class names
    encoder = joblib.load("models/serialized/enhanced_label_encoder.pkl")
    
    test_acc = 100 * sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    
    print(f"\nTest Accuracy: {test_acc:.2f}%")
    print(f"Best Val Accuracy: {best_val_acc:.2f}%")
    
    print("\nDetailed Classification Report:")
    print(classification_report(
        all_labels, all_preds,
        target_names=encoder.classes_
    ))
    
    print("\n" + "=" * 50)
    print("TIER-2 CNN+GRU TRAINING COMPLETE!")
    print("=" * 50)
    print(f"Model saved: models/serialized/tier2_cnn_gru.pth")

if __name__ == "__main__":
    train_cnn_gru()
