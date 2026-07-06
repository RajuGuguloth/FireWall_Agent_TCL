"""
Train Tier-2 LSTM Model
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import joblib
from pathlib import Path

class LSTMClassifier(nn.Module):
    def __init__(self, input_size=11, hidden_size=64, num_layers=2, num_classes=6):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # LSTM output
        out, _ = self.lstm(x)
        # Take last time step
        out = out[:, -1, :]
        # Fully connected layer
        out = self.fc(out)
        return out

def train_lstm():
    print("=" * 50)
    print("Loading sequence data...")
    print("=" * 50)
    
    # Load data
    X_train = np.load("data/splits/sequences/X_train.npy")
    X_val = np.load("data/splits/sequences/X_val.npy")
    X_test = np.load("data/splits/sequences/X_test.npy")
    y_train = np.load("data/splits/sequences/y_train.npy")
    y_val = np.load("data/splits/sequences/y_val.npy")
    y_test = np.load("data/splits/sequences/y_test.npy")
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    X_val = torch.FloatTensor(X_val)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.LongTensor(y_train)
    y_val = torch.LongTensor(y_val)
    y_test = torch.LongTensor(y_test)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)
    test_loader = DataLoader(test_dataset, batch_size=64)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    model = LSTMClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training
    print("\n" + "=" * 50)
    print("Training LSTM...")
    print("=" * 50)
    
    num_epochs = 20
    best_val_acc = 0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += y_batch.size(0)
            train_correct += (predicted == y_batch).sum().item()
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                _, predicted = torch.max(outputs.data, 1)
                val_total += y_batch.size(0)
                val_correct += (predicted == y_batch).sum().item()
        
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Acc: {train_acc:.2f}% "
              f"Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'models/serialized/tier2_lstm.pth')
    
    # Test
    print("\n" + "=" * 50)
    print("Testing on test set...")
    print("=" * 50)
    
    model.load_state_dict(torch.load('models/serialized/tier2_lstm.pth'))
    model.eval()
    
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs.data, 1)
            test_total += y_batch.size(0)
            test_correct += (predicted == y_batch).sum().item()
    
    test_acc = 100 * test_correct / test_total
    
    print(f"\n✅ Test Accuracy: {test_acc:.2f}%")
    print(f"✅ Best Val Accuracy: {best_val_acc:.2f}%")
    
    print("\n" + "=" * 50)
    print("🎉 TIER-2 LSTM TRAINING COMPLETE!")
    print("=" * 50)

if __name__ == "__main__":
    train_lstm()
