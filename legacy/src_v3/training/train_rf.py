"""
Train Tier-1 Random Forest Classifier
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
from pathlib import Path
import json

class Tier1Trainer:
    def __init__(self):
        self.model = None
        self.label_encoder = None
        self.feature_columns = [
            'packet_length', 'has_ip', 'has_tcp', 'has_udp', 'has_icmp',
            'ip_version', 'ip_ttl', 'ip_proto', 'src_port', 'dst_port', 'tcp_flags'
        ]
    
    def load_data(self):
        """Load train/val/test data"""
        print("=" * 50)
        print("Loading data...")
        print("=" * 50)
        
        train_df = pd.read_csv("data/splits/train/features.csv")
        val_df = pd.read_csv("data/splits/val/features.csv")
        test_df = pd.read_csv("data/splits/test/features.csv")
        
        print(f"Train: {len(train_df)} samples")
        print(f"Val:   {len(val_df)} samples")
        print(f"Test:  {len(test_df)} samples")
        
        return train_df, val_df, test_df
    
    def prepare_features(self, train_df, val_df, test_df):
        """Prepare features and labels"""
        print("\n" + "=" * 50)
        print("Preparing features...")
        print("=" * 50)
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        
        y_train = self.label_encoder.fit_transform(train_df['attack_type'])
        y_val = self.label_encoder.transform(val_df['attack_type'])
        y_test = self.label_encoder.transform(test_df['attack_type'])
        
        # Extract features
        X_train = train_df[self.feature_columns].values
        X_val = val_df[self.feature_columns].values
        X_test = test_df[self.feature_columns].values
        
        print(f"Features: {self.feature_columns}")
        print(f"Classes: {list(self.label_encoder.classes_)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train(self, X_train, y_train):
        """Train Random Forest"""
        print("\n" + "=" * 50)
        print("Training Random Forest...")
        print("=" * 50)
        
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        
        self.model.fit(X_train, y_train)
        print("✅ Training complete!")
        
        # Feature importance
        importances = self.model.feature_importances_
        print("\n📊 Feature Importances:")
        for feat, imp in sorted(zip(self.feature_columns, importances), 
                               key=lambda x: x[1], reverse=True):
            print(f"   {feat:20s}: {imp:.4f}")
    
    def evaluate(self, X_val, y_val, X_test, y_test):
        """Evaluate model"""
        print("\n" + "=" * 50)
        print("VALIDATION SET RESULTS")
        print("=" * 50)
        
        y_val_pred = self.model.predict(X_val)
        val_acc = accuracy_score(y_val, y_val_pred)
        print(f"Accuracy: {val_acc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_val, y_val_pred, 
                                   target_names=self.label_encoder.classes_))
        
        print("\n" + "=" * 50)
        print("TEST SET RESULTS")
        print("=" * 50)
        
        y_test_pred = self.model.predict(X_test)
        test_acc = accuracy_score(y_test, y_test_pred)
        print(f"Accuracy: {test_acc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_test_pred,
                                   target_names=self.label_encoder.classes_))
        
        return {
            'val_accuracy': val_acc,
            'test_accuracy': test_acc
        }
    
    def save_model(self):
        """Save model and encoder as pickle files"""
        print("\n" + "=" * 50)
        print("Saving models...")
        print("=" * 50)
        
        models_dir = Path("models/serialized")
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # Save Random Forest
        model_path = models_dir / "tier1_rf.pkl"
        joblib.dump(self.model, model_path)
        print(f"✅ Model saved: {model_path}")
        
        # Save Label Encoder
        encoder_path = models_dir / "label_encoder.pkl"
        joblib.dump(self.label_encoder, encoder_path)
        print(f"✅ Encoder saved: {encoder_path}")
        
        # Save feature names
        feature_path = models_dir / "feature_names.json"
        with open(feature_path, 'w') as f:
            json.dump(self.feature_columns, f)
        print(f"✅ Features saved: {feature_path}")

if __name__ == "__main__":
    trainer = Tier1Trainer()
    
    # Load data
    train_df, val_df, test_df = trainer.load_data()
    
    # Prepare features
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_features(
        train_df, val_df, test_df
    )
    
    # Train
    trainer.train(X_train, y_train)
    
    # Evaluate
    results = trainer.evaluate(X_val, y_val, X_test, y_test)
    
    # Save
    trainer.save_model()
    
    print("\n" + "=" * 50)
    print("🎉 TIER-1 TRAINING COMPLETE!")
    print("=" * 50)
    print(f"Validation Accuracy: {results['val_accuracy']:.4f}")
    print(f"Test Accuracy: {results['test_accuracy']:.4f}")
