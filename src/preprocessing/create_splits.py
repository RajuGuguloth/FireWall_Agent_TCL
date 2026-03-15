"""
Create train/validation/test splits
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

def create_splits(input_csv="data/processed/features/all_packets.csv",
                  train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """Split data into train/val/test sets"""
    
    print("=" * 50)
    print("Creating train/val/test splits...")
    print("=" * 50)
    
    # Load data
    df = pd.read_csv(input_csv)
    print(f"Total samples: {len(df)}")
    
    # Check class distribution
    print("\nClass distribution:")
    print(df['attack_type'].value_counts())
    
    # Separate rare classes (< 10 samples) from common classes
    class_counts = df['attack_type'].value_counts()
    rare_classes = class_counts[class_counts < 10].index.tolist()
    
    if rare_classes:
        print(f"\n⚠️  Rare classes detected: {rare_classes}")
        print("These will be split without stratification")
        
        # Split rare and common classes
        df_rare = df[df['attack_type'].isin(rare_classes)]
        df_common = df[~df['attack_type'].isin(rare_classes)]
        
        # Split common classes with stratification
        train_common, temp_common = train_test_split(
            df_common,
            test_size=(val_ratio + test_ratio),
            stratify=df_common['attack_type'],
            random_state=42
        )
        
        val_common, test_common = train_test_split(
            temp_common,
            test_size=test_ratio / (val_ratio + test_ratio),
            stratify=temp_common['attack_type'],
            random_state=42
        )
        
        # Split rare classes without stratification
        train_rare, temp_rare = train_test_split(
            df_rare,
            test_size=(val_ratio + test_ratio),
            random_state=42
        )
        
        val_rare, test_rare = train_test_split(
            temp_rare,
            test_size=test_ratio / (val_ratio + test_ratio),
            random_state=42
        )
        
        # Combine
        train_df = pd.concat([train_common, train_rare]).sample(frac=1, random_state=42)
        val_df = pd.concat([val_common, val_rare]).sample(frac=1, random_state=42)
        test_df = pd.concat([test_common, test_rare]).sample(frac=1, random_state=42)
    else:
        # All classes have enough samples
        train_df, temp_df = train_test_split(
            df,
            test_size=(val_ratio + test_ratio),
            stratify=df['attack_type'],
            random_state=42
        )
        
        val_df, test_df = train_test_split(
            temp_df,
            test_size=test_ratio / (val_ratio + test_ratio),
            stratify=temp_df['attack_type'],
            random_state=42
        )
    
    # Save splits
    splits_dir = Path("data/splits")
    train_df.to_csv(splits_dir / "train" / "features.csv", index=False)
    val_df.to_csv(splits_dir / "val" / "features.csv", index=False)
    test_df.to_csv(splits_dir / "test" / "features.csv", index=False)
    
    print(f"\n✅ Splits created:")
    print(f"   Train: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"   Val:   {len(val_df)} samples ({len(val_df)/len(df)*100:.1f}%)")
    print(f"   Test:  {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")
    
    print("\n📊 Train set distribution:")
    print(train_df['attack_type'].value_counts())
    
    print("\n📊 Val set distribution:")
    print(val_df['attack_type'].value_counts())
    
    print("\n📊 Test set distribution:")
    print(test_df['attack_type'].value_counts())
    
    return train_df, val_df, test_df

if __name__ == "__main__":
    train_df, val_df, test_df = create_splits()
