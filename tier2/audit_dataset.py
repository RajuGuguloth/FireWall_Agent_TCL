import pandas as pd
import numpy as np
import os
from scipy.stats import pointbiserialr
import json

# Settings
DATA_PATH = "combined_dataset_v4_flow.csv"
OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def audit():
    print("="*60)
    print("STEP 0: Dataset Audit & Leakage Detection")
    print("="*60)

    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found.")
        return

    df = pd.read_csv(DATA_PATH)
    
    # 1. Class Distribution
    print("\n[1/3] Class Distribution:")
    counts = df['label'].value_counts()
    percent = df['label'].value_counts(normalize=True) * 100
    dist_report = pd.DataFrame({'Count': counts, 'Percentage': percent})
    print(dist_report)
    
    # 2. Leakage Audit
    print("\n[2/3] Checking for Feature Leakage (|corr| > 0.85):")
    # Identify numeric columns for correlation (excluding labels)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'label' in numeric_cols:
        numeric_cols.remove('label')
    
    leaking_features = []
    target_classes = df['attack_type'].unique()
    
    for col in numeric_cols:
        for attack in target_classes:
            if attack == 'BENIGN': continue
            
            binary = (df['attack_type'] == attack).astype(int)
            # Skip columns with zero variance
            if df[col].nunique() <= 1: continue
            
            corr, _ = pointbiserialr(df[col], binary)
            if abs(corr) > 0.85:
                print(f"  WARNING: {col} leakes {attack} (corr={corr:.2f})")
                leaking_features.append({'feature': col, 'attack': attack, 'correlation': corr})

    # 3. Save Report
    report_path = os.path.join(OUTPUT_DIR, "dataset_audit.txt")
    with open(report_path, "w") as f:
        f.write("Dataset Audit Report\n")
        f.write("="*20 + "\n")
        f.write(str(dist_report) + "\n\n")
        f.write("Leaking Features:\n")
        for leak in leaking_features:
            f.write(f"  {leak['feature']} -> {leak['attack']} (corr={leak['correlation']:.2f})\n")
    
    print(f"\n[3/3] Audit complete. Report saved to {report_path}")
    
    # Export leaking list for Step 4
    leaking_names = list(set([l['feature'] for l in leaking_features]))
    with open(os.path.join(OUTPUT_DIR, "leaking_features.json"), "w") as f:
        json.dump(leaking_names, f)

if __name__ == "__main__":
    audit()
