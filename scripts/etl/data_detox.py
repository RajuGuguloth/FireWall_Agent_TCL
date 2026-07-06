import pandas as pd
import numpy as np
from scapy.all import PcapReader, Raw
import os
import hashlib
from sklearn.model_selection import train_test_split

# --- CONFIG ---
RAW_DATA_DIR = "data/raw"
BENIGN_DIR = os.path.join(RAW_DATA_DIR, "benign")
ATTACK_DIR = os.path.join(RAW_DATA_DIR, "attacks")
OUTPUT_CSV = "detoxed_dataset.csv"

def extract_payloads(pcap_path, label, attack_type):
    """Extract sequences of (payload_hex, label, attack_type) from a pcap."""
    rows = []
    print(f"  Reading {pcap_path} ...", end="", flush=True)
    try:
        with PcapReader(pcap_path) as pcap:
            for pkt in pcap:
                payload = b""
                if pkt.haslayer(Raw):
                    payload = bytes(pkt[Raw])
                
                rows.append({
                    "payload": payload.hex(),
                    "label": label,
                    "attack_type": attack_type
                })
        print(f" Done ({len(rows)} packets)")
    except Exception as e:
        print(f" Error: {e}")
    return rows

def run_detox():
    print("="*60)
    print("🚀  STARTING FULL DATA DETOX")
    print("="*60)

    # --- DATA COLLECTION ---
    all_rows = []
    
    # Process Benign
    benign_files = [f for f in os.listdir(BENIGN_DIR) if f.endswith(".pcap")]
    for bf in benign_files:
        all_rows.extend(extract_payloads(os.path.join(BENIGN_DIR, bf), "BENIGN", "BENIGN"))
    
    # Process Attacks
    attack_files = [f for f in os.listdir(ATTACK_DIR) if f.endswith(".pcap")]
    attack_mapping = {
        "portscan": "PORT_SCAN",
        "httpflood": "DDOS_HTTP_FLOOD",
        "bruteforce": "BRUTE_FORCE",
        "slowhttp": "SLOW_HTTP",
        "dnstunnel": "DNS_TUNNELING"
    }
    
    for af in attack_files:
        a_type = "UNKNOWN"
        for key, val in attack_mapping.items():
            if key in af.lower():
                a_type = val
                break
        all_rows.extend(extract_payloads(os.path.join(ATTACK_DIR, af), "MALICIOUS", a_type))

    df = pd.DataFrame(all_rows)
    initial_count = len(df)
    print(f"\n[INITIAL] Total packets extracted: {initial_count:,}")

    # --- STEP 1: REMOVE DUPLICATES ---
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    duplicates_removed = before - len(df)
    print(f"[STEP 1] Found and removed {duplicates_removed:,} exact duplicate rows.")

    # --- STEP 2: FIX WRONG LABELS (Conflicts) ---
    # Find payloads that exist in both Malicious and Benign
    conflicting_payloads = df.groupby("payload")["label"].nunique()
    conflicts = conflicting_payloads[conflicting_payloads > 1].index.tolist()
    
    label_conflicts_found = len(conflicts)
    if label_conflicts_found > 0:
        # Strategy: Prioritize MALICIOUS to ensure we don't miss threats
        # or remove them if they are too noisy. Here we remove both to 'detox' the uncertainty.
        df = df[~df["payload"].isin(conflicts)]
        print(f"[STEP 2] Found {label_conflicts_found:,} payload strings shared between Benign and Malicious. (Removed all associated rows)")
    else:
        print(f"[STEP 2] No label conflicts found.")

    # --- STEP 3: HANDLE MISSING VALUES ---
    before = len(df)
    # Empty hex string means no payload
    df = df[df["payload"] != ""].reset_index(drop=True)
    empty_payloads_dropped = before - len(df)
    print(f"[STEP 3] Dropped {empty_payloads_dropped:,} rows with empty payloads.")

    # --- STEP 4: FIX CLASS IMBALANCE ---
    class_counts = df["label"].value_counts()
    malicious_count = class_counts.get("MALICIOUS", 0)
    benign_count = class_counts.get("BENIGN", 0)
    total = len(df)
    
    flag_imbalance = False
    if total > 0:
        mal_ratio = malicious_count / total
        ben_ratio = benign_count / total
        if mal_ratio < 0.30 or ben_ratio < 0.30:
            flag_imbalance = True

    print(f"[STEP 4] Class Distribution:")
    print(f"  MALICIOUS: {malicious_count:,} ({malicious_count/total*100:.1f}%)")
    print(f"  BENIGN:    {benign_count:,} ({benign_count/total*100:.1f}%)")
    
    if flag_imbalance:
        print(f"  ⚠️  FLAGGED: Class imbalance detected (one class < 30%).")
        print(f"  Suggestion: Use Undersampling on the majority class or SMOTE on payloads.")
    else:
        print(f"  ✅ Dataset is balanced.")

    # --- STEP 5: REMOVE OUTLIERS (Payload Length) ---
    df["payload_len"] = df["payload"].str.len() // 2  # hex to byte length
    mean_len = df["payload_len"].mean()
    std_len = df["payload_len"].std()
    threshold = 3 * std_len
    
    outliers = df[(df["payload_len"] > mean_len + threshold) | (df["payload_len"] < mean_len - threshold)]
    outlier_count = len(outliers)
    df = df.drop(outliers.index).reset_index(drop=True)
    print(f"[STEP 5] Removed {outlier_count:,} outlier packets (> 3 standard deviations from mean length).")

    # --- STEP 6: CHECK DATA LEAKAGE ---
    # Split 80/20
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
    
    train_payloads = set(train_df["payload"])
    leaked_indices = test_df[test_df["payload"].isin(train_payloads)].index
    leaked_count = len(leaked_indices)
    
    if leaked_count > 0:
        test_df = test_df.drop(leaked_indices)
        print(f"[STEP 6] Found {leaked_count:,} leaked samples in test set (matching training payloads). Removed them.")
    else:
        print(f"[STEP 6] No data leakage detected between Train and Test sets.")

    final_df = pd.concat([train_df, test_df])
    final_count = len(final_df)
    
    # --- STEP 7: FINAL REPORT ---
    print("\n" + "="*60)
    print("📊  FINAL DETOX REPORT")
    print("="*60)
    print(f"Total samples before detox: {initial_count:,}")
    print(f"Total samples after detox:  {final_count:,}")
    print("-" * 30)
    print(f"1. Duplicates Removed:      {duplicates_removed:,}")
    print(f"2. Label Conflicts Resolved: {label_conflicts_found:,}")
    print(f"3. Empty Payloads Dropped:   {empty_payloads_dropped:,}")
    print(f"4. Imbalance Status:         {'⚠️ IMPROVE' if flag_imbalance else '✅ OK'}")
    print(f"5. Outliers Removed:         {outlier_count:,}")
    print(f"6. Leaked Samples Removed:   {leaked_count:,}")
    print("-" * 30)
    
    ready = "YES" if not flag_imbalance and final_count > 0 else "NO (Fix imbalance/size)"
    print(f"Is the dataset now ready for training? {ready}")
    
    final_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Detoxed dataset saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    run_detox()
