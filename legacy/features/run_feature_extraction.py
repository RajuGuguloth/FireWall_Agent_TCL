import os
import subprocess

RAW_ROOT = "data/raw"
OUT_ROOT = "data/processed"

LABELS = {
    "benign": "benign",
    "attacks": "attacks"
}

for label, subdir in LABELS.items():
    raw_dir = os.path.join(RAW_ROOT, subdir)
    out_dir = os.path.join(OUT_ROOT, subdir)

    os.makedirs(out_dir, exist_ok=True)

    for fname in os.listdir(raw_dir):
        if not fname.endswith(".pcap"):
            continue

        pcap_path = os.path.join(raw_dir, fname)
        subprocess.run([
            "python",
            "features/extract_features.py",
            pcap_path,
            out_dir
        ])

