import os
import numpy as np
from utils.sliding_window import generate_sliding_windows

DATA_ROOT = "data/processed"
WINDOW_SIZE = 64
STRIDE = 32

def load_one_sample(label_dir):
    files = os.listdir(label_dir)
    files = [f for f in files if f.endswith(".npy")]
    files.sort()
    path = os.path.join(label_dir, files[0])
    return np.load(path)

# Test benign
benign_path = os.path.join(DATA_ROOT, "benign")
benign_feats = load_one_sample(benign_path)
benign_windows = generate_sliding_windows(
    benign_feats, WINDOW_SIZE, STRIDE
)

print("Benign packets:", benign_feats.shape)
print("Benign windows:", benign_windows.shape)

# Test attack
attack_path = os.path.join(DATA_ROOT, "attacks")
attack_feats = load_one_sample(attack_path)
attack_windows = generate_sliding_windows(
    attack_feats, WINDOW_SIZE, STRIDE
)

print("Attack packets:", attack_feats.shape)
print("Attack windows:", attack_windows.shape)

assert benign_windows.ndim == 3
assert attack_windows.ndim == 3

print("Real PCAP → window generation PASSED")

