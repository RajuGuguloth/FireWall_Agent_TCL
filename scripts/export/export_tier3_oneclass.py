"""Fit Tier-3 one-class Mahalanobis detector on r18 train-benign embeddings."""
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import joblib
import numpy as np
import torch

import config
from src.models.cnn_gru_v6 import CNNGRUClassifier

OUT = config.TIER3_ONECLASS
META = os.path.join(config.BASE_DIR, "models", "tier3_oneclass_v6.json")


def embed(model, X):
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 512):
            x = torch.FloatTensor(X[i:i + 512])
            x = model.relu(model.bn1(model.conv1(x.transpose(1, 2)))).transpose(1, 2)
            out.append(model.gru(x)[0][:, -1, :].numpy())
    return np.concatenate(out)


def main():
    le = joblib.load(config.ENCODER_PATH)
    B = list(le.classes_).index("BENIGN")
    Xtr = np.load(os.path.join(config.SEQ_DIR, "X_train.npy"))
    ytr = np.load(os.path.join(config.SEQ_DIR, "y_train.npy"))
    ben = Xtr[ytr == B]
    print(f"Fitting on {len(ben):,} benign train sequences")

    m = CNNGRUClassifier(num_classes=len(le.classes_))
    m.load_state_dict(torch.load(config.TIER2_PTH, map_location="cpu"))
    m.eval()
    E = embed(m, ben)
    mu = E.mean(0)
    inv = np.linalg.inv(np.cov(E.T) + 1e-3 * np.eye(E.shape[1]))
    d = E - mu
    scores = np.einsum("ij,jk,ik->i", d, inv, d)
    thr = float(np.percentile(scores, 99))

    joblib.dump({"mu": mu, "inv_cov": inv, "threshold": thr, "dim": E.shape[1]}, OUT)
    with open(META, "w") as f:
        json.dump({
            "threshold": thr, "percentile": 99, "dim": int(E.shape[1]),
            "train_benign_n": int(len(ben)),
            "source_model": os.path.basename(config.TIER2_PTH),
            "scaler": os.path.basename(config.SCALER_PATH),
        }, f, indent=2)
    print(f"Saved {OUT} threshold={thr:.6f}")


if __name__ == "__main__":
    main()
