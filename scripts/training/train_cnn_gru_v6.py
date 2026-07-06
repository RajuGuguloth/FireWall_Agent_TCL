"""
Round 18 — Tier-2 CNN-GRU, 6 classes (BENIGN + 5 attacks).
"""
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, TensorDataset

import config as cfg
from src.models.cnn_gru_v6 import CNNGRUClassifier, FocalLoss

DATA = cfg.SEQ_DIR
ENC = cfg.ENCODER_PATH
MODEL = cfg.TIER2_PTH
TEMP = cfg.TIER2_TEMP


def main():
    print("=" * 60); print("  Tier-2 CNN-GRU — Round 18 (6 classes incl. BENIGN)"); print("=" * 60)
    Xtr = np.load(f"{DATA}/X_train.npy"); ytr = np.load(f"{DATA}/y_train.npy")
    Xva = np.load(f"{DATA}/X_val.npy");   yva = np.load(f"{DATA}/y_val.npy")
    Xte = np.load(f"{DATA}/X_test.npy");  yte = np.load(f"{DATA}/y_test.npy")
    le  = joblib.load(ENC); names = list(le.classes_)
    print("  classes:", names)

    device = torch.device("cpu")
    # alpha: protect BENIGN (rare + critical to never miss) and minority DNS
    alpha = [1.0] * len(names)
    for n, w in {"BENIGN": 2.0, "DNS_TUNNELING": 1.3, "SLOW_HTTP": 1.2,
                 "BRUTE_FORCE": 0.8, "DDOS_HTTP_FLOOD": 0.8}.items():
        if n in names:
            alpha[names.index(n)] = w
    alpha_t = torch.tensor(alpha)
    print("  focal alpha:", alpha)

    tl = DataLoader(TensorDataset(torch.FloatTensor(Xtr), torch.LongTensor(ytr)), batch_size=64, shuffle=True)
    vl = DataLoader(TensorDataset(torch.FloatTensor(Xva), torch.LongTensor(yva)), batch_size=256)

    model = CNNGRUClassifier(num_classes=len(names)).to(device)
    crit  = FocalLoss(alpha=alpha_t, gamma=2)
    opt   = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
    sch   = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)

    best_f1, best_ep, no_imp, patience, epochs = 0.0, 0, 0, 10, 60
    for ep in range(epochs):
        model.train(); tot = 0.0
        for xb, yb in tl:
            opt.zero_grad(); loss = crit(model(xb), yb); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); tot += loss.item()
        model.eval(); vp, vt = [], []
        with torch.no_grad():
            for xb, yb in vl:
                vp.extend(model(xb).argmax(1).numpy()); vt.extend(yb.numpy())
        vf1 = f1_score(vt, vp, average="macro", zero_division=0)
        sch.step(vf1)
        print(f"  Epoch {ep+1:2d}/{epochs} | T-Loss {tot/len(tl):.5f} | Val-F1 {vf1:.4f}")
        if vf1 > best_f1:
            best_f1, best_ep, no_imp = vf1, ep + 1, 0
            torch.save(model.state_dict(), MODEL); print(f"     -> best (Val-F1 {best_f1:.4f}) saved")
        else:
            no_imp += 1
        if no_imp >= patience:
            print(f"  Early stop at epoch {ep+1}"); break

    model.load_state_dict(torch.load(MODEL, map_location=device)); model.eval()

    # Temperature calibration on VAL
    vlog, vy = [], []
    with torch.no_grad():
        for xb, yb in vl:
            vlog.append(model(xb)); vy.extend(yb.numpy())
    vlog = torch.cat(vlog); vy_t = torch.tensor(vy)
    T = nn.Parameter(torch.ones(1)); o = optim.LBFGS([T], lr=0.01, max_iter=50)
    def step():
        o.zero_grad(); l = nn.CrossEntropyLoss()(vlog / T, vy_t); l.backward(); return l
    o.step(step); Tv = max(T.item(), 0.01)
    json.dump({"temperature": Tv, "best_epoch": best_ep, "val_f1": best_f1},
              open(TEMP, "w"), indent=2)
    print(f"\n  Calibrated T = {Tv:.4f}")

    # Final report on TEST (clean held-out)
    with torch.no_grad():
        tlog = model(torch.FloatTensor(Xte))
    pred = (tlog / Tv).argmax(1).numpy()
    print("\n  TEST report (clean held-out):")
    print(classification_report(yte, pred, target_names=names, digits=3, zero_division=0))
    print("  Macro-F1:", round(f1_score(yte, pred, average="macro"), 4))
    bi = names.index("BENIGN")
    cm = confusion_matrix(yte, pred, labels=range(len(names)))
    benign_total = cm[bi].sum(); benign_as_attack = benign_total - cm[bi, bi]
    print(f"  BENIGN false-positive rate (benign->attack): "
          f"{100*benign_as_attack/benign_total:.2f}%  ({benign_as_attack}/{benign_total})")


if __name__ == "__main__":
    main()
