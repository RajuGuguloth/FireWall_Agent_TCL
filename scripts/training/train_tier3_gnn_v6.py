"""
Round 18 — Tier-3 Edge-Conditioned GNN (zero-day / topological anomaly),
REALIGNED to the R18 contract: r18 (6-class) embeddings + v6_scaler.

Unsupervised: trains an edge-conditioned bipartite autoencoder on BENIGN flow
behaviour only; flags flows whose reconstruction error exceeds the 95th-pct
benign threshold. Uses the SAME scaler/feature schema as Tier-1 and Tier-2.

Saves: models/tier3_gnn_v6.pth, results/zero_day_metrics_v6.txt
"""
import os, json
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc
from torch_geometric.data import Data
from torch_geometric.nn import NNConv

import config

RESULTS_DIR = os.path.join(config.BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


class CNNGRUClassifier(nn.Module):
    """R18 architecture; extract_features returns the 128-D GRU hidden state."""
    def __init__(self, input_size=17, num_classes=6):
        super().__init__()
        self.conv1 = nn.Conv1d(input_size, 64, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm1d(64, eps=1e-3)
        self.gru   = nn.GRU(64, 128, num_layers=2, batch_first=True, dropout=0.3)
        self.fc    = nn.Linear(128, num_classes)
        self.relu  = nn.ReLU(); self.dropout = nn.Dropout(0.3)
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x.transpose(1, 2)))).transpose(1, 2)
        return self.fc(self.dropout(self.gru(x)[0][:, -1, :]))
    def extract_features(self, x):
        x = self.relu(self.bn1(self.conv1(x.transpose(1, 2)))).transpose(1, 2)
        return self.gru(x)[0][:, -1, :]


class EdgeConditionedAutoencoder(nn.Module):
    def __init__(self, node_dim=16, edge_dim=128, hidden_dim=32):
        super().__init__()
        self.nn1 = nn.Sequential(nn.Linear(edge_dim, 64), nn.ReLU(),
                                 nn.Linear(64, node_dim * hidden_dim))
        self.conv1 = NNConv(node_dim, hidden_dim, self.nn1, aggr="mean")
        self.decoder = nn.Sequential(nn.Linear(hidden_dim * 2, 64), nn.ReLU(),
                                     nn.Linear(64, edge_dim))
    def forward(self, x, edge_index, edge_attr):
        z = F.elu(self.conv1(x, edge_index, edge_attr))
        src, dst = edge_index
        return self.decoder(torch.cat([z[src], z[dst]], dim=1))


def extract_windows(df, feature_cols, max_seq=6000):
    seqs, ports = [], []
    for port, g in df.groupby("dst_port"):
        if len(g) < config.WINDOW_SIZE:
            continue
        v = g[feature_cols].values
        for i in range(0, len(v) - config.WINDOW_SIZE + 1, config.STRIDE):
            seqs.append(v[i:i + config.WINDOW_SIZE]); ports.append(port)
            if len(seqs) >= max_seq:
                return np.array(seqs), np.array(ports)
    return np.array(seqs), np.array(ports)


def build_graph(ports, embeds, port_to_id, num_producers, node_dim=16):
    ei, ea = [], []
    for i in range(len(embeds)):
        c = num_producers + i; p = port_to_id[ports[i]]
        ei += [[c, p], [p, c]]; ea += [embeds[i], embeds[i]]
    return Data(x=torch.ones(num_producers + len(embeds), node_dim),
                edge_index=torch.tensor(ei, dtype=torch.long).t().contiguous(),
                edge_attr=torch.stack(ea))


def main():
    print("=" * 60); print("  Tier-3 EC-GNN — R18 aligned (r18 embeds + v6_scaler)"); print("=" * 60)
    device = torch.device("cpu")

    scaler = joblib.load(config.SCALER_PATH)
    feats = config.FEATURES_17
    df = pd.read_csv(config.DATASET_CSV)
    df = df[df["label"].isin(["BENIGN"] + [c for c in df["label"].unique() if c != "BENIGN"])]

    normal_df = df[df["label"] == "BENIGN"].copy()
    attack_df = df[df["label"] != "BENIGN"].copy()
    n_seq, n_ports = extract_windows(normal_df, feats, max_seq=6000)
    a_seq, a_ports = extract_windows(attack_df, feats, max_seq=4000)

    def scale(s):
        f = scaler.transform(s.reshape(-1, config.N_FEATURES))
        return np.clip(f, -config.CLIP_VAL, config.CLIP_VAL).reshape(s.shape)
    n_seq, a_seq = scale(n_seq).astype(np.float32), scale(a_seq).astype(np.float32)
    print(f"  Benign flows {len(n_seq)} | Attack flows {len(a_seq)}")

    # R18 embeddings (128-D GRU hidden)
    t2 = CNNGRUClassifier(num_classes=6).to(device)
    t2.load_state_dict(torch.load(config.TIER2_PTH, map_location=device)); t2.eval()
    with torch.no_grad():
        ne = t2.extract_features(torch.tensor(n_seq)).cpu()
        ae = t2.extract_features(torch.tensor(a_seq)).cpu()

    uports = np.unique(np.concatenate([n_ports, a_ports]))
    pid = {p: i for i, p in enumerate(uports)}; nprod = len(uports)
    split = int(len(ne) * 0.7)
    train_g = build_graph(n_ports[:split], ne[:split], pid, nprod).to(device)
    test_embeds = torch.cat([ne[split:], ae], 0)
    test_ports = np.concatenate([n_ports[split:], a_ports])
    test_lab = np.concatenate([np.zeros(len(ne) - split), np.ones(len(ae))])
    test_g = build_graph(test_ports, test_embeds, pid, nprod).to(device)

    model = EdgeConditionedAutoencoder(16, 128, 32).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.005); crit = nn.MSELoss()
    model.train()
    for ep in range(100):
        opt.zero_grad()
        loss = crit(model(train_g.x, train_g.edge_index, train_g.edge_attr), train_g.edge_attr)
        loss.backward(); opt.step()
        if (ep + 1) % 20 == 0:
            print(f"  Epoch {ep+1:03d}/100  recon-loss {loss.item():.6f}")

    model.eval()
    with torch.no_grad():
        tr = model(train_g.x, train_g.edge_index, train_g.edge_attr)
        tr_err = torch.mean((tr[0::2] - train_g.edge_attr[0::2]) ** 2, 1).cpu().numpy()
        thr = np.percentile(tr_err, 95)
        te = model(test_g.x, test_g.edge_index, test_g.edge_attr)
        te_err = torch.mean((te[0::2] - test_g.edge_attr[0::2]) ** 2, 1).cpu().numpy()

    pred = (te_err > thr).astype(int)
    tpr = pred[test_lab == 1].sum() / (test_lab == 1).sum()
    fpr = pred[test_lab == 0].sum() / (test_lab == 0).sum()
    fr, tr_, _ = roc_curve(test_lab, te_err); roc_auc = auc(fr, tr_)

    torch.save(model.state_dict(), config.TIER3_GNN)
    with open(os.path.join(RESULTS_DIR, "zero_day_metrics_v6.txt"), "w") as f:
        f.write("=== Tier-3 EC-GNN (R18 aligned) ===\n")
        f.write(f"Embeddings: r18 (6-class) | Scaler: v6\n")
        f.write(f"Threshold(95pct benign MSE): {thr:.6f}\n")
        f.write(f"TPR: {tpr*100:.2f}%  FPR: {fpr*100:.2f}%  ROC-AUC: {roc_auc:.4f}\n")

    print("=" * 60)
    print(f"  Zero-day TPR (detection): {tpr*100:.2f}%")
    print(f"  Zero-day FPR (false alarm): {fpr*100:.2f}%")
    print(f"  ROC-AUC: {roc_auc:.4f}")
    print(f"  Saved -> {config.TIER3_GNN}")
    print("=" * 60)


if __name__ == "__main__":
    main()
