import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from torch_geometric.data import Data
from torch_geometric.nn import NNConv

# ─── Settings ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "combined_dataset_v5_final.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ─── Tier-2 CNN-GRU Definition ───────────────────────────────────────────────
class CNNGRUClassifier(nn.Module):
    """Conv1D(in→64) → GRU(64→128, 2L) → Linear(128→5)"""
    def __init__(self, input_size=17, num_classes=5):
        super().__init__()
        self.conv1   = nn.Conv1d(input_size, 64, kernel_size=3, padding=1)
        self.bn1     = nn.BatchNorm1d(64, eps=1e-3)
        self.gru     = nn.GRU(64, 128, num_layers=2, batch_first=True, dropout=0.3)
        self.fc      = nn.Linear(128, num_classes)
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):                     
        x = self.relu(self.bn1(self.conv1(x.transpose(1, 2)))).transpose(1, 2)
        last = self.gru(x)[0][:, -1, :]
        return self.fc(self.dropout(last))
        
    def extract_features(self, x):
        """Extract the 128D terminal GRU hidden state (Tier-3 Edge Embeddings)"""
        x = self.relu(self.bn1(self.conv1(x.transpose(1, 2)))).transpose(1, 2)
        last = self.gru(x)[0][:, -1, :]
        return last

# ─── Tier-3 Edge-Conditioned Bipartite Autoencoder ───────────────────────────
class EdgeConditionedAutoencoder(nn.Module):
    def __init__(self, node_dim=16, edge_dim=128, hidden_dim=32):
        super(EdgeConditionedAutoencoder, self).__init__()
        
        # Encoder (NNConv modulates message passing based on 128D flow behavior)
        self.nn1 = nn.Sequential(
            nn.Linear(edge_dim, 64),
            nn.ReLU(),
            nn.Linear(64, node_dim * hidden_dim)
        )
        self.conv1 = NNConv(node_dim, hidden_dim, self.nn1, aggr='mean')
        
        # Decoder (Reconstructs the 128D temporal flow embedding from connected nodes)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, edge_dim)
        )

    def forward(self, x, edge_index, edge_attr):
        # 1. Encode Nodes based on modulated flow features
        z = F.elu(self.conv1(x, edge_index, edge_attr))
        
        # 2. Decode Edges (Flows) from node representations
        src, dst = edge_index
        edge_repr = torch.cat([z[src], z[dst]], dim=1)
        pred_edge_attr = self.decoder(edge_repr)
        
        return pred_edge_attr

# ─── Helper Functions ────────────────────────────────────────────────────────
def extract_sequences_with_ports(df, feature_cols, window_size=20, stride=10, max_seq=4000):
    """Group by destination port and extract temporal flow windows."""
    sequences = []
    dst_ports = []
    
    for port, group in df.groupby('dst_port'):
        if len(group) < window_size:
            continue
            
        values = group[feature_cols].values
        
        for i in range(0, len(values) - window_size + 1, stride):
            sequences.append(values[i : i + window_size])
            dst_ports.append(port)
            if len(sequences) >= max_seq:
                return np.array(sequences), np.array(dst_ports)
                
    return np.array(sequences), np.array(dst_ports)

def build_bipartite_graph(ports, embeddings, port_to_id, num_producers, node_dim=16):
    """Construct a Consumer-to-Producer bipartite graph.
       Consumers are uniquely generated IDs representing flow origins.
       Producers are the targeted dst_ports.
       Edges are undirected but carry the directional flow embedding.
    """
    num_consumers = len(embeddings)
    num_nodes = num_producers + num_consumers
    
    edge_index = []
    edge_attr = []
    
    for i in range(num_consumers):
        consumer_id = num_producers + i
        producer_id = port_to_id[ports[i]]
        
        # Undirected edge pairing
        edge_index.append([consumer_id, producer_id])
        edge_index.append([producer_id, consumer_id])
        
        # Duplicate the flow embedding for both directions
        edge_attr.append(embeddings[i])
        edge_attr.append(embeddings[i])
        
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.stack(edge_attr)
    x = torch.ones(num_nodes, node_dim) # Topology is derived entirely from edges
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

# ─── Main Execution ──────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Tier-3 Edge-Conditioned Bipartite GNN (SOTA Zero-Day)")
    print("=" * 60)
    
    # 1. Feature Prep
    print(f"\n[1/6] Loading data and simulating NDN Endpoints...")
    actual_data_path = DATA_PATH
    if not os.path.exists(actual_data_path):
        actual_data_path = os.path.join(BASE_DIR, "combined_dataset_v5_final.csv")
    
    df = pd.read_csv(actual_data_path)
    
    scaler_path = os.path.join(MODELS_DIR, "serialized", "v5_scaler.pkl")
    scaler = joblib.load(scaler_path)
    feature_cols = list(getattr(scaler, 'feature_names_in_', []))
    if not feature_cols:
        feature_cols = ['packet_length', 'has_tcp', 'has_udp', 'has_icmp', 'payload_length', 
                        'payload_entropy', 'is_ack', 'is_rst', 'is_fin', 'is_psh', 'is_high_port_src', 
                        'ip_ttl', 'ip_proto', 'dst_port', 'tcp_flags', 'flow_total_bytes', 'flow_mean_pkt_len']
    
    normal_df = df[df['label'] == 'BENIGN'].copy()
    attack_df = df[df['label'] != 'BENIGN'].copy()
    
    normal_seqs, normal_ports = extract_sequences_with_ports(normal_df, feature_cols, max_seq=6000)
    attack_seqs, attack_ports = extract_sequences_with_ports(attack_df, feature_cols, max_seq=4000)
    
    def scale_3d(seqs):
        N, S, F = seqs.shape
        flat = seqs.reshape(-1, F)
        return scaler.transform(flat).reshape(N, S, F)
        
    normal_seqs_scaled = scale_3d(normal_seqs)
    attack_seqs_scaled = scale_3d(attack_seqs)
    
    print(f"      Extracted {len(normal_seqs_scaled)} BENIGN and {len(attack_seqs_scaled)} ATTACK flows.")
    
    # 2. Extract 128D Temporal Edge Features
    print("\n[2/6] Extracting 128D Flow Behavioral Embeddings via Tier-2 CNN-GRU...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tier2_model = CNNGRUClassifier().to(device)
    tier2_model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "tier2_cnn_gru_v1_r17.pth"), map_location=device))
    tier2_model.eval()
    
    with torch.no_grad():
        X_norm_t = torch.tensor(normal_seqs_scaled, dtype=torch.float32).to(device)
        X_att_t = torch.tensor(attack_seqs_scaled, dtype=torch.float32).to(device)
        
        normal_embeds = tier2_model.extract_features(X_norm_t).cpu()
        attack_embeds = tier2_model.extract_features(X_att_t).cpu()

    # 3. Construct Bipartite Graphs
    print("\n[3/6] Constructing Bipartite Consumer-Producer Topology...")
    unique_ports = np.unique(np.concatenate([normal_ports, attack_ports]))
    port_to_id = {port: idx for idx, port in enumerate(unique_ports)}
    num_producers = len(unique_ports)
    print(f"      Mapped {num_producers} unique content producers (services).")
    
    split_idx = int(len(normal_embeds) * 0.7)
    train_normal_embeds = normal_embeds[:split_idx]
    train_normal_ports = normal_ports[:split_idx]
    
    test_normal_embeds = normal_embeds[split_idx:]
    test_normal_ports = normal_ports[split_idx:]
    
    train_data = build_bipartite_graph(train_normal_ports, train_normal_embeds, port_to_id, num_producers).to(device)
    
    test_embeds = torch.cat([test_normal_embeds, attack_embeds], dim=0)
    test_ports = np.concatenate([test_normal_ports, attack_ports])
    test_labels = np.concatenate([np.zeros(len(test_normal_embeds)), np.ones(len(attack_embeds))])
    
    test_data = build_bipartite_graph(test_ports, test_embeds, port_to_id, num_producers).to(device)

    # 4. Train Edge-Conditioned Autoencoder
    print("\n[4/6] Training Edge-Conditioned GNN (Unsupervised on BENIGN)...")
    model = EdgeConditionedAutoencoder(node_dim=16, edge_dim=128, hidden_dim=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        pred_edges = model(train_data.x, train_data.edge_index, train_data.edge_attr)
        loss = criterion(pred_edges, train_data.edge_attr)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            print(f"      Epoch {epoch+1:03d}/100, Edge Reconstruction Loss: {loss.item():.6f}")

    # Threshold on directed Consumer->Producer edges (even indices 0, 2, 4...)
    model.eval()
    with torch.no_grad():
        train_recon = model(train_data.x, train_data.edge_index, train_data.edge_attr)
        train_directed_errors = torch.mean((train_recon[0::2] - train_data.edge_attr[0::2])**2, dim=1).cpu().numpy()
        threshold = np.percentile(train_directed_errors, 95)
    
    print(f"      Calculated Anomaly Threshold (95th Percentile): {threshold:.6f}")

    # 5. Evaluate
    print("\n[5/6] Evaluating on Mixed Graph (Normal + Zero-Day Attacks)...")
    with torch.no_grad():
        test_recon = model(test_data.x, test_data.edge_index, test_data.edge_attr)
        test_directed_errors = torch.mean((test_recon[0::2] - test_data.edge_attr[0::2])**2, dim=1).cpu().numpy()
        
    predictions = (test_directed_errors > threshold).astype(int)
    
    normal_test_err = test_directed_errors[test_labels == 0]
    attack_test_err = test_directed_errors[test_labels == 1]
    
    detected_attacks = np.sum(predictions[test_labels == 1])
    false_alarms = np.sum(predictions[test_labels == 0])
    
    tpr = detected_attacks / len(attack_embeds)
    fpr = false_alarms / len(test_normal_embeds)
    
    print(f"      True Positive Rate (TPR / Detection): {tpr*100:.2f}%")
    print(f"      False Positive Rate (FPR / False Alarms): {fpr*100:.2f}%")

    # 6. Save Plots
    print("\n[6/6] Generating Thesis Evaluation Plots...")
    
    metrics_path = os.path.join(RESULTS_DIR, "zero_day_metrics.txt")

    # compute ROC AUC here so we can save it to metrics
    fpr_roc, tpr_roc, _ = roc_curve(test_labels, test_directed_errors)
    roc_auc = auc(fpr_roc, tpr_roc)

    with open(metrics_path, "w") as f:
        f.write("=== Tier-3 EC-GNN Zero-Day Metrics (SOTA) ===\n")
        f.write(f"Anomaly Threshold (MSE, 95th pct of benign): {threshold:.6f}\n")
        f.write(f"Training Benign Flows: {split_idx}\n")
        f.write(f"Tested Normal Flows: {len(test_normal_embeds)}\n")
        f.write(f"Tested Attack Flows: {len(attack_embeds)}\n")
        f.write(f"Detected Zero-Day Attacks: {detected_attacks}\n")
        f.write(f"Detection Rate (TPR): {tpr*100:.2f}%\n")
        f.write(f"False Alarm Rate (FPR): {fpr*100:.2f}%\n")
        f.write(f"ROC AUC: {roc_auc:.4f}\n")

    # Dist Plot
    plt.figure(figsize=(10, 6))
    plt.hist(normal_test_err, bins=50, alpha=0.6, color='green', label='BENIGN Traffic (Normal)', density=True)
    plt.hist(attack_test_err, bins=50, alpha=0.6, color='red', label='ZERO-DAY Traffic (Attack)', density=True)
    plt.axvline(threshold, color='black', linestyle='dashed', linewidth=2, label=f'Anomaly Threshold ({threshold:.4f})')
    plt.title('Edge-Conditioned GNN Reconstruction Error\n(Bipartite Topological Anomaly Detection)')
    plt.xlabel('Mean Squared Error (Reconstruction Loss)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    dist_path = os.path.join(RESULTS_DIR, "gnn_anomaly_distribution.png")
    plt.savefig(dist_path, dpi=300, bbox_inches='tight')
    plt.close()

    # ROC Curve (fpr_roc, tpr_roc, roc_auc already computed above)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr_roc, tpr_roc, color='darkorange', lw=2, label=f'EC-GNN ROC (area = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)\nZero-Day Detection via EC-GNN')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    roc_path = os.path.join(RESULTS_DIR, "gnn_roc_curve.png")
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"      Distribution plot saved to: {dist_path}")
    print(f"      ROC Curve saved to: {roc_path}")
    print(f"      Metrics saved to: {metrics_path}")
    print("\n✅ Zero-Day EC-GNN Evaluation Completed Successfully.")

if __name__ == "__main__":
    main()
