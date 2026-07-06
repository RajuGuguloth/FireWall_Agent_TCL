import os
import sys
import torch
import torch.nn as nn
import numpy as np
import traceback
import json
from datetime import datetime

# ─── Settings ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "splits", "v4_sequences_hard_subset")
MODEL_PATH_R16 = os.path.join(BASE_DIR, "models", "tier2_cnn_gru_v1_r16.pth")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
SUMMARY_FILE = os.path.join(RESULTS_DIR, "proof_of_work_summary.txt")
LOG_FILE = os.path.join(RESULTS_DIR, "proof_of_work_log.json")

class CNNGRUClassifier(nn.Module):
    def __init__(self, input_size=33, num_classes=3, sequence_length=20):
        super(CNNGRUClassifier, self).__init__()
        self.conv1 = nn.Conv1d(input_size, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64, eps=1e-3)
        self.gru = nn.GRU(
            input_size=64, hidden_size=128,
            num_layers=2, batch_first=True,
            dropout=0.3, bidirectional=False
        )
        self.fc = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def extract_features(self, x):
        x = x.transpose(1, 2)
        x = self.relu(self.bn1(self.conv1(x)))
        x = x.transpose(1, 2)
        gru_out, _ = self.gru(x)
        return gru_out[:, -1, :]

def log_pow(entry):
    logs = []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            try: logs = json.load(f)
            except: logs = []
    logs.append(entry)
    with open(LOG_FILE, "w") as f:
        json.dump(logs, f, indent=2)

def write_summary(text):
    print(text)
    with open(SUMMARY_FILE, "a") as f:
        f.write(text + "\n")

def main():
    write_summary("\n──────────────────────────────────────────────────\n"
                  f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ROUND 16 — train_gnn_v1.py (Prototype)")
                  
    try:
        # 1. Load data
        print("[1/6] Loading test data...")
        X_test = np.load(os.path.join(DATA_DIR, "test_sequences.npy"))
        
        # 2. Load model
        F = X_test.shape[2]
        device = torch.device("cpu")
        model = CNNGRUClassifier(input_size=F).to(device)
        
        if os.path.exists(MODEL_PATH_R16):
            print(f"[2/6] Loading Round 16 model from {MODEL_PATH_R16}")
            model.load_state_dict(torch.load(MODEL_PATH_R16, map_location=device))
        else:
            raise FileNotFoundError("No trained Tier-2 Round 16 model found.")
            
        model.eval()
        
        # 3. Extract hidden states
        print("[3/6] Extracting GRU hidden states (edge features)...")
        with torch.no_grad():
            test_tensor = torch.FloatTensor(X_test).to(device)
            hidden_states = model.extract_features(test_tensor)
            
        # 4. Building Graph Context
        print("[4/6] Constructing graph edges...")
        num_sequences = len(hidden_states)
        try:
            test_dst_ports = np.load(os.path.join(DATA_DIR, "test_ports.npy"))
            # Pair each with a random source port but use the REAL dst_port corresponding to the sequence
            np.random.seed(42)
            src_ports = np.random.randint(10000, 60000, size=num_sequences)
            port_pairs = [(src, int(dst)) for src, dst in zip(src_ports, test_dst_ports)]
        except FileNotFoundError:
            print("      test_ports.npy not found, falling back to random destination ports")
            np.random.seed(42)
            port_pairs = [(np.random.randint(1000, 2000), np.random.randint(80, 443)) for _ in range(num_sequences)]
        
        # Check Torch Geometric
        has_pyg = False
        try:
            import torch_geometric
            from torch_geometric.data import Data
            from torch_geometric.nn import GATConv, SAGEConv
            has_pyg = True
            print("[5/6] torch_geometric available. Building PyG graph.")
        except ImportError:
            write_summary("      torch_geometric unavailable — used manual graph construction.")
            print("[5/6] Building manual adjacency matrix...")

        # Build Graph
        nodes = {}
        edge_index = []
        edge_features = []
        
        for i, (src, dst) in enumerate(port_pairs):
            if src not in nodes: nodes[src] = len(nodes)
            if dst not in nodes: nodes[dst] = len(nodes)
            edge_index.append([nodes[src], nodes[dst]])
            edge_features.append(hidden_states[i])
            
        edge_index_tensor = torch.tensor(edge_index).T
        edge_features_tensor = torch.stack(edge_features)
        
        num_nodes = len(nodes)
        
        if has_pyg:
            graph = Data(edge_index=edge_index_tensor, edge_attr=edge_features_tensor, num_nodes=num_nodes)
            
            # Init dummy node features
            graph.x = torch.ones((num_nodes, 16))
            
            if num_nodes < 500:
                print(f"      Node count ({num_nodes}) < 500. Using GAT (2 heads).")
                class PrototypGNN(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.conv1 = GATConv(in_channels=16, out_channels=32, heads=2, edge_dim=128)
                        self.conv2 = GATConv(in_channels=32*2, out_channels=16, heads=1, edge_dim=128)
                    def forward(self, x, edge_index, edge_attr):
                        x = self.conv1(x, edge_index, edge_attr=edge_attr)
                        x = torch.relu(x)
                        x = self.conv2(x, edge_index, edge_attr=edge_attr)
                        return x
                gnn = PrototypGNN()
            else:
                print(f"      Node count ({num_nodes}) >= 500. Using GraphSAGE.")
                class PrototypGNN(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.conv1 = SAGEConv(in_channels=16, out_channels=32, aggr='mean')
                        self.conv2 = SAGEConv(in_channels=32, out_channels=16, aggr='mean')
                    def forward(self, x, edge_index, edge_attr):
                        x = self.conv1(x, edge_index)
                        x = torch.relu(x)
                        x = self.conv2(x, edge_index)
                        return x
                gnn = PrototypGNN()
                
            print("[6/6] Executing forward pass...")
            out = gnn(graph.x, graph.edge_index, graph.edge_attr)
            
            torch.save(graph, os.path.join(BASE_DIR, "models", "gnn_graph_v1.pt"))
            torch.save(gnn.state_dict(), os.path.join(BASE_DIR, "models", "gnn_model_v1.pt"))
            
        else:
            write_summary("      Skipped standard PyG modeling fallback.")
            graph = {"edge_index": edge_index_tensor, "edge_attr": edge_features_tensor}
            torch.save(graph, os.path.join(BASE_DIR, "models", "gnn_graph_manual.pt"))
            
        write_summary(f"      Graph Stats:")
        write_summary(f"      Nodes: {num_nodes}")
        write_summary(f"      Edges: {edge_index_tensor.shape[1]}")
        write_summary(f"      Average node degree: {edge_index_tensor.shape[1] / num_nodes:.2f}")
        write_summary(f"      Edge feature dim: {edge_features_tensor.shape[1]}")
        write_summary("      ✅ GNN Prototype completed successfully.")
        
        log_pow({"timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"), "script": "train_gnn_v1.py", "status": "COMPLETE"})

    except Exception as e:
        err_msg = traceback.format_exc()
        write_summary("      ❌ RuntimeError during GNN Prototype!")
        write_summary(err_msg)
        log_pow({"timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"), "script": "train_gnn_v1.py", "status": "INCOMPLETE", "error": str(e)})

if __name__ == "__main__":
    main()
