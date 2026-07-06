"""
Single source of truth for the R18 pipeline.

Every tier (gate, CNN-GRU, GNN) and the API must import paths/features from
here so they can never drift onto different schemas again. This file is the
structural fix for the original Tier-1 / Tier-2 disconnect.
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── The canonical 17-feature schema (do NOT redefine elsewhere) ──────────────
FEATURES_17 = [
    "packet_length", "has_tcp", "has_udp", "has_icmp",
    "payload_length", "payload_entropy",
    "is_ack", "is_rst", "is_fin", "is_psh",
    "is_high_port_src", "ip_ttl", "ip_proto",
    "dst_port", "tcp_flags",
    "flow_total_bytes", "flow_mean_pkt_len",
]
N_FEATURES  = len(FEATURES_17)
WINDOW_SIZE = 20
STRIDE      = 10
CLIP_VAL    = 5.0

# ── Canonical R18 artifacts (one scaler, one encoder, shared by all tiers) ───
DATA_DIR     = os.path.join(BASE_DIR, "data")
RAW_DIR      = os.path.join(DATA_DIR, "raw")
DATASET_CSV  = os.path.join(RAW_DIR, "combined_dataset_v5_final.csv")
SEQ_DIR      = os.path.join(BASE_DIR, "data", "splits", "v6_sequences")
SCALER_PATH  = os.path.join(BASE_DIR, "models", "serialized", "v6_scaler.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "models", "serialized", "v6_encoder.pkl")

TIER1_GATE   = os.path.join(BASE_DIR, "models", "tier1_gate_v6.pkl")
TIER2_PTH    = os.path.join(BASE_DIR, "models", "tier2_cnn_gru_v1_r18.pth")
TIER2_TEMP   = os.path.join(BASE_DIR, "models", "tier2_r18_temperature.json")
TIER2_ONNX   = os.path.join(BASE_DIR, "models", "onnx", "tier2_cnn_gru_r18.onnx")
TIER2_EMBED  = os.path.join(BASE_DIR, "models", "onnx", "tier2_embedding_r18.onnx")
TIER3_ONECLASS = os.path.join(BASE_DIR, "models", "tier3_oneclass_v6.pkl")
TIER3_GNN    = os.path.join(BASE_DIR, "models", "tier3_gnn_v6.pth")
ALERTS_DB    = os.path.join(BASE_DIR, "data", "alerts.db")

GATE_THRESHOLD   = 0.90   # P(BENIGN) above which Tier-1 fast-paths ALLOW
BLOCK_THRESHOLD  = 0.95   # Tier-2 attack confidence -> BLOCK
FLAG_THRESHOLD   = 0.80   # Tier-2 attack confidence -> FLAG
