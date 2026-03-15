# NDN AI Firewall Project

This repository contains the Round 16 implementation of the NDN AI Firewall, featuring behavior-based attack detection using CNN-GRU and GNN models.

## Project Structure
- `tier2/refine_dataset.py`: Cleans and prepares the dataset into sequences.
- `tier2/train_cnn_gru_v4.py`: Trains the behavior-based classifier (CNN-GRU).
- `tier2/train_gnn_v1.py`: Extracts topology features using Graph Neural Networks (GAT).
- `requirements.txt`: Python dependencies.

## Setup Instructions for Team Members

### 1. Environment Setup
Create a virtual environment and install dependencies:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Data Preparation
For patent security and performance reasons, raw CSV files and trained model weights are **not** uploaded to Git. To replicate the results:
1. Obtain the `combined_dataset_v4_flow.csv` file from the team shared drive.
2. Place it in the project root.
3. Run the refinement pipeline:
   ```bash
   python3 tier2/refine_dataset.py
   ```
   This will generate the `data/splits/v4_sequences_hard_subset` directory used for training.

### 3. Model Training
Train the CNN-GRU model from scratch:
```bash
python3 tier2/train_cnn_gru_v4.py
```
This saves the model to `models/tier2_cnn_gru_v1_r16.pth`.

### 4. GNN Topology
Run the GNN prototype to verify topological detection:
```bash
python3 tier2/train_gnn_v1.py
```

## Round 16 Metrics
- **Macro F1 Score**: 0.99
- **Confidence Thresholds**: 0.3 for all classes
- **Entropy Baseline**: 0.68 (The model significantly outperforms simple heuristics)
