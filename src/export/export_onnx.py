import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import numpy as np
import joblib
from pathlib import Path
import json

# Re-define model class for loading
class CNNGRUClassifier(nn.Module):
    def __init__(self, input_size=25, hidden_size=128,
                 num_classes=6, sequence_length=10):
        super(CNNGRUClassifier, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.batch_norm1 = nn.BatchNorm1d(32)
        self.batch_norm2 = nn.BatchNorm1d(64)
        self.gru = nn.GRU(
            input_size=64, hidden_size=hidden_size,
            num_layers=2, batch_first=True,
            dropout=0.2, bidirectional=True
        )
        self.attention = nn.Linear(hidden_size * 2, 1)
        self.fc1 = nn.Linear(hidden_size * 2, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def attention_weights(self, gru_output):
        weights = torch.softmax(self.attention(gru_output), dim=1)
        return (gru_output * weights).sum(dim=1)

    def forward(self, x):
        batch_size = x.size(0)
        x_cnn = x.view(batch_size * x.size(1), 1, -1)
        x_cnn = self.relu(self.batch_norm1(self.conv1(x_cnn)))
        x_cnn = self.relu(self.batch_norm2(self.conv2(x_cnn)))
        x_cnn = x_cnn.mean(dim=2)
        x_gru = x_cnn.view(batch_size, -1, 64)
        gru_out, _ = self.gru(x_gru)
        attended = self.attention_weights(gru_out)
        out = self.relu(self.fc1(attended))
        out = self.dropout(out)
        return self.fc2(out)

def export_pytorch_to_onnx(model_path, onnx_path):
    print(f"Exporting PyTorch model from {model_path} to {onnx_path}...")
    checkpoint = torch.load(model_path, map_location='cpu')
    
    input_size = checkpoint.get('input_size', 25)
    sequence_length = checkpoint.get('sequence_length', 10)
    num_classes = checkpoint.get('num_classes', 6)
    
    model = CNNGRUClassifier(
        input_size=input_size,
        hidden_size=128,
        num_classes=num_classes,
        sequence_length=sequence_length
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, sequence_length, input_size)
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print("✅ PyTorch ONNX export complete.")

def export_xgboost_to_onnx(model_path, onnx_path):
    import onnxmltools
    from onnxmltools.convert import convert_xgboost
    from onnxmltools.convert.common.data_types import FloatTensorType
    import xgboost as xgb
    
    print(f"Exporting XGBoost model from {model_path} to {onnx_path}...")
    
    # Load model
    # Note: For convert_xgboost, it often expects a booster or a sklearn-wrapped object
    # If it's a JSON/CBM, we might need a specific loading approach
    model = xgb.Booster()
    model.load_model(model_path)
    
    # Convert to ONNX
    # initial_types expects a list of tuples: (name, data_type)
    initial_type = [('input', FloatTensorType([None, 25]))]
    onnx_model = convert_xgboost(model, initial_types=initial_type, target_opset=12)
    
    # Save ONNX
    onnxmltools.utils.save_model(onnx_model, onnx_path)
    print("✅ XGBoost ONNX export complete.")

def main():
    MODELS_DIR = Path("models/serialized")
    ONNX_DIR = Path("models/onnx")
    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    
    # Export CNN+GRU
    cnn_gru_path = MODELS_DIR / "tier2_cnn_gru.pth"
    if cnn_gru_path.exists():
        export_pytorch_to_onnx(str(cnn_gru_path), str(ONNX_DIR / "tier2_cnn_gru.onnx"))
    else:
        print(f"❌ {cnn_gru_path} not found.")
        
    # Export XGBoost
    xgb_path = MODELS_DIR / "tier2_xgboost.json"
    if xgb_path.exists():
        try:
            export_xgboost_to_onnx(str(xgb_path), str(ONNX_DIR / "tier2_xgboost.onnx"))
        except Exception as e:
            print(f"❌ Failed to export XGBoost: {e}")
    else:
        print(f"❌ {xgb_path} not found.")

if __name__ == "__main__":
    main()
