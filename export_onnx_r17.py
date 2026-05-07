"""
Step 1: Export Round 17 Model to ONNX
"""
import torch
import torch.nn as nn
import os
import onnxruntime as ort
import numpy as np

# Same architecture as in train_cnn_gru_v4.py
class CNNGRUClassifier(nn.Module):
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

def export_onnx():
    model_path = "models/tier2_cnn_gru_v1_r17.pth"
    onnx_path  = "models/onnx/tier2_cnn_gru_r17.onnx"
    
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    
    device = torch.device("cpu")
    model = CNNGRUClassifier(input_size=17, num_classes=5)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Dummy input: (batch_size, sequence_length, features) -> (1, 20, 17)
    dummy_input = torch.randn(1, 20, 17, device=device)
    
    # Export
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["sequence"],
        output_names=["logits"],
        dynamic_axes={"sequence": {0: "batch_size"}, "logits": {0: "batch_size"}}
    )
    print(f"Exported successfully to {onnx_path}")
    
    # Test ONNX Export with onnxruntime
    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name
    
    dummy_np = np.random.randn(1, 20, 17).astype(np.float32)
    onnx_out = session.run(None, {input_name: dummy_np})
    
    out_shape = np.array(onnx_out).shape
    print(f"ONNX Test Output Shape: {out_shape} (Wait, list wrapping -> {onnx_out[0].shape})")
    
    if onnx_out[0].shape == (1, 5):
        print("ONNX EXPORT VERIFIED: SUCCESS (Output shape is 1, 5)")
    else:
        print(f"ONNX EXPORT VERIFIED: FAILED (Output shape is {onnx_out[0].shape})")

if __name__ == "__main__":
    export_onnx()
