"""Export Tier-2 GRU hidden-state extractor to ONNX (128-D) for Tier-3 API."""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn

import config
from src.models.cnn_gru_v6 import CNNGRUClassifier

OUT = config.TIER2_EMBED


class EmbeddingExtractor(nn.Module):
    def __init__(self, base: CNNGRUClassifier):
        super().__init__()
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.gru = base.gru
        self.relu = base.relu

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x.transpose(1, 2)))).transpose(1, 2)
        return self.gru(x)[0][:, -1, :]


def main():
    import os
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    n_classes = len(__import__("joblib").load(config.ENCODER_PATH).classes_)
    base = CNNGRUClassifier(num_classes=n_classes)
    base.load_state_dict(torch.load(config.TIER2_PTH, map_location="cpu"))
    base.eval()
    model = EmbeddingExtractor(base).eval()
    dummy = torch.randn(1, config.WINDOW_SIZE, config.N_FEATURES)
    torch.onnx.export(
        model, dummy, OUT,
        export_params=True, opset_version=14,
        input_names=["sequence"], output_names=["embedding"],
        dynamic_axes={"sequence": {0: "batch"}, "embedding": {0: "batch"}},
    )
    sess = ort.InferenceSession(OUT)
    out = sess.run(None, {"sequence": dummy.numpy()})[0]
    with torch.no_grad():
        pt_out = model(dummy).numpy()
    max_diff = float(np.max(np.abs(out - pt_out)))
    print(f"Exported {OUT} shape={out.shape} max|onnx-pt|={max_diff:.6f}")
    if max_diff > 1e-4:
        raise RuntimeError(f"ONNX embedding parity failed: {max_diff}")


if __name__ == "__main__":
    main()
