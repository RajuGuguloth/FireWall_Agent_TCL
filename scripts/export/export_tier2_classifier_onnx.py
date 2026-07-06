"""Export R18 Tier-2 6-class CNN-GRU classifier to ONNX (production inference path)."""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import joblib
import numpy as np
import onnxruntime as ort
import torch

import config
from src.models.cnn_gru_v6 import CNNGRUClassifier

OUT = config.TIER2_ONNX


def main():
    import os
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    n_classes = len(joblib.load(config.ENCODER_PATH).classes_)
    model = CNNGRUClassifier(num_classes=n_classes)
    model.load_state_dict(torch.load(config.TIER2_PTH, map_location="cpu"))
    model.eval()

    dummy = torch.randn(1, config.WINDOW_SIZE, config.N_FEATURES)
    torch.onnx.export(
        model,
        dummy,
        OUT,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["sequence"],
        output_names=["logits"],
        dynamic_axes={"sequence": {0: "batch"}, "logits": {0: "batch"}},
    )

    sess = ort.InferenceSession(OUT)
    inp = sess.get_inputs()[0].name
    onnx_out = sess.run(None, {inp: dummy.numpy()})[0]
    with torch.no_grad():
        pt_out = model(dummy).numpy()
    max_diff = float(np.max(np.abs(onnx_out - pt_out)))
    print(f"Exported {OUT} shape={onnx_out.shape} classes={n_classes} max|onnx-pt|={max_diff:.6f}")
    if max_diff > 1e-4:
        raise RuntimeError(f"ONNX parity check failed: max diff {max_diff}")


if __name__ == "__main__":
    main()
