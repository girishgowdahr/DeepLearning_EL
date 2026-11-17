"""
ONNX Export Script for MobileNetV2 — ForestGuard AI
Exports PyTorch weights to ONNX (CPU execution), verifies numerical tolerance,
and updates model_weights/MODEL_INFO.json.
"""

import os
import json
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torchvision import models

def main():
    parser = argparse.ArgumentParser(description="Export MobileNetV2 PyTorch model to ONNX")
    parser.add_argument("--weights", type=str, default="model_weights/cnn_mobilenetv2.pt", help="Path to .pt state dict")
    parser.add_argument("--out_onnx", type=str, default="model_weights/cnn_mobilenetv2.onnx", help="Path to output .onnx file")
    args = parser.parse_args()

    weights_path = Path(args.weights)
    out_onnx_path = Path(args.out_onnx)
    out_onnx_path.parent.mkdir(parents=True, exist_ok=True)

    if not weights_path.exists():
        print(f"Error: Weights file {weights_path} not found.")
        return

    # 1. Instantiate Model
    model = models.mobilenet_v2(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(model.last_channel, 2)
    )
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()

    # 2. Export to ONNX
    dummy_input = torch.randn(1, 3, 224, 224, requires_grad=False)
    torch.onnx.export(
        model,
        dummy_input,
        str(out_onnx_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )
    print(f"Exported ONNX model to {out_onnx_path}")

    # 3. Verify with ONNX Runtime
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(str(out_onnx_path), providers=["CPUExecutionProvider"])
        ort_inputs = {session.get_inputs()[0].name: dummy_input.numpy()}
        ort_outputs = session.run(None, ort_inputs)[0]

        with torch.no_grad():
            torch_outputs = model(dummy_input).numpy()

        diff = np.max(np.abs(ort_outputs - torch_outputs))
        print(f"Maximum absolute numerical difference: {diff:.6f}")
        assert diff < 1e-4, f"ONNX output differs from PyTorch by {diff}"
        print("ONNX verification PASSED!")
    except ImportError:
        print("onnxruntime not installed yet; skipping verification check.")

    # 4. Update MODEL_INFO.json
    model_info_path = Path("model_weights/MODEL_INFO.json")
    info = {}
    if model_info_path.exists():
        try:
            with open(model_info_path, "r", encoding="utf-8") as f:
                info = json.load(f)
        except Exception:
            info = {}

    file_size_mb = round(os.path.getsize(out_onnx_path) / (1024 * 1024), 2)
    info["cnn"] = {
        "name": "MobileNetV2 Classifier",
        "architecture": "MobileNetV2 Transfer Learning",
        "version": "1.0.0",
        "trained_by": "ForestGuard AI Training Pipeline",
        "input_shape": [1, 3, 224, 224],
        "classes": ["fire", "no_fire"],
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "file_size_mb": file_size_mb,
        "format": "ONNX (CPU)"
    }

    with open(model_info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    print(f"Updated {model_info_path} with CNN metadata.")

if __name__ == "__main__":
    main()
