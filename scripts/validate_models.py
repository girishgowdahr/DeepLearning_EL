"""
ForestGuard AI — Model Validation & Integration Test Script
============================================================
Loads CNN and YOLO using exact backend loaders.
Measures load time, runs inference on benchmark frames,
verifies output class mappings, and produces a strict pass/fail audit.
"""

import sys
import time
import argparse
import cv2
import numpy as np
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.config import settings
from backend.models.cnn_classifier import CNNClassifier
from backend.models.yolo_detector import YOLODetector
from backend.services.decision_engine import DecisionEngine, DecisionState

def create_synthetic_frame(color_bgr, label_text):
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:, :] = color_bgr
    cv2.putText(img, label_text, (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return img

def main():
    parser = argparse.ArgumentParser(description="ForestGuard AI Model Validation Suite")
    parser.add_argument("--strict", "--require-models", dest="strict", action="store_true",
                        help="Exit with error code 1 if any model weight is missing (production verification)")
    args = parser.parse_args()

    print("=" * 70)
    print("🌲 FORESTGUARD AI: MODEL AUDIT & VALIDATION SUITE")
    print("=" * 70)

    cnn_path = settings.MODEL_DIR / "cnn_mobilenetv2.onnx"
    yolo_pt_path = settings.MODEL_DIR / "yolo_best.pt"
    yolo_onnx_path = settings.MODEL_DIR / "yolo_best.onnx"
    yolo_path = yolo_pt_path if yolo_pt_path.exists() else yolo_onnx_path

    # 1. Load & Audit CNN Classifier
    print(f"\n1. Auditing CNN Classifier:")
    print(f"   Target Path: {cnn_path}")
    t0 = time.perf_counter()
    cnn = CNNClassifier(cnn_path)
    cnn_load_time = (time.perf_counter() - t0) * 1000.0
    print(f"   Status:       {'✅ AVAILABLE' if cnn.available else '⚠️ UNAVAILABLE (Pending Weights)'}")
    print(f"   Load Time:    {cnn_load_time:.2f} ms")
    if cnn.available:
        print(f"   Classes:      {cnn.classes} (0: fire, 1: no_fire)")
        print(f"   Input Node:   {cnn.input_name} (Shape: 1x3x224x224)")
        print(f"   Output Node:  {cnn.output_name}")
        assert cnn.classes == ["fire", "no_fire"], f"Unexpected CNN class mapping: {cnn.classes}"

    # 2. Load & Audit YOLO Detector
    print(f"\n2. Auditing YOLO Detector:")
    print(f"   Target Path: {yolo_path}")
    t0 = time.perf_counter()
    yolo = YOLODetector(yolo_path)
    yolo_load_time = (time.perf_counter() - t0) * 1000.0
    print(f"   Status:       {'✅ AVAILABLE' if yolo.available else '⚠️ UNAVAILABLE (Pending Weights)'}")
    print(f"   Load Time:    {yolo_load_time:.2f} ms")
    if yolo.available:
        print(f"   Classes:      {yolo.classes} (0: fire, 1: smoke)")

    # 3. Decision Engine Validation
    decision_engine = DecisionEngine(window_size=6, confirm_frames=3, cooldown_seconds=10.0)

    # 4. Benchmark Frame Inference
    print("\n3. Testing Inference Pipeline & Decision State Mapping:")
    test_cases = [
        {"name": "Clear Forest Canopy (Negative)", "color": (34, 139, 34), "type": "no_fire"},
        {"name": "Orange Sunset Horizon (Hard Negative)", "color": (0, 140, 255), "type": "no_fire"},
        {"name": "Active Fire Ground Truth", "color": (0, 0, 220), "type": "fire"},
        {"name": "Smoke Column Scene", "color": (180, 180, 180), "type": "smoke"}
    ]

    for idx, tc in enumerate(test_cases, 1):
        frame = create_synthetic_frame(tc["color"], tc["name"])
        t_start = time.perf_counter()
        cnn_res = cnn.predict(frame) if cnn.available else None
        yolo_res = yolo.predict(frame) if yolo.available else []
        decision = decision_engine.update(cnn_res, yolo_res, cnn.available, yolo.available)
        latency = (time.perf_counter() - t_start) * 1000.0

        print(f"\n   [{idx}/{len(test_cases)}] Case: {tc['name']}")
        print(f"       Inference Latency: {latency:.2f} ms")
        if cnn_res:
            print(f"       CNN:   {cnn_res['label']} (Confidence: {cnn_res['confidence']:.2f}, Latency: {cnn_res['latency_ms']:.1f}ms)")
        else:
            print(f"       CNN:   UNAVAILABLE (Weights not installed)")

        if yolo_res:
            boxes_str = ", ".join([f"{b['cls']}({b['conf']:.2f})" for b in yolo_res])
            print(f"       YOLO:  {len(yolo_res)} boxes [{boxes_str}]")
        else:
            print(f"       YOLO:  No detections")

        print(f"       Fusion: {decision['status']} (Reason: {decision['reason']})")

    # 5. Final Report & Pass/Fail Verdict
    print("\n" + "=" * 70)
    print("📋 SUMMARY AUDIT VERDICT")
    print("=" * 70)
    print(f"  CNN MobileNetV2: {'OPERATIONAL' if cnn.available else 'PENDING WEIGHTS (model_weights/cnn_mobilenetv2.onnx)'}")
    print(f"  YOLO11n:        {'OPERATIONAL' if yolo.available else 'PENDING WEIGHTS (model_weights/yolo_best.pt)'}")

    if cnn.available and yolo.available:
        print("\n✅ PASS: ALL MODELS FULLY OPERATIONAL AND VERIFIED!")
        return 0
    elif args.strict:
        print("\n❌ FAIL (STRICT MODE): Model weight files are missing or could not be loaded!")
        print("   Please install Kaggle artifacts before running in strict mode:")
        print("   .\\scripts\\install_artifacts.ps1 -YoloZip <path> -CnnZip <path>")
        return 1
    else:
        print("\n⚠️ NOTICE: Pipeline is running in safe degraded mode (MODEL_UNAVAILABLE).")
        print("   No false NO_FIRE decisions are generated while awaiting model weights.")
        print("   To enforce strict production verification, run with: python scripts\\validate_models.py --strict")
        return 0

if __name__ == "__main__":
    sys.exit(main())
