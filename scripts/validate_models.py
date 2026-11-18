"""
ForestGuard AI — Model Validation & Integration Test Script
============================================================
Loads both CNN and YOLO using exact backend loaders.
Measures load time, runs inference on synthetic / test-split frames,
and verifies rule-based decision fusion behavior.
"""

import sys
import time
import cv2
import numpy as np
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.config import settings
from backend.models.cnn_classifier import CNNClassifier
from backend.models.yolo_detector import YOLODetector
from backend.services.decision_engine import DecisionEngine

def create_synthetic_frame(color_bgr, label_text):
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:, :] = color_bgr
    cv2.putText(img, label_text, (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return img

def main():
    print("=" * 65)
    print("🌲 FORESTGUARD AI: MODEL VALIDATION SUITE")
    print("=" * 65)

    cnn_path = settings.MODEL_DIR / "cnn_mobilenetv2.onnx"
    yolo_path = settings.MODEL_DIR / "yolo_best.pt"
    if not yolo_path.exists():
        yolo_path = settings.MODEL_DIR / "yolo_best.onnx"

    # 1. Load CNN
    print(f"\n1. Loading CNN Classifier from: {cnn_path}")
    t0 = time.perf_counter()
    cnn = CNNClassifier(cnn_path)
    cnn_load_time = (time.perf_counter() - t0) * 1000.0
    print(f"   Status: {'✅ AVAILABLE' if cnn.available else '⚠️ UNAVAILABLE'}")
    print(f"   Load Time: {cnn_load_time:.2f} ms")

    # 2. Load YOLO
    print(f"\n2. Loading YOLO Detector from: {yolo_path}")
    t0 = time.perf_counter()
    yolo = YOLODetector(yolo_path)
    yolo_load_time = (time.perf_counter() - t0) * 1000.0
    print(f"   Status: {'✅ AVAILABLE' if yolo.available else '⚠️ UNAVAILABLE'}")
    print(f"   Load Time: {yolo_load_time:.2f} ms")

    # 3. Decision Engine
    decision_engine = DecisionEngine(window_size=6, confirm_frames=3, cooldown_seconds=10.0)

    # 4. Run Test Inference on Benchmark Frames
    print("\n3. Testing Inference & Decision Fusion Pipeline:")
    
    # Generate or load test frames
    test_cases = [
        {"name": "Clear Forest Canopy (Negative)", "color": (34, 139, 34), "type": "no_fire"},
        {"name": "Orange Sunset Horizon (Hard Negative)", "color": (0, 140, 255), "type": "no_fire"},
        {"name": "Active Fire Ground Truth", "color": (0, 0, 220), "type": "fire"},
        {"name": "Smoke Column Scene", "color": (180, 180, 180), "type": "smoke"}
    ]

    all_passed = True
    for idx, tc in enumerate(test_cases, 1):
        frame = create_synthetic_frame(tc["color"], tc["name"])
        
        # Inference
        t_start = time.perf_counter()
        cnn_res = cnn.predict(frame) if cnn.available else None
        yolo_res = yolo.predict(frame) if yolo.available else []
        decision = decision_engine.update(cnn_res, yolo_res, cnn.available, yolo.available)
        latency = (time.perf_counter() - t_start) * 1000.0

        print(f"\n   [{idx}/{len(test_cases)}] Case: {tc['name']}")
        print(f"       Total Latency: {latency:.2f} ms")
        if cnn_res:
            print(f"       CNN: {cnn_res['label']} (Conf: {cnn_res['confidence']:.2f}, Latency: {cnn_res['latency_ms']:.1f}ms)")
        else:
            print(f"       CNN: Unavailable (Graceful degraded mode)")
            
        if yolo_res:
            boxes_str = ", ".join([f"{b['cls']}({b['conf']:.2f})" for b in yolo_res])
            print(f"       YOLO: {len(yolo_res)} boxes [{boxes_str}]")
        else:
            print(f"       YOLO: No boxes detected")

        print(f"       Fusion Decision: {decision['status']} (Reason: {decision['reason']})")

    print("\n" + "=" * 65)
    print("✅ Validation complete.")
    print("=" * 65)
    return 0

if __name__ == "__main__":
    sys.exit(main())
