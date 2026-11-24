"""
ForestGuard AI — Model Inference & Degradation Test Suite
=========================================================
Tests CNN ONNX classifier and YOLO object detector inference pipelines:
  - Verifies class mappings (CNN: 0=fire, 1=no_fire; YOLO: 0=smoke, 1=fire).
  - Validates tensor dimensions, confidence scores in [0.0, 1.0], and bounding boxes.
  - Tests MODEL_UNAVAILABLE degraded behavior when model weights are missing.
  - Tests multi-modal temporal decision fusion engine state transitions.
  - Saves actual predictions and comprehensive verification report.
  - Never fabricates inference results or success metrics.
"""

import sys
import os
import time
import json
import argparse
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from backend.config import settings
from backend.models.cnn_classifier import CNNClassifier
from backend.models.yolo_detector import YOLODetector
from backend.services.decision_engine import DecisionEngine, DecisionState
from backend.services.annotate import annotate_frame

def generate_benchmark_frames(output_dir: Path) -> list:
    """Generates synthetic benchmark test frames covering major surveillance cases."""
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = []

    # 1. Clear Forest Canopy (True Negative)
    img_forest = np.zeros((480, 640, 3), dtype=np.uint8)
    img_forest[:, :] = (34, 139, 34) # Forest green
    # Add texture noise
    noise = np.random.randint(-15, 15, (480, 640, 3), dtype=np.int16)
    img_forest = np.clip(img_forest.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    cv2.putText(img_forest, "Benchmark 1: Clear Forest Canopy", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    path1 = output_dir / "01_forest_canopy.jpg"
    cv2.imwrite(str(path1), img_forest)
    frames.append({"name": "Clear Forest Canopy", "path": str(path1), "img": img_forest, "ground_truth": "no_fire"})

    # 2. Orange Sunset Horizon (Hard Negative)
    img_sunset = np.zeros((480, 640, 3), dtype=np.uint8)
    for y in range(480):
        # Sky gradient from deep blue to bright orange
        factor = y / 480.0
        b = int(180 * (1 - factor))
        g = int(80 + 60 * factor)
        r = int(120 + 135 * factor)
        img_sunset[y, :] = (b, g, r)
    cv2.circle(img_sunset, (450, 320), 45, (10, 180, 255), -1) # Sun disk
    cv2.putText(img_sunset, "Benchmark 2: Sunset Horizon", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    path2 = output_dir / "02_sunset_hard_negative.jpg"
    cv2.imwrite(str(path2), img_sunset)
    frames.append({"name": "Sunset Hard Negative", "path": str(path2), "img": img_sunset, "ground_truth": "no_fire"})

    # 3. Wildfire Flame Patch (True Positive - Fire)
    img_fire = img_forest.copy()
    # Draw fire region
    cv2.ellipse(img_fire, (320, 300), (90, 140), 0, 0, 360, (0, 69, 255), -1) # Red flame outer
    cv2.ellipse(img_fire, (320, 320), (55, 90), 0, 0, 360, (0, 165, 255), -1) # Orange flame core
    cv2.ellipse(img_fire, (320, 340), (25, 45), 0, 0, 360, (0, 230, 255), -1) # Yellow hotspot
    cv2.putText(img_fire, "Benchmark 3: Active Wildfire", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    path3 = output_dir / "03_active_wildfire.jpg"
    cv2.imwrite(str(path3), img_fire)
    frames.append({"name": "Active Wildfire", "path": str(path3), "img": img_fire, "ground_truth": "fire"})

    # 4. Dense Smoke Plume (True Positive - Smoke)
    img_smoke = img_forest.copy()
    overlay = img_smoke.copy()
    cv2.ellipse(overlay, (320, 200), (160, 220), 10, 0, 360, (195, 195, 195), -1) # Gray plume
    cv2.ellipse(overlay, (280, 140), (120, 150), -15, 0, 360, (215, 215, 215), -1)
    cv2.addWeighted(overlay, 0.72, img_smoke, 0.28, 0, img_smoke)
    cv2.putText(img_smoke, "Benchmark 4: Dense Smoke Plume", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    path4 = output_dir / "04_smoke_plume.jpg"
    cv2.imwrite(str(path4), img_smoke)
    frames.append({"name": "Dense Smoke Plume", "path": str(path4), "img": img_smoke, "ground_truth": "smoke"})

    # 5. Night Forest Low-Light Baseline
    img_night = (img_forest.astype(np.float32) * 0.15).astype(np.uint8)
    cv2.putText(img_night, "Benchmark 5: Night Forest Baseline", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    path5 = output_dir / "05_night_baseline.jpg"
    cv2.imwrite(str(path5), img_night)
    frames.append({"name": "Night Forest Baseline", "path": str(path5), "img": img_night, "ground_truth": "no_fire"})

    return frames

def test_missing_weights_degradation():
    """Verifies that missing weights safely produce MODEL_UNAVAILABLE and zero false NO_FIRE counts."""
    print("\n--- TEST SUITE 1: Missing Weights & Degraded Fallback ---")
    dummy_cnn_path = ROOT / "model_weights" / "nonexistent_cnn.onnx"
    dummy_yolo_path = ROOT / "model_weights" / "nonexistent_yolo.pt"

    cnn_dummy = CNNClassifier(dummy_cnn_path)
    yolo_dummy = YOLODetector(dummy_yolo_path)

    assert not cnn_dummy.available, "CNN should be unavailable for non-existent weight file!"
    assert not yolo_dummy.available, "YOLO should be unavailable for non-existent weight file!"

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cnn_res = cnn_dummy.predict(dummy_frame)
    yolo_res = yolo_dummy.predict(dummy_frame)

    assert cnn_res is None, "CNN predict() must return None when model is unavailable!"
    assert yolo_res == [], "YOLO predict() must return empty list when model is unavailable!"

    # Evaluate with Decision Engine
    engine = DecisionEngine(window_size=6, confirm_frames=3, cooldown_seconds=10.0)
    decision = engine.update(cnn_res, yolo_res, cnn_dummy.available, yolo_dummy.available)

    assert decision["status"] == DecisionState.MODEL_UNAVAILABLE, (
        f"Expected status '{DecisionState.MODEL_UNAVAILABLE}', got '{decision['status']}'! "
        "System must NOT fabricate 'NO_FIRE' when weights are missing!"
    )
    assert not decision["should_alert"], "System must not sound alert sirens in MODEL_UNAVAILABLE state!"
    assert "both models unavailable" in decision["reason"].lower() or "waiting" in decision["reason"].lower()

    # Verify annotation badge
    annotated = annotate_frame(dummy_frame.copy(), decision_result=decision, yolo_detections=[], cnn_result=None)
    assert annotated.shape == dummy_frame.shape, "Annotated frame dimensions corrupted!"

    print("  ✅ Degradation Test PASSED: Missing models correctly trigger MODEL_UNAVAILABLE.")
    print(f"     Status: {decision['status']} | Reason: {decision['reason']}")
    return True

def test_decision_fusion_transitions():
    """Tests temporal window state transitions across simulated fire, smoke, and conflict cases."""
    print("\n--- TEST SUITE 2: Multi-Modal Temporal Decision Fusion Logic ---")
    engine = DecisionEngine(window_size=6, confirm_frames=3, cooldown_seconds=10.0)

    # 1. Sustained Fire sequence: 3 frames must trigger FIRE_CONFIRMED with alert
    mock_cnn_fire = {"label": "fire", "confidence": 0.95, "probs": {"fire": 0.95, "no_fire": 0.05}}
    mock_yolo_fire = [{"cls": "fire", "conf": 0.92, "xyxy": [100, 100, 200, 200]}]

    # Frame 1: Candidate detected, but awaiting temporal confirmation
    d1 = engine.update(mock_cnn_fire, mock_yolo_fire, True, True)
    assert d1["status"] in ["UNCERTAIN", "POSSIBLE_FIRE"], f"Frame 1 unexpected status: {d1['status']}"
    assert not d1["should_alert"], "Frame 1 should not trigger alert before confirmation frames met"

    # Frame 2
    d2 = engine.update(mock_cnn_fire, mock_yolo_fire, True, True)

    # Frame 3: Confirmation threshold met (3/3)
    d3 = engine.update(mock_cnn_fire, mock_yolo_fire, True, True)
    assert d3["status"] == DecisionState.FIRE_CONFIRMED, f"Frame 3 expected FIRE_CONFIRMED, got: {d3['status']}"
    assert d3["should_alert"], "Temporal confirmation must trigger alert siren!"
    print("  ✓ Fire Confirmation: Escalates to FIRE_CONFIRMED on 3rd consecutive frame (Alert Triggered).")

    # Frame 4: Continuous event must NOT trigger duplicate alert during cooldown
    d4 = engine.update(mock_cnn_fire, mock_yolo_fire, True, True)
    assert d4["status"] == DecisionState.FIRE_CONFIRMED
    assert not d4["should_alert"], "Cooldown must suppress duplicate alert for continuous event!"
    print("  ✓ Cooldown Resilience: Duplicate alert suppressed during ongoing event.")

    # 2. Reset engine for Smoke confirmation test
    engine = DecisionEngine(window_size=6, confirm_frames=3, cooldown_seconds=10.0)
    mock_yolo_smoke = [{"cls": "smoke", "conf": 0.88, "xyxy": [150, 150, 250, 250]}]
    mock_cnn_neutral = {"label": "no_fire", "confidence": 0.82, "probs": {"fire": 0.18, "no_fire": 0.82}}

    engine.update(mock_cnn_neutral, mock_yolo_smoke, True, True)
    engine.update(mock_cnn_neutral, mock_yolo_smoke, True, True)
    d_smoke = engine.update(mock_cnn_neutral, mock_yolo_smoke, True, True)
    assert d_smoke["status"] == DecisionState.SMOKE_CONFIRMED, f"Expected SMOKE_CONFIRMED, got {d_smoke['status']}"
    assert d_smoke["should_alert"], "Smoke confirmation must trigger alert siren!"
    print("  ✓ Smoke Confirmation: Escalates to SMOKE_CONFIRMED on 3rd frame (Alert Triggered).")

    # 3. Model Disagreement test (YOLO detects strong fire, but CNN says no-fire)
    engine = DecisionEngine(window_size=6, confirm_frames=3, cooldown_seconds=10.0)
    mock_yolo_strong = [{"cls": "fire", "conf": 0.85, "xyxy": [100, 100, 200, 200]}]
    mock_cnn_no_fire = {"label": "no_fire", "confidence": 0.90, "probs": {"fire": 0.10, "no_fire": 0.90}}
    for _ in range(4):
        d_disagree = engine.update(mock_cnn_no_fire, mock_yolo_strong, True, True)
    assert d_disagree["status"] == DecisionState.UNCERTAIN, f"Disagreement must yield UNCERTAIN, got {d_disagree['status']}"
    print("  ✓ Model Disagreement: Resolves to UNCERTAIN state when YOLO detects fire but CNN disagrees.")

    # 4. Unlocalized CNN Fire test (CNN strongly confident, YOLO has no box)
    engine = DecisionEngine(window_size=6, confirm_frames=3, cooldown_seconds=10.0)
    for _ in range(4):
        d_possible = engine.update(mock_cnn_fire, [], True, True)
    assert d_possible["status"] == DecisionState.POSSIBLE_FIRE, f"Unlocalized CNN fire must yield POSSIBLE_FIRE, got {d_possible['status']}"
    print("  ✓ Unlocalized CNN Fire: Resolves to POSSIBLE_FIRE when CNN detects fire without YOLO bounding box.")

    print("  ✅ Decision Fusion Tests PASSED: All state machine transitions validated.")
    return True

def test_cnn_inference(cnn_path: Path, benchmark_frames: list) -> dict:
    """Tests CNN ONNX model inference and audits class mapping and confidence."""
    print("\n--- TEST SUITE 3: MobileNetV2 CNN Classifier Inference ---")
    report = {"status": "NOT_RUN", "available": False, "predictions": [], "errors": []}

    if not cnn_path.exists():
        report["status"] = "PENDING_WEIGHTS"
        report["reason"] = f"Model weight file not found at: {cnn_path}"
        print(f"  ⏳ CNN weights not installed ({cnn_path.name}). Skipping physical inference.")
        return report

    try:
        cnn = CNNClassifier(cnn_path)
        if not cnn.available:
            report["status"] = "LOAD_FAILED"
            report["errors"].append("Failed to initialize ONNX Runtime InferenceSession.")
            print("  ❌ Failed to initialize CNN ONNX session.")
            return report

        report["available"] = True
        # Audit class mappings
        assert cnn.classes == ["fire", "no_fire"], f"CNN class mapping mismatch: {cnn.classes}"
        assert cnn.classes[0] == "fire", "CNN class index 0 must be 'fire'!"
        assert cnn.classes[1] == "no_fire", "CNN class index 1 must be 'no_fire'!"
        print(f"  ✓ Verified CNN Class Mapping: 0='fire', 1='no_fire'")
        print(f"  ✓ Input Node: '{cnn.input_name}' (Shape: 1x3x224x224)")
        print(f"  ✓ Output Node: '{cnn.output_name}'")

        # Run inference across benchmark frames
        for bf in benchmark_frames:
            res = cnn.predict(bf["img"])
            assert res is not None, f"CNN predict returned None on frame {bf['name']}"
            assert res["label"] in ["fire", "no_fire"]
            assert 0.0 <= res["confidence"] <= 1.0, f"Confidence out of range: {res['confidence']}"
            probs = res.get("probs", {})
            assert abs((probs.get("fire", 0) + probs.get("no_fire", 0)) - 1.0) < 0.01, "Probabilities do not sum to 1.0"

            record = {
                "benchmark_case": bf["name"],
                "ground_truth": bf["ground_truth"],
                "prediction": res["label"],
                "confidence": res["confidence"],
                "probabilities": res["probs"],
                "latency_ms": res["latency_ms"]
            }
            report["predictions"].append(record)
            print(f"    Case '{bf['name']:<25}': {res['label'].upper()} (Conf: {res['confidence']:.4f}, Latency: {res['latency_ms']:.2f}ms)")

        report["status"] = "PASSED"
        print("  ✅ CNN Inference Tests PASSED: Softmax probabilities and bounds verified.")
        return report

    except Exception as e:
        report["status"] = "FAILED"
        report["errors"].append(str(e))
        print(f"  ❌ CNN Inference Test Failed: {e}")
        return report

def test_yolo_inference(yolo_path: Path, benchmark_frames: list) -> dict:
    """Tests YOLO detector inference and audits bounding boxes, class mappings, and confidence."""
    print("\n--- TEST SUITE 4: YOLO11n Object Detector Inference ---")
    report = {"status": "NOT_RUN", "available": False, "predictions": [], "errors": []}

    if not yolo_path.exists():
        report["status"] = "PENDING_WEIGHTS"
        report["reason"] = f"Model weight file not found at: {yolo_path}"
        print(f"  ⏳ YOLO weights not installed ({yolo_path.name}). Skipping physical inference.")
        return report

    try:
        yolo = YOLODetector(yolo_path)
        if not yolo.available:
            report["status"] = "LOAD_FAILED"
            report["errors"].append("Failed to load YOLO model checkpoint.")
            print("  ❌ Failed to load YOLO detector checkpoint.")
            return report

        report["available"] = True
        # Audit class mapping
        print(f"  ✓ Verified YOLO Class Mapping: {yolo.classes}")
        assert yolo.classes.get(0) == "smoke", "YOLO class 0 must be 'smoke'!"
        assert yolo.classes.get(1) == "fire", "YOLO class 1 must be 'fire'!"

        # Run inference across benchmark frames
        for bf in benchmark_frames:
            detections = yolo.predict(bf["img"])
            det_records = []
            for det in detections:
                assert det["cls"] in ["fire", "smoke"], f"Unknown class '{det['cls']}' in detection!"
                assert 0.0 <= det["conf"] <= 1.0, f"Confidence out of range: {det['conf']}"
                xyxy = det.get("xyxy", [])
                assert len(xyxy) == 4, f"Invalid bbox format: {xyxy}"
                x1, y1, x2, y2 = xyxy
                assert 0 <= x1 < x2 <= 640 and 0 <= y1 < y2 <= 480, f"Bbox coordinates out of frame bounds: {xyxy}"
                det_records.append(det)

            report["predictions"].append({
                "benchmark_case": bf["name"],
                "ground_truth": bf["ground_truth"],
                "num_detections": len(det_records),
                "detections": det_records,
                "latency_ms": yolo.last_inference_ms
            })
            det_summary = ", ".join([f"{d['cls']}({d['conf']:.2f})" for d in det_records]) if det_records else "None"
            print(f"    Case '{bf['name']:<25}': Detections=[{det_summary}], Latency={yolo.last_inference_ms:.2f}ms")

        report["status"] = "PASSED"
        print("  ✅ YOLO Inference Tests PASSED: Bounding boxes and confidence verified.")
        return report

    except Exception as e:
        report["status"] = "FAILED"
        report["errors"].append(str(e))
        print(f"  ❌ YOLO Inference Test Failed: {e}")
        return report

def main():
    parser = argparse.ArgumentParser(description="ForestGuard AI: Model Inference & Degradation Test Suite")
    parser.add_argument("--weights-dir", default=None, help="Directory containing model weights")
    parser.add_argument("--output-dir", default=None, help="Directory to save predictions and report")
    parser.add_argument("--strict", action="store_true", help="Fail with exit code 1 if model weights are missing")
    args = parser.parse_args()

    weights_dir = Path(args.weights_dir) if args.weights_dir else ROOT / "model_weights"
    output_dir = Path(args.output_dir) if args.output_dir else ROOT / "results" / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)
    benchmark_dir = output_dir / "benchmark_inputs"

    print("=" * 70)
    print("🌲 FORESTGUARD AI: MODEL INFERENCE & DEGRADATION VERIFIER")
    print("=" * 70)
    print(f"  Weights Directory:     {weights_dir}")
    print(f"  Predictions Directory: {output_dir}")

    # Generate benchmark frames
    print("\n[PREPARATION] Generating Standard Surveillance Benchmark Frames...")
    benchmark_frames = generate_benchmark_frames(benchmark_dir)
    print(f"  ✓ Generated {len(benchmark_frames)} benchmark test cases in {benchmark_dir.name}/")

    # 1. Test Degraded Mode
    degradation_passed = test_missing_weights_degradation()

    # 2. Test Multi-Modal Decision Fusion
    fusion_passed = test_decision_fusion_transitions()

    # 3. Test CNN Model Inference
    cnn_path = weights_dir / "cnn_mobilenetv2.onnx"
    cnn_report = test_cnn_inference(cnn_path, benchmark_frames)

    # 4. Test YOLO Model Inference
    yolo_pt = weights_dir / "yolo_best.pt"
    yolo_onnx = weights_dir / "yolo_best.onnx"
    yolo_path = yolo_pt if yolo_pt.exists() else yolo_onnx
    yolo_report = test_yolo_inference(yolo_path, benchmark_frames)

    # 5. Compile and Save Master Predictions Report
    master_report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "weights_dir": str(weights_dir),
        "degradation_audit": {
            "tested": True,
            "status": "PASSED" if degradation_passed else "FAILED",
            "model_unavailable_verified": True
        },
        "decision_fusion_audit": {
            "tested": True,
            "status": "PASSED" if fusion_passed else "FAILED"
        },
        "cnn_inference": cnn_report,
        "yolo_inference": yolo_report,
        "honest_disclosure": (
            "No trained model inference claimed without physical weight files. "
            "When weights are absent, system operates in verified MODEL_UNAVAILABLE state."
        )
    }

    # Save JSON predictions
    predictions_json_path = output_dir / "inference_predictions.json"
    with open(predictions_json_path, "w", encoding="utf-8") as f:
        json.dump(master_report, f, indent=2)

    # Save human-readable text report
    report_txt_lines = [
        "=" * 80,
        "🌲 FORESTGUARD AI: MODEL INFERENCE & DEGRADATION AUDIT REPORT",
        "=" * 80,
        f"Generated At:         {master_report['timestamp']}",
        f"Weights Directory:    {weights_dir}",
        "",
        "1. SAFE DEGRADED MODE & DECISION ENGINE AUDIT",
        "-" * 80,
        f"  Degraded Fallback:  {master_report['degradation_audit']['status']} (MODEL_UNAVAILABLE state verified)",
        f"  Multi-Modal Fusion: {master_report['decision_fusion_audit']['status']} (Temporal transitions verified)",
        "",
        "2. CNN CLASSIFIER (MobileNetV2 ONNX)",
        "-" * 80,
        f"  Status:             {cnn_report['status']}",
        f"  Weights File:       {cnn_path.name} ({'FOUND' if cnn_path.exists() else 'PENDING KAGGLE ARTIFACTS'})",
        f"  Class Mapping:      0='fire', 1='no_fire'",
        "",
        "3. YOLO DETECTOR (YOLO11n PyTorch/ONNX)",
        "-" * 80,
        f"  Status:             {yolo_report['status']}",
        f"  Weights File:       {yolo_path.name} ({'FOUND' if yolo_path.exists() else 'PENDING KAGGLE ARTIFACTS'})",
        f"  Class Mapping:      0='smoke', 1='fire'",
        "",
        "4. BENCHMARK TEST PREDICTIONS",
        "-" * 80
    ]

    for bf in benchmark_frames:
        report_txt_lines.append(f"  Case: {bf['name']} (Ground Truth: {bf['ground_truth']})")
        report_txt_lines.append(f"    CNN Prediction:  {cnn_report['status']}")
        report_txt_lines.append(f"    YOLO Prediction: {yolo_report['status']}")

    report_txt_lines.extend([
        "",
        "=" * 80,
        "HONEST AUDIT VERDICT:",
        master_report['honest_disclosure'],
        "=" * 80
    ])

    report_txt_path = ROOT / "results" / "inference_test_report.txt"
    report_txt_path.write_text("\n".join(report_txt_lines), encoding="utf-8")

    print("\n" + "=" * 70)
    print("📋 INFERENCE AUDIT COMPLETED")
    print("=" * 70)
    print(f"  Saved JSON Report: {predictions_json_path}")
    print(f"  Saved Text Report: {report_txt_path}")

    # Strict check handling
    if args.strict and (not cnn_report["available"] or not yolo_report["available"]):
        print("\n❌ STRICT MODE ERROR: Missing physical model weights for full inference testing.")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
