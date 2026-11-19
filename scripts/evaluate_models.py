"""
ForestGuard AI — Comprehensive Model Evaluation Driver
======================================================
Academic Prototype: Early Wildfire & Smoke Detection Decision System
Evaluates genuine trained CNN and YOLO models on real held-out test splits.
Calculates precision, recall, F1, mAP50, mAP50-95, and inference latency.
Generates genuine confusion matrices and evaluation graphs.
Never substitutes synthetic benchmark results for real test-dataset performance.
"""

import sys
import os
import time
import json
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

def evaluate_cnn_test_set(cnn_path: Path, test_dir: Path, graphs_dir: Path) -> dict:
    """Evaluates CNN on actual held-out test split (fire / no_fire subdirectories)."""
    if not cnn_path.exists():
        return {"status": "SKIPPED", "reason": f"CNN weights not found at {cnn_path}"}
    if not test_dir.exists():
        return {"status": "SKIPPED", "reason": f"CNN test split not found at {test_dir}"}

    from backend.models.cnn_classifier import CNNClassifier
    classifier = CNNClassifier(cnn_path)
    if not classifier.available:
        return {"status": "FAILED", "reason": "Failed to load CNN model session"}

    fire_dir = test_dir / "fire"
    nofire_dir = test_dir / "no_fire"
    if not fire_dir.exists() or not nofire_dir.exists():
        return {"status": "SKIPPED", "reason": f"Expected fire/ and no_fire/ subdirectories in {test_dir}"}

    fire_images = list(fire_dir.glob("*.[jJ][pP][gG]")) + list(fire_dir.glob("*.[pP][nN][gG]"))
    nofire_images = list(nofire_dir.glob("*.[jJ][pP][gG]")) + list(nofire_dir.glob("*.[pP][nN][gG]"))

    total_samples = len(fire_images) + len(nofire_images)
    if total_samples == 0:
        return {"status": "SKIPPED", "reason": f"Zero test images found in {test_dir}"}

    print(f"\n--- Evaluating CNN on Genuine Test Split ({total_samples} samples) ---")
    tp = fp = tn = fn = 0
    latencies = []

    # True Fire (label 0)
    for img_p in fire_images:
        img = cv2.imread(str(img_p))
        if img is None: continue
        t0 = time.perf_counter()
        res = classifier.predict(img)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        if res and res["label"] == "fire":
            tp += 1
        else:
            fn += 1

    # True No-Fire (label 1)
    for img_p in nofire_images:
        img = cv2.imread(str(img_p))
        if img is None: continue
        t0 = time.perf_counter()
        res = classifier.predict(img)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        if res and res["label"] == "no_fire":
            tn += 1
        else:
            fp += 1

    total_eval = tp + fp + tn + fn
    if total_eval == 0:
        return {"status": "FAILED", "reason": "No valid images could be evaluated."}

    accuracy = (tp + tn) / total_eval
    prec_fire = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec_fire = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_fire = (2 * prec_fire * rec_fire) / (prec_fire + rec_fire) if (prec_fire + rec_fire) > 0 else 0.0

    prec_nofire = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    rec_nofire = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_nofire = (2 * prec_nofire * rec_nofire) / (prec_nofire + rec_nofire) if (prec_nofire + rec_nofire) > 0 else 0.0

    macro_f1 = (f1_fire + f1_nofire) / 2.0
    avg_latency = float(np.mean(latencies)) if latencies else 0.0

    # Generate Confusion Matrix graph
    graphs_dir.mkdir(parents=True, exist_ok=True)
    cm_path = graphs_dir / "cnn_confusion_matrix.png"
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cm = np.array([[tp, fn], [fp, tn]])
        fig, ax = plt.subplots(figsize=(6, 5))
        cax = ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.8)
        fig.colorbar(cax)
        for i in range(2):
            for j in range(2):
                ax.text(x=j, y=i, s=str(cm[i, j]), va="center", ha="center", size="xx-large")
        plt.xlabel("Predicted Label", fontsize=12)
        plt.ylabel("True Label", fontsize=12)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["fire", "no_fire"])
        ax.set_yticklabels(["fire", "no_fire"])
        plt.title(f"MobileNetV2 Test Confusion Matrix\nAccuracy: {accuracy*100:.2f}% | Macro F1: {macro_f1:.4f}", fontsize=12)
        plt.tight_layout()
        plt.savefig(str(cm_path), dpi=200)
        plt.close()
    except Exception as e:
        print(f"⚠️ Notice: Confusion matrix plot skipped: {e}")

    metrics = {
        "model": "MobileNetV2",
        "test_samples": total_eval,
        "test_accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "latency_ms": round(avg_latency, 2),
        "confusion_matrix": {"TP": tp, "FP": fp, "TN": tn, "FN": fn},
        "classes": {
            "fire": {
                "precision": round(prec_fire, 4),
                "recall": round(rec_fire, 4),
                "f1": round(f1_fire, 4)
            },
            "no_fire": {
                "precision": round(prec_nofire, 4),
                "recall": round(rec_nofire, 4),
                "f1": round(f1_nofire, 4)
            }
        }
    }
    return {"status": "PASSED", "metrics": metrics}

def evaluate_yolo_test_set(yolo_path: Path, data_yaml: Path, graphs_dir: Path) -> dict:
    """Evaluates YOLO on actual test split using Ultralytics validation engine."""
    if not yolo_path.exists():
        return {"status": "SKIPPED", "reason": f"YOLO weights not found at {yolo_path}"}
    if not data_yaml.exists():
        return {"status": "SKIPPED", "reason": f"YOLO data.yaml not found at {data_yaml}"}

    print(f"\n--- Evaluating YOLO on Genuine Test Split ({data_yaml.name}) ---")
    try:
        from ultralytics import YOLO
        model = YOLO(str(yolo_path))
        val_results = model.val(
            data=str(data_yaml),
            split="test",
            device="cpu",
            plots=True,
            verbose=False
        )

        metrics = {
            "model": "YOLO11n",
            "mAP50": round(float(val_results.results_dict.get("metrics/mAP50(B)", 0.0)), 4),
            "mAP50-95": round(float(val_results.results_dict.get("metrics/mAP50-95(B)", 0.0)), 4),
            "precision": round(float(val_results.results_dict.get("metrics/precision(B)", 0.0)), 4),
            "recall": round(float(val_results.results_dict.get("metrics/recall(B)", 0.0)), 4),
            "fitness": round(float(val_results.fitness), 4) if hasattr(val_results, "fitness") else 0.0,
            "classes": {
                "fire": {"class_id": 0},
                "smoke": {"class_id": 1}
            }
        }

        # Copy generated plots if present
        if hasattr(val_results, "save_dir"):
            sdir = Path(val_results.save_dir)
            for p in ["results.png", "confusion_matrix.png", "PR_curve.png", "F1_curve.png"]:
                src = sdir / p
                if src.exists():
                    graphs_dir.mkdir(parents=True, exist_ok=True)
                    dest = graphs_dir / f"yolo_{p}"
                    import shutil
                    shutil.copy2(src, dest)

        return {"status": "PASSED", "metrics": metrics}

    except Exception as e:
        return {"status": "FAILED", "reason": str(e)}

def main():
    parser = argparse.ArgumentParser(description="ForestGuard AI: Genuine Model Evaluation")
    parser.add_argument("--weights-dir", default="model_weights", help="Directory with model weights")
    parser.add_argument("--data-dir", default="datasets", help="Directory with datasets")
    parser.add_argument("--output-dir", default="results", help="Directory to save metrics and graphs")
    parser.add_argument("--strict", action="store_true", help="Fail if weights or datasets are missing")
    args = parser.parse_args()

    weights_dir = ROOT / args.weights_dir
    data_dir = ROOT / args.data_dir
    metrics_dir = ROOT / args.output_dir / "metrics"
    graphs_dir = ROOT / args.output_dir / "graphs"

    metrics_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("🌲 FORESTGUARD AI: GENUINE MODEL EVALUATION DRIVER")
    print("=" * 70)
    print(f"Weights Directory: {weights_dir}")
    print(f"Data Directory:    {data_dir}")

    # 1. Evaluate CNN
    cnn_path = weights_dir / "cnn_mobilenetv2.onnx"
    if not cnn_path.exists():
        cnn_path = weights_dir / "cnn_mobilenetv2.pt"
    cnn_test_dir = data_dir / "cnn" / "test"

    cnn_res = evaluate_cnn_test_set(cnn_path, cnn_test_dir, graphs_dir)
    if cnn_res["status"] == "PASSED":
        cnn_metrics_file = metrics_dir / "cnn_metrics.json"
        with open(cnn_metrics_file, "w", encoding="utf-8") as f:
            json.dump(cnn_res["metrics"], f, indent=2)
        print(f"  ✅ CNN Evaluation PASSED -> {cnn_metrics_file.name}")
        print(f"     Test Accuracy: {cnn_res['metrics']['test_accuracy']*100:.2f}% | Macro F1: {cnn_res['metrics']['macro_f1']}")
    else:
        print(f"  ℹ CNN Evaluation: {cnn_res['status']} ({cnn_res['reason']})")
        print("    Notice: Never substituting synthetic benchmark results for real test-dataset performance.")

    # 2. Evaluate YOLO
    yolo_path = weights_dir / "yolo_best.pt"
    yolo_yaml = data_dir / "yolo" / "data.yaml"

    yolo_res = evaluate_yolo_test_set(yolo_path, yolo_yaml, graphs_dir)
    if yolo_res["status"] == "PASSED":
        yolo_metrics_file = metrics_dir / "yolo_metrics.json"
        with open(yolo_metrics_file, "w", encoding="utf-8") as f:
            json.dump(yolo_res["metrics"], f, indent=2)
        print(f"  ✅ YOLO Evaluation PASSED -> {yolo_metrics_file.name}")
        print(f"     mAP@50: {yolo_res['metrics']['mAP50']} | Precision: {yolo_res['metrics']['precision']} | Recall: {yolo_res['metrics']['recall']}")
    else:
        print(f"  ℹ YOLO Evaluation: {yolo_res['status']} ({yolo_res['reason']})")
        print("    Notice: Never substituting synthetic benchmark results for real test-dataset performance.")

    # 3. Synchronize with IEEE paper if metrics were generated
    if cnn_res["status"] == "PASSED" and yolo_res["status"] == "PASSED":
        paper_sync = ROOT / "scripts" / "update_paper_metrics.py"
        if paper_sync.exists():
            import subprocess
            subprocess.run([sys.executable, str(paper_sync)])

    if args.strict and (cnn_res["status"] != "PASSED" or yolo_res["status"] != "PASSED"):
        print("\n❌ STRICT MODE: Required weights or test datasets missing.")
        return 1

    print("\n" + "=" * 70)
    print("✅ EVALUATION RUN COMPLETE")
    print("=" * 70)
    return 0

if __name__ == "__main__":
    sys.exit(main())
