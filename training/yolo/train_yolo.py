"""
YOLO Nano Fine-Tuning Script — ForestGuard AI
Fine-tunes YOLO11n on D-Fire dataset (fire & smoke classes),
evaluates on the held-out test split, and exports best.onnx.
"""

import os
import json
import argparse
import shutil
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Fine-tune YOLO Nano on D-Fire Dataset")
    parser.add_argument("--data", type=str, default="datasets/yolo/data.yaml", help="Path to data.yaml")
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="Base model weights")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--out_dir", type=str, default="model_weights", help="Target model weights directory")
    args = parser.parse_args()

    from ultralytics import YOLO

    print(f"Loading base YOLO model: {args.model}")
    model = YOLO(args.model)

    print("Starting YOLO Fine-Tuning...")
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=15,
        project="runs",
        name="forestguard",
        seed=0,
        device="0" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_pt_src = Path("runs/forestguard/weights/best.pt")

    if best_pt_src.exists():
        best_pt_target = out_dir / "yolo_best.pt"
        shutil.copy2(best_pt_src, best_pt_target)
        print(f"Copied best model weights to {best_pt_target}")

        # Export ONNX for portable CPU execution
        print("Exporting best model to ONNX (imgsz=416)...")
        best_model = YOLO(str(best_pt_target))
        best_model.export(format="onnx", imgsz=416, simplify=True)
        
        onnx_exported = best_pt_target.with_suffix(".onnx")
        if onnx_exported.exists():
            print(f"Exported ONNX model to {onnx_exported}")

    # Evaluate on test split
    print("Evaluating fine-tuned model on test split...")
    test_metrics = model.val(data=args.data, split="test", imgsz=416)

    # Save metrics JSON
    metrics_summary = {
        "model": "YOLO11 Nano Fine-Tuned",
        "mAP50": round(float(test_metrics.box.map50), 4),
        "mAP50-95": round(float(test_metrics.box.map), 4),
        "precision": round(float(test_metrics.box.mp), 4),
        "recall": round(float(test_metrics.box.mr), 4),
        "classes": {
            "fire": {
                "precision": round(float(test_metrics.box.p[0]), 4) if len(test_metrics.box.p) > 0 else 0.0,
                "recall": round(float(test_metrics.box.r[0]), 4) if len(test_metrics.box.r) > 0 else 0.0,
                "mAP50": round(float(test_metrics.box.ap50[0]), 4) if len(test_metrics.box.ap50) > 0 else 0.0
            },
            "smoke": {
                "precision": round(float(test_metrics.box.p[1]), 4) if len(test_metrics.box.p) > 1 else 0.0,
                "recall": round(float(test_metrics.box.r[1]), 4) if len(test_metrics.box.r) > 1 else 0.0,
                "mAP50": round(float(test_metrics.box.ap50[1]), 4) if len(test_metrics.box.ap50) > 1 else 0.0
            }
        }
    }

    metrics_path = Path("results/metrics/yolo_metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"YOLO metrics saved to {metrics_path}")

    # Copy curves and confusion matrix to results/graphs/
    runs_dir = Path("runs/forestguard")
    graphs_dir = Path("results/graphs")
    graphs_dir.mkdir(parents=True, exist_ok=True)

    for plot_name in ["results.png", "confusion_matrix.png", "F1_curve.png", "PR_curve.png"]:
        src_plot = runs_dir / plot_name
        if src_plot.exists():
            shutil.copy2(src_plot, graphs_dir / f"yolo_{plot_name}")
            print(f"Copied {plot_name} to results/graphs/yolo_{plot_name}")

if __name__ == "__main__":
    main()
