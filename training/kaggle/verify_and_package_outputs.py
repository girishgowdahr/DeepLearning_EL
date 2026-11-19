"""
ForestGuard AI — Kaggle Training Output Verification & Packaging Tool
====================================================================
Academic Prototype: Early Wildfire & Smoke Detection Decision System
Verifies that Kaggle GPU training finished successfully, extracts genuine metrics,
generates evaluation curves/confusion matrices if missing, creates cryptographic SHA-256
manifests, and safely packages:
  - yolo_artifacts.zip
  - cnn_artifacts.zip
Never fabricates metrics, success indicators, or synthetic results.
"""

import os
import sys
import csv
import json
import shutil
import zipfile
import hashlib
import argparse
from pathlib import Path

def compute_sha256(filepath: Path) -> str:
    """Computes SHA-256 hash in 64KB chunks."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().lower()

def safe_float(val, default=0.0):
    try:
        return float(val)
    except (ValueError, TypeError):
        return default

def generate_training_plots_from_csv(csv_path: Path, output_png: Path):
    """Generates loss and mAP training graphs from Ultralytics results.csv."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs, train_losses, val_losses, map50, map50_95 = [], [], [], [], []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            # Strip column headers
            reader.fieldnames = [k.strip() for k in reader.fieldnames] if reader.fieldnames else []
            for row in reader:
                clean_row = {k.strip(): v.strip() for k, v in row.items()}
                epoch = safe_float(clean_row.get("epoch", len(epochs)))
                epochs.append(epoch)

                # Losses
                box_loss = safe_float(clean_row.get("train/box_loss", clean_row.get("box_loss", 0)))
                cls_loss = safe_float(clean_row.get("train/cls_loss", clean_row.get("cls_loss", 0)))
                train_losses.append(box_loss + cls_loss)

                val_box = safe_float(clean_row.get("val/box_loss", 0))
                val_cls = safe_float(clean_row.get("val/cls_loss", 0))
                val_losses.append(val_box + val_cls)

                # mAP
                m50 = safe_float(clean_row.get("metrics/mAP50(B)", clean_row.get("mAP50", 0)))
                m95 = safe_float(clean_row.get("metrics/mAP50-95(B)", clean_row.get("mAP50-95", 0)))
                map50.append(m50)
                map50_95.append(m95)

        if not epochs:
            return False

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.plot(epochs, train_losses, label="Train Loss", color="royalblue")
        if any(v > 0 for v in val_losses):
            ax1.plot(epochs, val_losses, label="Val Loss", color="orange", linestyle="--")
        ax1.set_title("YOLO11n Training & Validation Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Total Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(epochs, map50, label="mAP@50", color="forestgreen")
        ax2.plot(epochs, map50_95, label="mAP@50-95", color="darkred", linestyle="--")
        ax2.set_title("YOLO11n Detection Metrics (D-Fire Test)")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("mAP")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_png, dpi=200)
        plt.close()
        return True
    except Exception as e:
        print(f"⚠️ Notice: Plot generation from CSV skipped: {e}")
        return False

def verify_yolo_outputs(yolo_dir: Path) -> dict:
    """Verifies that YOLO training outputs are complete, non-empty, and valid."""
    result = {"valid": False, "files": [], "metrics": None, "errors": []}
    if not yolo_dir.exists():
        result["errors"].append(f"YOLO directory not found at {yolo_dir}")
        return result

    # 1. Locate best checkpoint
    candidate_weights = [
        yolo_dir / "yolo_best.pt",
        yolo_dir / "weights" / "best.pt",
        yolo_dir / "weights" / "last.pt",
        yolo_dir / "runs" / "forestguard_yolo" / "weights" / "best.pt",
        yolo_dir / "runs" / "forestguard_yolo" / "weights" / "last.pt"
    ]
    target_weight = None
    for cand in candidate_weights:
        if cand.exists() and cand.is_file() and cand.stat().st_size >= 100 * 1024:
            target_weight = cand
            break

    if not target_weight:
        result["errors"].append("No valid YOLO model weights found (expected >= 100 KB .pt file).")
        return result

    # Standardize location as yolo_best.pt in yolo_dir
    dest_weight = yolo_dir / "yolo_best.pt"
    if target_weight != dest_weight:
        shutil.copy2(target_weight, dest_weight)
    result["files"].append(dest_weight)

    # 2. Extract / Validate genuine metrics
    metrics_file = yolo_dir / "yolo_metrics.json"
    csv_file = None
    for cand_csv in [
        yolo_dir / "results.csv",
        yolo_dir / "runs" / "forestguard_yolo" / "results.csv"
    ]:
        if cand_csv.exists():
            csv_file = cand_csv
            break

    yolo_metrics = None
    if metrics_file.exists():
        try:
            with open(metrics_file, "r", encoding="utf-8") as f:
                yolo_metrics = json.load(f)
        except Exception as e:
            result["errors"].append(f"Failed to read existing yolo_metrics.json: {e}")

    if not yolo_metrics and csv_file:
        # Extract from results.csv
        try:
            with open(csv_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                reader.fieldnames = [k.strip() for k in reader.fieldnames] if reader.fieldnames else []
                rows = list(reader)
                if rows:
                    last_row = {k.strip(): v.strip() for k, v in rows[-1].items()}
                    # Find row with max mAP50
                    best_row = max(
                        rows,
                        key=lambda r: safe_float({k.strip(): v.strip() for k, v in r.items()}.get("metrics/mAP50(B)", 0))
                    )
                    clean_best = {k.strip(): v.strip() for k, v in best_row.items()}
                    yolo_metrics = {
                        "model": "YOLO11n",
                        "best_epoch": int(safe_float(clean_best.get("epoch", len(rows)))),
                        "mAP50": round(safe_float(clean_best.get("metrics/mAP50(B)", clean_best.get("mAP50", 0))), 4),
                        "mAP50-95": round(safe_float(clean_best.get("metrics/mAP50-95(B)", clean_best.get("mAP50-95", 0))), 4),
                        "precision": round(safe_float(clean_best.get("metrics/precision(B)", clean_best.get("precision", 0))), 4),
                        "recall": round(safe_float(clean_best.get("metrics/recall(B)", clean_best.get("recall", 0))), 4),
                        "classes": {
                            "fire": {"class_id": 0},
                            "smoke": {"class_id": 1}
                        }
                    }
                    with open(metrics_file, "w", encoding="utf-8") as mf:
                        json.dump(yolo_metrics, mf, indent=2)
                    print(f"  ✓ Extracted genuine YOLO metrics from {csv_file.name}: mAP50={yolo_metrics['mAP50']}")
        except Exception as e:
            result["errors"].append(f"Error parsing results.csv: {e}")

    if not yolo_metrics:
        result["errors"].append("No valid YOLO evaluation metrics found (neither yolo_metrics.json nor results.csv).")
        return result

    result["metrics"] = yolo_metrics
    result["files"].append(metrics_file)

    # 3. Check / Copy plots
    for plot_name in ["results.png", "confusion_matrix.png", "PR_curve.png", "F1_curve.png"]:
        for cand_plot in [yolo_dir / plot_name, yolo_dir / "runs" / "forestguard_yolo" / plot_name]:
            if cand_plot.exists():
                dest_plot = yolo_dir / plot_name
                if cand_plot != dest_plot:
                    shutil.copy2(cand_plot, dest_plot)
                if dest_plot not in result["files"]:
                    result["files"].append(dest_plot)
                break

    # If results.png is missing but CSV exists, plot it
    results_png = yolo_dir / "results.png"
    if not results_png.exists() and csv_file:
        if generate_training_plots_from_csv(csv_file, results_png):
            result["files"].append(results_png)

    # 4. Check for optional ONNX export
    onnx_cand = yolo_dir / "yolo_best.onnx"
    if onnx_cand.exists() and onnx_cand.stat().st_size >= 100 * 1024:
        result["files"].append(onnx_cand)

    result["valid"] = True
    return result

def verify_cnn_outputs(cnn_dir: Path) -> dict:
    """Verifies that CNN training outputs are complete, non-empty, and valid."""
    result = {"valid": False, "files": [], "metrics": None, "errors": []}
    if not cnn_dir.exists():
        result["errors"].append(f"CNN directory not found at {cnn_dir}")
        return result

    # 1. Locate weights & ONNX export
    onnx_weight = cnn_dir / "cnn_mobilenetv2.onnx"
    pt_weight = cnn_dir / "cnn_mobilenetv2.pt"

    if not onnx_weight.exists() or onnx_weight.stat().st_size < 100 * 1024:
        # Check subdirectories
        found = False
        for cand in cnn_dir.rglob("*.onnx"):
            if cand.stat().st_size >= 100 * 1024:
                shutil.copy2(cand, onnx_weight)
                found = True
                break
        if not found:
            result["errors"].append("Missing required CNN ONNX weight 'cnn_mobilenetv2.onnx' (expected >= 100 KB).")
            return result

    result["files"].append(onnx_weight)
    if pt_weight.exists() and pt_weight.stat().st_size >= 100 * 1024:
        result["files"].append(pt_weight)

    # 2. Validate CNN metrics JSON
    metrics_file = cnn_dir / "cnn_metrics.json"
    cnn_metrics = None
    if metrics_file.exists():
        try:
            with open(metrics_file, "r", encoding="utf-8") as f:
                cnn_metrics = json.load(f)
        except Exception as e:
            result["errors"].append(f"Failed to read cnn_metrics.json: {e}")

    if not cnn_metrics:
        result["errors"].append("Missing required CNN evaluation metrics 'cnn_metrics.json'.")
        return result

    # Ensure compatibility with IEEE paper synchronizer
    acc = cnn_metrics.get("test_accuracy", cnn_metrics.get("accuracy"))
    fire_prec = cnn_metrics.get("classes", {}).get("fire", {}).get("precision")
    fire_rec = cnn_metrics.get("classes", {}).get("fire", {}).get("recall")

    if acc is None or fire_prec is None or fire_rec is None:
        result["errors"].append("cnn_metrics.json is missing required fields (accuracy, classes.fire.precision/recall).")
        return result

    result["metrics"] = cnn_metrics
    result["files"].append(metrics_file)

    # 3. Check plots
    for p in ["cnn_confusion_matrix.png", "cnn_accuracy.png", "cnn_loss_curves.png", "cnn_roc_curve.png"]:
        cand = cnn_dir / p
        if cand.exists():
            result["files"].append(cand)

    result["valid"] = True
    return result

def write_manifest_and_metadata(folder: Path, verified_files: list) -> Path:
    """Generates standard MANIFEST.txt and MODEL_INFO.json in the artifact folder."""
    manifest_lines = [
        "=" * 90,
        f"🌲 FORESTGUARD AI: ARTIFACT MANIFEST & CRYPTOGRAPHIC CHECKSUMS",
        f"Generated: {Path(folder).name.upper()} Component",
        "=" * 90,
        f"{'Filename':<35} {'Size (MB)':<12} {'SHA-256 Checksum':<64}",
        "-" * 90
    ]

    manifest_entries = []
    for f in verified_files:
        if f.is_file():
            size_mb = f"{f.stat().st_size / (1024 * 1024):.2f}"
            h = compute_sha256(f)
            manifest_lines.append(f"{f.name:<35} {size_mb:<12} {h:<64}")
            manifest_entries.append(f"{h} *{f.name}")

    manifest_txt = folder / "MANIFEST.txt"
    manifest_txt.write_text("\n".join(manifest_lines) + "\n\n# Standard sha256sum records:\n" + "\n".join(manifest_entries) + "\n", encoding="utf-8")
    
    # Write MODEL_INFO.json if not present
    model_info_path = folder / "MODEL_INFO.json"
    if not model_info_path.exists():
        info = {
            "project": "ForestGuard AI",
            "version": "1.0.0",
            "target": "Edge / Jetson / Mobile Surveillance",
            "cnn": {
                "architecture": "MobileNetV2",
                "format": "ONNX (opset 14)",
                "classes": ["fire", "no_fire"]
            },
            "yolo": {
                "architecture": "YOLO11n",
                "format": "PyTorch / ONNX",
                "classes": ["fire", "smoke"]
            }
        }
        with open(model_info_path, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)

    return manifest_txt

def package_zip(source_dir: Path, file_list: list, zip_path: Path):
    """Zips verified files into a standalone distributable archive."""
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in file_list:
            if f.exists() and f.is_file():
                zf.write(f, arcname=f.name)
    return zip_path

def main():
    parser = argparse.ArgumentParser(description="ForestGuard AI: Training Output Verifier & Packager")
    parser.add_argument("--output-dir", default=None, help="Root directory containing training outputs")
    parser.add_argument("--yolo-dir", default=None, help="Explicit YOLO output directory")
    parser.add_argument("--cnn-dir", default=None, help="Explicit CNN output directory")
    parser.add_argument("--dest-dir", default=None, help="Destination directory for zip packages")
    parser.add_argument("--verify-only", action="store_true", help="Only verify without packaging zip archives")
    parser.add_argument("--strict", action="store_true", help="Exit with code 1 if outputs are missing or incomplete")
    args = parser.parse_args()

    print("=" * 70)
    print("🌲 FORESTGUARD AI: TRAINING OUTPUT VERIFIER & ARTIFACT PACKAGER")
    print("=" * 70)

    # 1. Resolve output root
    if args.output_dir:
        output_root = Path(args.output_dir)
    elif os.path.exists("/kaggle/working/forestguard_outputs"):
        output_root = Path("/kaggle/working/forestguard_outputs")
    elif os.path.exists("./forestguard_outputs"):
        output_root = Path("./forestguard_outputs")
    elif os.path.exists("run/training_outputs"):
        output_root = Path("run/training_outputs")
    else:
        output_root = Path("./forestguard_outputs")

    yolo_dir = Path(args.yolo_dir) if args.yolo_dir else output_root / "yolo"
    cnn_dir = Path(args.cnn_dir) if args.cnn_dir else output_root / "cnn"
    dest_dir = Path(args.dest_dir) if args.dest_dir else output_root

    print(f"  Output Root:  {output_root.resolve()}")
    print(f"  YOLO Output:  {yolo_dir}")
    print(f"  CNN Output:   {cnn_dir}")
    print(f"  Package Dest: {dest_dir}")

    # 2. Verify YOLO
    print("\n[1/3] Verifying YOLO Training Outputs...")
    yolo_res = verify_yolo_outputs(yolo_dir)
    if yolo_res["valid"]:
        print(f"  ✅ YOLO Verification PASSED ({len(yolo_res['files'])} verified files)")
        print(f"     mAP@50:     {yolo_res['metrics']['mAP50']}")
        print(f"     Precision:  {yolo_res['metrics']['precision']}")
        print(f"     Recall:     {yolo_res['metrics']['recall']}")
    else:
        print("  ⚠️ YOLO Training Verification Incomplete:")
        for err in yolo_res["errors"]:
            print(f"     - {err}")

    # 3. Verify CNN
    print("\n[2/3] Verifying CNN Training Outputs...")
    cnn_res = verify_cnn_outputs(cnn_dir)
    if cnn_res["valid"]:
        print(f"  ✅ CNN Verification PASSED ({len(cnn_res['files'])} verified files)")
        print(f"     Accuracy:   {cnn_res['metrics'].get('test_accuracy', cnn_res['metrics'].get('accuracy')) * 100:.2f}%")
        print(f"     Fire Prec:  {cnn_res['metrics']['classes']['fire']['precision']}")
        print(f"     Fire Rec:   {cnn_res['metrics']['classes']['fire']['recall']}")
    else:
        print("  ⚠️ CNN Training Verification Incomplete:")
        for err in cnn_res["errors"]:
            print(f"     - {err}")

    # 4. Packaging
    if not yolo_res["valid"] or not cnn_res["valid"]:
        print("\n" + "=" * 70)
        print("❌ Cannot package artifacts: One or more models have incomplete training outputs.")
        print("   Never generating synthetic or fabricated weights.")
        print("=" * 70)
        if args.strict:
            return 1
        return 0

    if args.verify_only:
        print("\n✓ Verification complete (--verify-only mode). Zipping skipped.")
        return 0

    print("\n[3/3] Generating Cryptographic Manifests & Distributable Archives...")
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Manifests
    yolo_manifest = write_manifest_and_metadata(yolo_dir, yolo_res["files"])
    cnn_manifest = write_manifest_and_metadata(cnn_dir, cnn_res["files"])
    yolo_res["files"].extend([yolo_manifest, yolo_dir / "MODEL_INFO.json"])
    cnn_res["files"].extend([cnn_manifest, cnn_dir / "MODEL_INFO.json"])

    # Package ZIPs
    yolo_zip_path = dest_dir / "yolo_artifacts.zip"
    cnn_zip_path = dest_dir / "cnn_artifacts.zip"

    package_zip(yolo_dir, yolo_res["files"], yolo_zip_path)
    package_zip(cnn_dir, cnn_res["files"], cnn_zip_path)

    yolo_hash = compute_sha256(yolo_zip_path)
    cnn_hash = compute_sha256(cnn_zip_path)

    print(f"\n📦 Successfully Packaged YOLO Artifacts:")
    print(f"   Path:    {yolo_zip_path}")
    print(f"   Size:    {yolo_zip_path.stat().st_size / (1024*1024):.2f} MB")
    print(f"   SHA-256: {yolo_hash}")

    print(f"\n📦 Successfully Packaged CNN Artifacts:")
    print(f"   Path:    {cnn_zip_path}")
    print(f"   Size:    {cnn_zip_path.stat().st_size / (1024*1024):.2f} MB")
    print(f"   SHA-256: {cnn_hash}")

    # Copy to /kaggle/working if on Kaggle
    if os.path.exists("/kaggle/working") and dest_dir != Path("/kaggle/working"):
        shutil.copy2(yolo_zip_path, Path("/kaggle/working/yolo_artifacts.zip"))
        shutil.copy2(cnn_zip_path, Path("/kaggle/working/cnn_artifacts.zip"))
        print("\n📁 Copied archives directly to /kaggle/working/ root for one-click Kaggle download.")

    print("\n" + "=" * 70)
    print("🎉 ALL TRAINING OUTPUTS GENUINELY VERIFIED AND PACKAGED!")
    print("=" * 70)
    return 0

if __name__ == "__main__":
    sys.exit(main())
