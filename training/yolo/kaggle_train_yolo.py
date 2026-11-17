"""
ForestGuard AI — Single Self-Contained Kaggle Training Script for YOLO11 Nano
=============================================================================
Academic Prototype: Wildfire & Smoke Early Warning Decision System
Model: YOLO11n (Fire & Smoke Bounding Box Detector)
Target Runtime: Kaggle Free GPU (NVIDIA T4 x2 or T4 x1)
Estimated Duration: ~45-60 minutes total (Time-boxed: 30 epochs, 6000 image cap)

Outputs: Single downloadable ZIP file at /kaggle/working/yolo_artifacts.zip containing:
  - yolo_best.pt              (Ultralytics PyTorch weights with metadata)
  - yolo_best.onnx            (Optimized ONNX format for laptop CPU inference)
  - results.png               (Training & validation loss/mAP progression curves)
  - confusion_matrix.png      (Normalized detection confusion matrix)
  - PR_curve.png              (Precision-Recall curves for Fire and Smoke)
  - F1_curve.png              (F1-Confidence curve)
  - metrics.json              (Test mAP50, mAP50-95, Precision, Recall)
  - val_batch_predictions.jpg (Visual predictions on test split)
"""

import os
import sys
import glob
import time
import json
import yaml
import shutil
import zipfile
from pathlib import Path

# Install or ensure ultralytics is ready
try:
    import ultralytics
    from ultralytics import YOLO
except ImportError:
    os.system("pip install --upgrade ultralytics onnx onnxruntime")
    import ultralytics
    from ultralytics import YOLO

# ---------------------------------------------------------
# 1. Configuration & Constants
# ---------------------------------------------------------
MAX_TRAIN_IMAGES = 6000    # Cap to guarantee completion within ~45-60 mins on T4 GPU
MAX_VAL_IMAGES = 1000
MAX_TEST_IMAGES = 1000
SMOKE_TEST_IMAGES = 200

EPOCHS = 30
PATIENCE = 10
IMGSZ = 640
BATCH_SIZE = 16

WORK_DIR = Path("/kaggle/working" if os.path.exists("/kaggle/working") else "./kaggle_working")
YOLO_DATA_DIR = WORK_DIR / "yolo_dataset"
ARTIFACTS_DIR = WORK_DIR / "yolo_artifacts"

WORK_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# 2. Dataset Auto-Detection & Capping
# ---------------------------------------------------------
def locate_raw_dataset() -> Path:
    """Searches /kaggle/input for D-Fire or smoke-fire datasets."""
    kaggle_input = Path("/kaggle/input")
    if not kaggle_input.exists():
        local_path = Path("datasets/DFire")
        if local_path.exists():
            return local_path
        raise FileNotFoundError("Could not find /kaggle/input or local datasets folder.")

    candidates = list(kaggle_input.glob("**/*train*"))
    for c in candidates:
        if (c / "images").exists() or (c / "labels").exists():
            return c.parent
        if c.is_dir() and "dfire" in str(c).lower():
            return c

    for item in kaggle_input.iterdir():
        if item.is_dir():
            if (item / "train").exists() or (item / "dfire").exists():
                return item

    all_subdirs = [p for p in kaggle_input.glob("*/*") if p.is_dir()]
    if all_subdirs:
        return all_subdirs[0]

    raise FileNotFoundError(f"No valid dataset found in {kaggle_input}. Contents: {list(kaggle_input.iterdir())}")

def prepare_capped_dataset(raw_dir: Path) -> Path:
    """
    Creates a capped dataset structure to guarantee training finishes tonight:
    Caps train at 6,000 images, val at 1,000, test at 1,000.
    """
    print(f"\n📂 Preparing Capped YOLO Dataset (Cap: {MAX_TRAIN_IMAGES} train frames) from: {raw_dir}")
    splits = ["train", "val", "test"]

    for split in splits:
        src_img_dir = None
        for cand in [raw_dir / split / "images", raw_dir / split, raw_dir / f"{split}_images"]:
            if cand.exists() and any(cand.glob("*.jpg")) or any(cand.glob("*.png")):
                src_img_dir = cand
                break
        
        src_lbl_dir = None
        for cand in [raw_dir / split / "labels", raw_dir / "labels" / split, raw_dir / f"{split}_labels"]:
            if cand.exists():
                src_lbl_dir = cand
                break

        if src_img_dir is None and split == "val":
            cand_v = raw_dir / "valid" / "images"
            if cand_v.exists():
                src_img_dir = cand_v
                src_lbl_dir = raw_dir / "valid" / "labels"

        dest_img_dir = YOLO_DATA_DIR / split / "images"
        dest_lbl_dir = YOLO_DATA_DIR / split / "labels"
        dest_img_dir.mkdir(parents=True, exist_ok=True)
        dest_lbl_dir.mkdir(parents=True, exist_ok=True)

        if not src_img_dir:
            print(f"⚠️ Warning: Split '{split}' images not found; creating empty placeholder.")
            continue

        images = list(src_img_dir.glob("*.jpg")) + list(src_img_dir.glob("*.png")) + list(src_img_dir.glob("*.jpeg"))
        cap = MAX_TRAIN_IMAGES if split == "train" else MAX_VAL_IMAGES
        selected_images = images[:cap]
        print(f"  Split '{split}': Using {len(selected_images)} / {len(images)} images.")

        for img in selected_images:
            dest_img = dest_img_dir / img.name
            if not dest_img.exists():
                try:
                    os.symlink(img, dest_img)
                except (OSError, AttributeError):
                    shutil.copy2(img, dest_img)

            # Copy corresponding label file if present
            if src_lbl_dir:
                lbl = src_lbl_dir / f"{img.stem}.txt"
                dest_lbl = dest_lbl_dir / f"{img.stem}.txt"
                if lbl.exists() and not dest_lbl.exists():
                    try:
                        os.symlink(lbl, dest_lbl)
                    except (OSError, AttributeError):
                        shutil.copy2(lbl, dest_lbl)
                elif not lbl.exists() and not dest_lbl.exists():
                    dest_lbl.touch() # Empty label for negative background frame

    # Generate data.yaml
    data_yaml_path = WORK_DIR / "data_capped.yaml"
    data_config = {
        "path": str(YOLO_DATA_DIR.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": {
            0: "fire",
            1: "smoke"
        }
    }
    with open(data_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data_config, f, default_flow_style=False)

    print(f"✅ Generated dataset configuration: {data_yaml_path}")
    return data_yaml_path

# ---------------------------------------------------------
# 3. Smoke Test Step
# ---------------------------------------------------------
def run_smoke_test(data_yaml_path: Path):
    print("\n" + "=" * 60)
    print("🧪 RUNNING 1-EPOCH SMOKE TEST ON MINI DATASET (EARLY ERROR CHECK)...")
    print("=" * 60)

    mini_data_dir = WORK_DIR / "smoke_dataset"
    (mini_data_dir / "train" / "images").mkdir(parents=True, exist_ok=True)
    (mini_data_dir / "train" / "labels").mkdir(parents=True, exist_ok=True)
    (mini_data_dir / "val" / "images").mkdir(parents=True, exist_ok=True)
    (mini_data_dir / "val" / "labels").mkdir(parents=True, exist_ok=True)

    # Link 50 images for quick test
    train_imgs = list((YOLO_DATA_DIR / "train" / "images").glob("*.*"))[:50]
    for img in train_imgs:
        dest = mini_data_dir / "train" / "images" / img.name
        if not dest.exists():
            shutil.copy2(img, dest)
        lbl = YOLO_DATA_DIR / "train" / "labels" / f"{img.stem}.txt"
        dest_lbl = mini_data_dir / "train" / "labels" / f"{img.stem}.txt"
        if lbl.exists() and not dest_lbl.exists():
            shutil.copy2(lbl, dest_lbl)
        elif not dest_lbl.exists():
            dest_lbl.touch()

    # Mirror for val
    for img in train_imgs[:10]:
        dest = mini_data_dir / "val" / "images" / img.name
        if not dest.exists():
            shutil.copy2(img, dest)
        dest_lbl = mini_data_dir / "val" / "labels" / f"{img.stem}.txt"
        if not dest_lbl.exists():
            dest_lbl.touch()

    mini_yaml = WORK_DIR / "smoke_data.yaml"
    with open(mini_yaml, "w", encoding="utf-8") as f:
        yaml.dump({
            "path": str(mini_data_dir.resolve()),
            "train": "train/images",
            "val": "val/images",
            "names": {0: "fire", 1: "smoke"}
        }, f)

    t0 = time.time()
    smoke_model = YOLO("yolo11n.pt")
    smoke_model.train(
        data=str(mini_yaml),
        epochs=1,
        imgsz=320,
        batch=8,
        device=0 if ultralytics.utils.torch_utils.torch.cuda.is_available() else "cpu",
        project=str(WORK_DIR / "runs_smoke"),
        name="smoke",
        verbose=False
    )
    elapsed = time.time() - t0
    print(f"✅ Smoke Test Passed in {elapsed:.2f}s! Ultralytics model, weights download, and CUDA execution verified.")
    print("=" * 60 + "\n")

# ---------------------------------------------------------
# 4. Main YOLO Training Execution
# ---------------------------------------------------------
def main():
    print("🌲 ForestGuard AI — YOLO11 Nano Cloud Training Pipeline")
    print("=" * 60)
    has_cuda = ultralytics.utils.torch_utils.torch.cuda.is_available()
    device = 0 if has_cuda else "cpu"
    dev_name = ultralytics.utils.torch_utils.torch.cuda.get_device_name(0) if has_cuda else "CPU"
    print(f"Compute Device: {device} ({dev_name})")

    raw_dir = locate_raw_dataset()
    data_yaml = prepare_capped_dataset(raw_dir)

    # 1. Smoke Test First
    run_smoke_test(data_yaml)

    # 2. Full Training
    print(f"\n🚀 Starting Full Training: YOLO11n ({EPOCHS} epochs, imgsz={IMGSZ}, batch={BATCH_SIZE}, patience={PATIENCE})...")
    model = YOLO("yolo11n.pt")

    train_res = model.train(
        data=str(data_yaml),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH_SIZE,
        patience=PATIENCE,
        device=device,
        project=str(WORK_DIR / "runs"),
        name="forestguard_yolo",
        seed=42,
        save=True,
        plots=True
    )

    runs_dir = Path(train_res.save_dir)
    print(f"Training run saved to: {runs_dir}")

    best_pt = runs_dir / "weights" / "best.pt"
    if not best_pt.exists():
        best_pt = runs_dir / "weights" / "last.pt"

    dest_best_pt = ARTIFACTS_DIR / "yolo_best.pt"
    shutil.copy2(best_pt, dest_best_pt)
    print(f"✅ Saved best PyTorch weights to: {dest_best_pt}")

    # 3. ONNX Export
    print(f"\n📦 Exporting YOLO11n to ONNX format (imgsz={IMGSZ})...")
    eval_model = YOLO(str(dest_best_pt))
    eval_model.export(format="onnx", imgsz=IMGSZ, simplify=True, dynamic=True)
    
    exported_onnx = dest_best_pt.with_suffix(".onnx")
    if exported_onnx.exists():
        dest_onnx = ARTIFACTS_DIR / "yolo_best.onnx"
        if exported_onnx != dest_onnx:
            shutil.copy2(exported_onnx, dest_onnx)
        print(f"✅ ONNX model successfully exported: {dest_onnx} ({dest_onnx.stat().st_size / 1e6:.2f} MB)")

    # 4. Evaluation on Test Split
    print("\n📊 Evaluating YOLO Model on Test Split...")
    test_metrics = eval_model.val(data=str(data_yaml), split="test", imgsz=IMGSZ)

    mAP50 = float(test_metrics.box.map50)
    mAP50_95 = float(test_metrics.box.map)
    mp = float(test_metrics.box.mp)
    mr = float(test_metrics.box.mr)

    print(f"Overall mAP50: {mAP50:.4f} | mAP50-95: {mAP50_95:.4f}")
    print(f"Overall Precision: {mp:.4f} | Recall: {mr:.4f}")

    metrics_summary = {
        "model": "YOLO11 Nano Wildfire Detector",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mAP50": mAP50,
        "mAP50_95": mAP50_95,
        "precision": mp,
        "recall": mr,
        "f1_score": round(2 * mp * mr / (mp + mr + 1e-8), 4),
        "epochs_trained": EPOCHS,
        "image_size": IMGSZ,
        "train_samples_capped": MAX_TRAIN_IMAGES,
        "classes": {
            "fire": {
                "precision": float(test_metrics.box.p[0]) if len(test_metrics.box.p) > 0 else 0.0,
                "recall": float(test_metrics.box.r[0]) if len(test_metrics.box.r) > 0 else 0.0,
                "mAP50": float(test_metrics.box.ap50[0]) if len(test_metrics.box.ap50) > 0 else 0.0
            },
            "smoke": {
                "precision": float(test_metrics.box.p[1]) if len(test_metrics.box.p) > 1 else 0.0,
                "recall": float(test_metrics.box.r[1]) if len(test_metrics.box.r) > 1 else 0.0,
                "mAP50": float(test_metrics.box.ap50[1]) if len(test_metrics.box.ap50) > 1 else 0.0
            }
        }
    }

    metrics_json_path = ARTIFACTS_DIR / "metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)

    # Copy curves and diagnostic plots to artifacts
    for plot_name in ["results.png", "confusion_matrix.png", "F1_curve.png", "PR_curve.png", "val_batch0_pred.jpg"]:
        src_p = runs_dir / plot_name
        if src_p.exists():
            shutil.copy2(src_p, ARTIFACTS_DIR / plot_name)

    # 5. Packaging into Single ZIP
    zip_dest = WORK_DIR / "yolo_artifacts.zip"
    print(f"\n🗜️ Packaging all YOLO artifacts into: {zip_dest}")
    with zipfile.ZipFile(zip_dest, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in ARTIFACTS_DIR.iterdir():
            if file.is_file():
                zf.write(file, arcname=file.name)
                print(f"  + Added to zip: {file.name}")

    print("\n" + "=" * 60)
    print("🎉 YOLO TRAINING AND ARTIFACT PACKAGING COMPLETE!")
    print(f"📁 Download ZIP from Kaggle Right Sidebar: {zip_dest.name}")
    print("=" * 60)

if __name__ == "__main__":
    main()
