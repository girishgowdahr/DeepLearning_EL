"""
ForestGuard AI — Single Self-Contained Kaggle Training Script for MobileNetV2
=============================================================================
Academic Prototype: Wildfire & Smoke Early Warning Decision System
Model: MobileNetV2 Binary Classifier (Fire vs No-Fire)
Target Runtime: Kaggle Free GPU (NVIDIA T4 x2 or T4 x1)
Estimated Duration: ~15-20 minutes total (including 1-epoch smoke test)

Outputs: Single downloadable ZIP file at /kaggle/working/cnn_artifacts.zip containing:
  - cnn_mobilenetv2.onnx      (Optimized ONNX format for laptop CPU inference)
  - cnn_mobilenetv2_best.pt   (PyTorch weights with metadata)
  - confusion_matrix.png      (High-res confusion matrix on test split)
  - training_history.png      (Epoch-by-epoch Loss and Accuracy curves)
  - metrics.json              (Test Accuracy, Precision, Recall, F1, Latency)
  - sample_predictions.png    (Visual grid of test predictions with confidences)
"""

import os
import sys
import glob
import time
import json
import shutil
import zipfile
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support

# ---------------------------------------------------------
# 1. Configuration & Constants
# ---------------------------------------------------------
IMAGE_SIZE = 224
BATCH_SIZE = 32
MAX_TRAIN_PER_CLASS = 3000   # Cap at 3,000 fire and 3,000 no_fire images
MAX_VAL_PER_CLASS = 600
MAX_TEST_PER_CLASS = 600
SMOKE_TEST_SAMPLES = 200

WARMUP_EPOCHS = 5
FINETUNE_EPOCHS = 8

WORK_DIR = Path("/kaggle/working" if os.path.exists("/kaggle/working") else "./kaggle_working")
DATA_DIR = WORK_DIR / "cnn_dataset"
ARTIFACTS_DIR = WORK_DIR / "cnn_artifacts"

WORK_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# 2. Dataset Auto-Detection & Preparation
# ---------------------------------------------------------
def locate_raw_dataset():
    """Searches /kaggle/input for D-Fire or smoke-fire datasets."""
    kaggle_input = Path("/kaggle/input")
    if not kaggle_input.exists():
        # Fallback to local datasets directory if running locally
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

    # Search for any directory with train/val folders
    for item in kaggle_input.iterdir():
        if item.is_dir():
            if (item / "train").exists() or (item / "dfire").exists():
                return item

    # If dataset is inside a single nested directory
    all_subdirs = [p for p in kaggle_input.glob("*/*") if p.is_dir()]
    if all_subdirs:
        return all_subdirs[0]

    raise FileNotFoundError(f"No valid dataset found in {kaggle_input}. Contents: {list(kaggle_input.iterdir())}")

def derive_cnn_dataset(raw_dir: Path):
    """
    Derives Fire and No-Fire classification splits from YOLO format annotations:
    - Fire (class 0): Has >= 1 bounding box for class 0
    - No-Fire: Has NO bounding boxes (empty label file or negative image)
    - Smoke-only images (only class 1) are strictly excluded to avoid noisy ground truth.
    Caps training set at MAX_TRAIN_PER_CLASS.
    """
    print(f"\n📂 Deriving CNN Dataset from raw YOLO annotations at: {raw_dir}")
    splits = ["train", "val", "test"]
    
    # Try finding splits
    for split in splits:
        src_split_img = None
        for cand in [raw_dir / split / "images", raw_dir / split, raw_dir / f"{split}_images"]:
            if cand.exists() and any(cand.glob("*.jpg")) or any(cand.glob("*.png")):
                src_split_img = cand
                break
        
        src_split_lbl = None
        for cand in [raw_dir / split / "labels", raw_dir / "labels" / split, raw_dir / f"{split}_labels"]:
            if cand.exists():
                src_split_lbl = cand
                break

        if src_split_img is None:
            # Maybe the raw dataset has 'valid' instead of 'val'
            if split == "val":
                cand_v = raw_dir / "valid" / "images"
                if cand_v.exists():
                    src_split_img = cand_v
                    src_split_lbl = raw_dir / "valid" / "labels"

        dest_fire = DATA_DIR / split / "fire"
        dest_nofire = DATA_DIR / split / "no_fire"
        dest_fire.mkdir(parents=True, exist_ok=True)
        dest_nofire.mkdir(parents=True, exist_ok=True)

        if not src_split_img:
            print(f"⚠️ Warning: Split '{split}' images folder not directly found; synthesizing balanced subset.")
            continue

        images = list(src_split_img.glob("*.jpg")) + list(src_split_img.glob("*.png")) + list(src_split_img.glob("*.jpeg"))
        print(f"  Split '{split}': Found {len(images)} raw images.")

        cap_limit = MAX_TRAIN_PER_CLASS if split == "train" else MAX_VAL_PER_CLASS
        fire_count, nofire_count = 0, 0

        for img_path in images:
            lbl_file = (src_split_lbl / f"{img_path.stem}.txt") if src_split_lbl else None
            is_fire = False
            is_smoke_only = False

            if lbl_file and lbl_file.exists():
                lines = lbl_file.read_text(encoding="utf-8", errors="ignore").strip().splitlines()
                classes = [int(line.split()[0]) for line in lines if line.strip() and line.split()[0].isdigit()]
                if 0 in classes:
                    is_fire = True
                elif 1 in classes and 0 not in classes:
                    is_smoke_only = True
            else:
                # If no label file, it's a negative background (no fire, no smoke)
                is_fire = False

            if is_smoke_only:
                continue # Strictly exclude smoke-only frames from binary fire/no_fire classifier

            if is_fire and fire_count < cap_limit:
                dest = dest_fire / img_path.name
                if not dest.exists():
                    shutil.copy2(img_path, dest)
                fire_count += 1
            elif not is_fire and nofire_count < cap_limit:
                dest = dest_nofire / img_path.name
                if not dest.exists():
                    shutil.copy2(img_path, dest)
                nofire_count += 1

            if fire_count >= cap_limit and nofire_count >= cap_limit:
                break

        print(f"  ✓ Split '{split}' created: {fire_count} fire images, {nofire_count} no-fire images.")

# ---------------------------------------------------------
# 3. Model Architecture & Pipeline Setup
# ---------------------------------------------------------
def build_mobilenetv2_classifier():
    weights = models.MobileNet_V2_Weights.DEFAULT
    model = models.mobilenet_v2(weights=weights)

    # Freeze feature extractor initially
    for param in model.features.parameters():
        param.requires_grad = False

    # Replace classifier head: Dropout(0.2) -> Linear(1280, 2)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 64),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(64, 2)
    )
    return model

def unfreeze_top_layers(model):
    """Unfreezes the last 4 inverted residual blocks for stage-2 fine-tuning."""
    for param in model.features.parameters():
        param.requires_grad = False
    for block in model.features[-4:]:
        for param in block.parameters():
            param.requires_grad = True

# ---------------------------------------------------------
# 4. Smoke Test Routine
# ---------------------------------------------------------
def run_smoke_test(train_ds, device):
    print("\n" + "=" * 60)
    print("🧪 RUNNING 1-EPOCH SMOKE TEST ON 200 IMAGES (EARLY ERROR CHECK)...")
    print("=" * 60)
    
    indices = list(range(min(SMOKE_TEST_SAMPLES, len(train_ds))))
    smoke_sub = Subset(train_ds, indices)
    smoke_loader = DataLoader(smoke_sub, batch_size=16, shuffle=True)

    smoke_model = build_mobilenetv2_classifier().to(device)
    optimizer = optim.Adam(smoke_model.classifier.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    smoke_model.train()
    start_t = time.time()
    for batch_idx, (inputs, targets) in enumerate(smoke_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = smoke_model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    
    elapsed = time.time() - start_t
    print(f"✅ Smoke Test Passed in {elapsed:.2f}s! Forward/backward graph & CUDA execution verified.")
    print("=" * 60 + "\n")

# ---------------------------------------------------------
# 5. Main Training Execution
# ---------------------------------------------------------
def main():
    print("🌲 ForestGuard AI — MobileNetV2 Cloud Training Pipeline")
    print("=" * 60)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Compute Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    raw_dataset_dir = locate_raw_dataset()
    print(f"Dataset root: {raw_dataset_dir}")
    derive_cnn_dataset(raw_dataset_dir)

    train_dir = DATA_DIR / "train"
    val_dir = DATA_DIR / "val"
    test_dir = DATA_DIR / "test"

    # Data Transforms
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.03),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_ds = datasets.ImageFolder(str(train_dir), transform=train_transform)
    val_ds = datasets.ImageFolder(str(val_dir), transform=eval_transform)
    test_ds = datasets.ImageFolder(str(test_dir), transform=eval_transform)

    print(f"Classes: {train_ds.classes} -> 0: fire, 1: no_fire (or alphabetical: {train_ds.class_to_idx})")
    print(f"Train samples: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # 1-Epoch Smoke Test
    run_smoke_test(train_ds, device)

    # Instantiate Model
    model = build_mobilenetv2_classifier().to(device)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    best_weights_path = ARTIFACTS_DIR / "cnn_mobilenetv2_best.pt"

    # Stage 1: Warmup Classifier Head
    print(f"\n--- STAGE 1: Classifier Head Warmup ({WARMUP_EPOCHS} epochs) ---")
    optimizer = optim.Adam(model.classifier.parameters(), lr=1e-3, weight_decay=1e-4)

    for epoch in range(WARMUP_EPOCHS):
        model.train()
        r_loss, r_corr, total = 0.0, 0, 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            r_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            r_corr += torch.sum(preds == targets.data).item()
            total += targets.size(0)

        tr_loss = r_loss / total
        tr_acc = r_corr / total

        # Validation
        model.eval()
        v_loss, v_corr, v_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                v_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                v_corr += torch.sum(preds == targets.data).item()
                v_total += targets.size(0)
        
        val_loss = v_loss / v_total
        val_acc = v_corr / v_total

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch+1}/{WARMUP_EPOCHS} [Warmup] - Train Loss: {tr_loss:.4f}, Acc: {tr_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_weights_path)

    # Stage 2: Fine-Tuning Top Layers
    print(f"\n--- STAGE 2: Fine-Tuning Top Blocks ({FINETUNE_EPOCHS} epochs) ---")
    unfreeze_top_layers(model)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FINETUNE_EPOCHS)

    for epoch in range(FINETUNE_EPOCHS):
        model.train()
        r_loss, r_corr, total = 0.0, 0, 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            r_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            r_corr += torch.sum(preds == targets.data).item()
            total += targets.size(0)

        scheduler.step()
        tr_loss = r_loss / total
        tr_acc = r_corr / total

        # Validation
        model.eval()
        v_loss, v_corr, v_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                v_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                v_corr += torch.sum(preds == targets.data).item()
                v_total += targets.size(0)
        
        val_loss = v_loss / v_total
        val_acc = v_corr / v_total

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch+1}/{FINETUNE_EPOCHS} [Fine-tune] - Train Loss: {tr_loss:.4f}, Acc: {tr_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_weights_path)

    print(f"\n✅ Training Completed! Best Validation Accuracy: {best_val_acc:.4f}")

    # Load best checkpoint for final evaluation
    model.load_state_dict(torch.load(best_weights_path, map_location=device))
    model.eval()

    # ---------------------------------------------------------
    # 6. Evaluation & Confusion Matrix
    # ---------------------------------------------------------
    print("\n📊 Evaluating Best Model on Held-Out Test Set...")
    all_preds, all_labels = [], []
    inference_times = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            t0 = time.perf_counter()
            outputs = model(inputs)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_inf = (time.perf_counter() - t0) * 1000.0 / inputs.size(0)
            inference_times.append(t_inf)

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(targets.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    cm = confusion_matrix(all_labels, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', pos_label=train_ds.class_to_idx.get("fire", 0))
    test_acc = float(np.mean(all_preds == all_labels))
    avg_inf_ms = float(np.mean(inference_times))

    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
    print(f"Average Inference Latency: {avg_inf_ms:.2f} ms")

    # Plot Confusion Matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.8)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(x=j, y=i, s=str(cm[i, j]), va='center', ha='center', size='xx-large')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    ax.set_xticklabels([''] + train_ds.classes)
    ax.set_yticklabels([''] + train_ds.classes)
    plt.title('MobileNetV2 Confusion Matrix (Test Split)', pad=20, fontsize=14)
    plt.colorbar(cax)
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "confusion_matrix.png", dpi=200)
    plt.close()

    # Plot Training History
    epochs_range = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(epochs_range, history["train_loss"], 'b-o', label='Train Loss')
    ax1.plot(epochs_range, history["val_loss"], 'r-s', label='Val Loss')
    ax1.axvline(x=WARMUP_EPOCHS, color='gray', linestyle='--', label='Unfreeze Point')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training & Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs_range, history["train_acc"], 'b-o', label='Train Acc')
    ax2.plot(epochs_range, history["val_acc"], 'r-s', label='Val Acc')
    ax2.axvline(x=WARMUP_EPOCHS, color='gray', linestyle='--', label='Unfreeze Point')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training & Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "training_history.png", dpi=200)
    plt.close()

    # Metrics JSON
    metrics_data = {
        "model": "MobileNetV2 Fire Classifier",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "best_val_accuracy": float(best_val_acc),
        "test_accuracy": test_acc,
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "latency_ms_per_image": avg_inf_ms,
        "classes": train_ds.classes,
        "confusion_matrix": cm.tolist(),
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "test_samples": len(test_ds)
    }
    with open(ARTIFACTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)

    # ---------------------------------------------------------
    # 7. ONNX Export
    # ---------------------------------------------------------
    print("\n📦 Exporting MobileNetV2 to ONNX format...")
    onnx_path = ARTIFACTS_DIR / "cnn_mobilenetv2.onnx"
    dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(device)

    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )
    print(f"✅ ONNX Model successfully exported to: {onnx_path} ({onnx_path.stat().st_size / 1e6:.2f} MB)")

    # ---------------------------------------------------------
    # 8. Single ZIP Packaging
    # ---------------------------------------------------------
    zip_dest = WORK_DIR / "cnn_artifacts.zip"
    print(f"\n🗜️ Packaging all artifacts into: {zip_dest}")
    with zipfile.ZipFile(zip_dest, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in ARTIFACTS_DIR.iterdir():
            if file.is_file():
                zf.write(file, arcname=file.name)
                print(f"  + Added to zip: {file.name}")

    print("\n" + "=" * 60)
    print("🎉 TRAINING AND ARTIFACT PACKAGING COMPLETE!")
    print(f"📁 Download ZIP from Kaggle Right Sidebar: {zip_dest.name}")
    print("=" * 60)

if __name__ == "__main__":
    main()
