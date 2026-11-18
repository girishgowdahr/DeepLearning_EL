"""
ForestGuard AI — Dual-GPU Kaggle Orchestration Script
=====================================================
Academic Prototype: Early Wildfire & Smoke Detection Decision System
Target Environment: Kaggle Notebook with 2 x NVIDIA Tesla T4 GPUs
Executes YOLO11n on GPU 0 and MobileNetV2 CNN on GPU 1 in parallel subprocesses.
Produces two self-contained zip files:
  - /kaggle/working/forestguard_outputs/yolo_artifacts.zip
  - /kaggle/working/forestguard_outputs/cnn_artifacts.zip
"""

import os
import sys
import time
import json
import yaml
import shutil
import zipfile
import subprocess
from pathlib import Path

# -------------------------------------------------------------------------
# Environment & Arguments
# -------------------------------------------------------------------------
SMOKE_MODE = os.environ.get("SMOKE", "0") == "1" or "--smoke" in sys.argv
RESUME_MODE = os.environ.get("RESUME", "0") == "1" or "--resume" in sys.argv

BASE_OUTPUT_DIR = Path("/kaggle/working/forestguard_outputs" if os.path.exists("/kaggle") else "./forestguard_outputs")
LOGS_DIR = BASE_OUTPUT_DIR / "logs"
YOLO_OUT_DIR = BASE_OUTPUT_DIR / "yolo"
CNN_OUT_DIR = BASE_OUTPUT_DIR / "cnn"
YOLO_DATA_DIR = BASE_OUTPUT_DIR / "yolo_dataset"
CNN_DATA_DIR = BASE_OUTPUT_DIR / "cnn_dataset"

for d in [LOGS_DIR, YOLO_OUT_DIR, CNN_OUT_DIR, YOLO_DATA_DIR, CNN_DATA_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------------
# 1. Dependency Installation & Environment Check
# -------------------------------------------------------------------------
print("=" * 70)
print("🌲 FORESTGUARD AI: DUAL-MODEL KAGGLE TRAINING PIPELINE")
print(f"Mode: {'🧪 SMOKE TEST (1 Epoch, ~200 samples)' if SMOKE_MODE else '🚀 FULL PRODUCTION RUN'}")
print("=" * 70)

# Ensure packages installed
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "ultralytics>=8.3.0", "onnx>=1.15.0", "onnxruntime>=1.17.0",
    "scikit-learn", "matplotlib", "pyyaml"
])

import torch
print(f"Python Version: {sys.version.split()[0]}")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    print("⚠️ WARNING: CUDA not detected! Ensure Accelerator = GPU T4 x2 in Kaggle settings.")
    gpu_count = 0
else:
    gpu_count = torch.cuda.device_count()
    print(f"GPUs Detected: {gpu_count}")
    for i in range(gpu_count):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)} (VRAM: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB)")

# -------------------------------------------------------------------------
# 2. Dataset Auto-Detection & Integrity Verification
# -------------------------------------------------------------------------
def find_raw_dataset() -> Path:
    kaggle_input = Path("/kaggle/input")
    if not kaggle_input.exists():
        local_cand = Path("datasets/DFire")
        if local_cand.exists():
            return local_cand
        raise FileNotFoundError("Could not find /kaggle/input or local datasets folder.")

    # Search for D-Fire or smoke-fire datasets
    for cand in kaggle_input.glob("**/*train*"):
        if (cand / "images").exists() or (cand / "labels").exists():
            return cand.parent
        if cand.is_dir() and any(x in str(cand).lower() for x in ["dfire", "smoke", "fire"]):
            return cand

    subdirs = [p for p in kaggle_input.iterdir() if p.is_dir()]
    if subdirs:
        return subdirs[0]

    raise FileNotFoundError(f"No dataset found in {kaggle_input}. Please attach smoke-fire-detection-yolo.")

raw_dataset_path = find_raw_dataset()
print(f"\n📂 Raw dataset located at: {raw_dataset_path}")

# Build Dataset Report
dataset_report = {
    "raw_dataset_path": str(raw_dataset_path),
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "splits": {}
}

MAX_YOLO_TRAIN = 200 if SMOKE_MODE else 6000
MAX_CNN_PER_CLASS = 100 if SMOKE_MODE else 3000

print("\n--- Validating Splits & Deriving Datasets ---")
splits = ["train", "val", "test"]

for s in splits:
    # Locate split images and labels
    src_img = None
    for cand in [raw_dataset_path / s / "images", raw_dataset_path / s, raw_dataset_path / f"{s}_images"]:
        if cand.exists() and (any(cand.glob("*.jpg")) or any(cand.glob("*.png"))):
            src_img = cand
            break
    if src_img is None and s == "val":
        for cand in [raw_dataset_path / "valid" / "images", raw_dataset_path / "valid"]:
            if cand.exists():
                src_img = cand
                break

    src_lbl = None
    for cand in [raw_dataset_path / s / "labels", raw_dataset_path / "labels" / s, raw_dataset_path / f"{s}_labels"]:
        if cand.exists():
            src_lbl = cand
            break
    if src_lbl is None and s == "val":
        cand = raw_dataset_path / "valid" / "labels"
        if cand.exists():
            src_lbl = cand

    images = list(src_img.glob("*.jpg")) + list(src_img.glob("*.png")) if src_img else []
    print(f"Split '{s}': Found {len(images)} raw images.")
    dataset_report["splits"][s] = {"raw_images": len(images)}

    # Destination directories
    (YOLO_DATA_DIR / s / "images").mkdir(parents=True, exist_ok=True)
    (YOLO_DATA_DIR / s / "labels").mkdir(parents=True, exist_ok=True)
    (CNN_DATA_DIR / s / "fire").mkdir(parents=True, exist_ok=True)
    (CNN_DATA_DIR / s / "no_fire").mkdir(parents=True, exist_ok=True)

    yolo_count = 0
    cnn_fire_count = 0
    cnn_nofire_count = 0

    yolo_cap = MAX_YOLO_TRAIN if s == "train" else (100 if SMOKE_MODE else 1000)
    cnn_cap = MAX_CNN_PER_CLASS if s == "train" else (50 if SMOKE_MODE else 600)

    for img in images:
        lbl_file = (src_lbl / f"{img.stem}.txt") if src_lbl else None
        has_fire = False
        is_smoke_only = False

        if lbl_file and lbl_file.exists():
            lines = lbl_file.read_text(encoding="utf-8", errors="ignore").strip().splitlines()
            classes = [int(line.split()[0]) for line in lines if line.strip() and line.split()[0].isdigit()]
            if 0 in classes:
                has_fire = True
            elif 1 in classes and 0 not in classes:
                is_smoke_only = True
        else:
            has_fire = False

        # Populate YOLO dataset
        if yolo_count < yolo_cap:
            dest_img = YOLO_DATA_DIR / s / "images" / img.name
            dest_lbl = YOLO_DATA_DIR / s / "labels" / f"{img.stem}.txt"
            if not dest_img.exists():
                shutil.copy2(img, dest_img)
            if lbl_file and lbl_file.exists() and not dest_lbl.exists():
                shutil.copy2(lbl_file, dest_lbl)
            elif not dest_lbl.exists():
                dest_lbl.touch()
            yolo_count += 1

        # Populate CNN dataset (Exclude smoke-only to avoid noisy label ground truth)
        if not is_smoke_only:
            if has_fire and cnn_fire_count < cnn_cap:
                dest = CNN_DATA_DIR / s / "fire" / img.name
                if not dest.exists():
                    shutil.copy2(img, dest)
                cnn_fire_count += 1
            elif not has_fire and cnn_nofire_count < cnn_cap:
                dest = CNN_DATA_DIR / s / "no_fire" / img.name
                if not dest.exists():
                    shutil.copy2(img, dest)
                cnn_nofire_count += 1

    dataset_report["splits"][s]["yolo_used"] = yolo_count
    dataset_report["splits"][s]["cnn_fire"] = cnn_fire_count
    dataset_report["splits"][s]["cnn_nofire"] = cnn_nofire_count
    print(f"  ✓ Split '{s}' prepared: YOLO={yolo_count}, CNN Fire={cnn_fire_count}, CNN No-Fire={cnn_nofire_count}")

# Generate data_capped.yaml for YOLO
data_yaml_path = BASE_OUTPUT_DIR / "data.yaml"
with open(data_yaml_path, "w", encoding="utf-8") as f:
    yaml.dump({
        "path": str(YOLO_DATA_DIR.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": {0: "fire", 1: "smoke"}
    }, f)

with open(BASE_OUTPUT_DIR / "dataset_report.json", "w", encoding="utf-8") as f:
    json.dump(dataset_report, f, indent=2)

print(f"✅ Generated dataset report: {BASE_OUTPUT_DIR / 'dataset_report.json'}")

# -------------------------------------------------------------------------
# 3. Execution Scripts for Parallel Subprocesses
# -------------------------------------------------------------------------
# Worker script for YOLO (GPU 0)
yolo_worker_script = BASE_OUTPUT_DIR / "worker_yolo.py"
yolo_worker_code = f"""
import os, sys, json, time, shutil
from pathlib import Path
from ultralytics import YOLO

os.environ['CUDA_VISIBLE_DEVICES'] = '{0 if gpu_count > 1 else 0}'
data_yaml = '{str(data_yaml_path)}'
out_dir = Path('{str(YOLO_OUT_DIR)}')
epochs = {1 if SMOKE_MODE else 30}
batch_size = 16
imgsz = 640

print("🚀 [YOLO Worker] Starting YOLO11n fine-tuning...")
model = YOLO('yolo11n.pt')
res = model.train(
    data=data_yaml,
    epochs=epochs,
    imgsz=imgsz,
    batch=batch_size,
    patience=10,
    device=0 if {str(gpu_count > 0)} else 'cpu',
    project=str(out_dir / 'runs'),
    name='train_run',
    seed=42,
    save=True,
    plots=True
)

save_dir = Path(res.save_dir)
best_pt = save_dir / 'weights' / 'best.pt'
if not best_pt.exists():
    best_pt = save_dir / 'weights' / 'last.pt'

target_pt = out_dir / 'yolo_best.pt'
shutil.copy2(best_pt, target_pt)
print(f"✅ [YOLO Worker] Saved weights to: {{target_pt}}")

# Export to ONNX
print("📦 [YOLO Worker] Exporting to ONNX...")
eval_model = YOLO(str(target_pt))
eval_model.export(format='onnx', imgsz=imgsz, simplify=True, dynamic=True)
onnx_cand = target_pt.with_suffix('.onnx')
if onnx_cand.exists():
    shutil.copy2(onnx_cand, out_dir / 'yolo_best.onnx')

# Test Split Validation
print("📊 [YOLO Worker] Validating best model on test split...")
val_res = eval_model.val(data=data_yaml, split='test', imgsz=imgsz)
metrics = {{
    "model": "YOLO11 Nano Wildfire Detector",
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "mAP50": float(val_res.box.map50),
    "mAP50-95": float(val_res.box.map),
    "precision": float(val_res.box.mp),
    "recall": float(val_res.box.mr),
    "epochs_trained": epochs,
    "image_size": imgsz,
    "classes": {{
        "fire": {{
            "precision": float(val_res.box.p[0]) if len(val_res.box.p) > 0 else 0.0,
            "recall": float(val_res.box.r[0]) if len(val_res.box.r) > 0 else 0.0,
            "mAP50": float(val_res.box.ap50[0]) if len(val_res.box.ap50) > 0 else 0.0
        }},
        "smoke": {{
            "precision": float(val_res.box.p[1]) if len(val_res.box.p) > 1 else 0.0,
            "recall": float(val_res.box.r[1]) if len(val_res.box.r) > 1 else 0.0,
            "mAP50": float(val_res.box.ap50[1]) if len(val_res.box.ap50) > 1 else 0.0
        }}
    }}
}}
with open(out_dir / 'yolo_metrics.json', 'w', encoding='utf-8') as f:
    json.dump(metrics, f, indent=2)

# Copy plots
for p in ['results.png', 'confusion_matrix.png', 'PR_curve.png', 'F1_curve.png']:
    src = save_dir / p
    if src.exists():
        shutil.copy2(src, out_dir / f"yolo_{{p}}")
        shutil.copy2(src, out_dir / p)

print("🎉 [YOLO Worker] Execution Complete.")
"""
yolo_worker_script.write_text(yolo_worker_code, encoding="utf-8")

# Worker script for CNN (GPU 1 if 2 GPUs, else GPU 0)
cnn_worker_script = BASE_OUTPUT_DIR / "worker_cnn.py"
cnn_gpu_id = 1 if gpu_count > 1 else 0
cnn_worker_code = f"""
import os, sys, json, time, shutil
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

os.environ['CUDA_VISIBLE_DEVICES'] = '{cnn_gpu_id}'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
out_dir = Path('{str(CNN_OUT_DIR)}')
data_dir = Path('{str(CNN_DATA_DIR)}')

warmup_epochs = {1 if SMOKE_MODE else 5}
finetune_epochs = {1 if SMOKE_MODE else 8}

print(f"🚀 [CNN Worker] Starting MobileNetV2 on device: {{device}}...")
weights = models.MobileNet_V2_Weights.DEFAULT
model = models.mobilenet_v2(weights=weights)

# Freeze backbone
for param in model.features.parameters():
    param.requires_grad = False

in_f = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(in_f, 64),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.2),
    nn.Linear(64, 2)
)
model = model.to(device)

train_tf = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.03),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
eval_tf = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_ds = datasets.ImageFolder(str(data_dir / 'train'), transform=train_tf)
val_ds = datasets.ImageFolder(str(data_dir / 'val'), transform=eval_tf)
test_ds = datasets.ImageFolder(str(data_dir / 'test'), transform=eval_tf)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=1e-3, weight_decay=1e-4)

best_val_acc = 0.0
best_pt_path = out_dir / 'cnn_mobilenetv2_best.pt'
history = {{"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}}

# Stage 1: Warmup
for epoch in range(warmup_epochs):
    model.train()
    r_loss, r_corr, total = 0.0, 0, 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        r_loss += loss.item() * x.size(0)
        r_corr += (torch.max(out, 1)[1] == y).sum().item()
        total += y.size(0)
    
    # Val
    model.eval()
    v_loss, v_corr, v_total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            v_loss += criterion(out, y).item() * x.size(0)
            v_corr += (torch.max(out, 1)[1] == y).sum().item()
            v_total += y.size(0)
    
    tr_acc = r_corr / total
    val_acc = v_corr / v_total
    history["train_acc"].append(tr_acc)
    history["val_acc"].append(val_acc)
    history["train_loss"].append(r_loss / total)
    history["val_loss"].append(v_loss / v_total)
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), best_pt_path)

# Stage 2: Fine-tuning
for block in model.features[-4:]:
    for param in block.parameters():
        param.requires_grad = True

opt2 = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-3)
for epoch in range(finetune_epochs):
    model.train()
    r_loss, r_corr, total = 0.0, 0, 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        opt2.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        opt2.step()
        r_loss += loss.item() * x.size(0)
        r_corr += (torch.max(out, 1)[1] == y).sum().item()
        total += y.size(0)
    
    model.eval()
    v_loss, v_corr, v_total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            v_loss += criterion(out, y).item() * x.size(0)
            v_corr += (torch.max(out, 1)[1] == y).sum().item()
            v_total += y.size(0)
    
    val_acc = v_corr / v_total
    history["train_acc"].append(r_corr / total)
    history["val_acc"].append(val_acc)
    history["train_loss"].append(r_loss / total)
    history["val_loss"].append(v_loss / v_total)
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), best_pt_path)

print(f"✅ [CNN Worker] Best Val Acc: {{best_val_acc:.4f}}")

# Load best for evaluation on test split
model.load_state_dict(torch.load(best_pt_path, map_location=device))
model.eval()

preds, targets = [], []
with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        out = model(x)
        preds.extend(torch.max(out, 1)[1].cpu().numpy())
        targets.extend(y.numpy())

preds = np.array(preds)
targets = np.array(targets)
cm = confusion_matrix(targets, preds)
p, r, f1, _ = precision_recall_fscore_support(targets, preds, average='binary', pos_label=0)
test_acc = float(np.mean(preds == targets))

metrics = {{
    "model": "MobileNetV2 Fire Classifier",
    "accuracy": test_acc,
    "test_accuracy": test_acc,
    "macro_f1": float(f1),
    "classes": {{
        "fire": {{"precision": float(p), "recall": float(r)}}
    }},
    "confusion_matrix": cm.tolist()
}}
with open(out_dir / 'cnn_metrics.json', 'w', encoding='utf-8') as f:
    json.dump(metrics, f, indent=2)

# Confusion Matrix Plot
fig, ax = plt.subplots(figsize=(6, 5))
cax = ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.8)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(x=j, y=i, s=str(cm[i, j]), va='center', ha='center', size='xx-large')
plt.xlabel('Predicted')
plt.ylabel('True')
ax.set_xticklabels(['', 'fire', 'no_fire'])
ax.set_yticklabels(['', 'fire', 'no_fire'])
plt.title('CNN Confusion Matrix (Test Split)')
plt.savefig(out_dir / 'cnn_confusion_matrix.png', dpi=200)
plt.close()

# Loss & Accuracy Plots
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(history['train_acc'], label='Train Acc')
ax.plot(history['val_acc'], label='Val Acc')
ax.legend()
plt.title('CNN Accuracy')
plt.savefig(out_dir / 'cnn_accuracy.png', dpi=200)
plt.close()

# Export ONNX
print("📦 [CNN Worker] Exporting MobileNetV2 to ONNX...")
dummy = torch.randn(1, 3, 224, 224).to(device)
onnx_path = out_dir / 'cnn_mobilenetv2.onnx'
torch.onnx.export(
    model, dummy, str(onnx_path),
    opset_version=14,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={{'input': {{0: 'batch_size'}}, 'output': {{0: 'batch_size'}}}}
)

# MODEL_INFO.json
info = {{
    "cnn": {{
        "name": "MobileNetV2",
        "version": "1.0.0",
        "trained_by": "ForestGuard AI (Kaggle Dual-T4)",
        "input_size": "224x224",
        "classes": ["fire", "no_fire"],
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225]
    }},
    "yolo": {{
        "name": "YOLO11n",
        "version": "1.0.0",
        "trained_by": "ForestGuard AI (Kaggle Dual-T4)",
        "input_size": "640x640",
        "classes": ["fire", "smoke"]
    }}
}}
with open(out_dir / 'MODEL_INFO.json', 'w', encoding='utf-8') as f:
    json.dump(info, f, indent=2)

print("🎉 [CNN Worker] Execution Complete.")
"""
cnn_worker_script.write_text(cnn_worker_code, encoding="utf-8")

# -------------------------------------------------------------------------
# 4. Launch Subprocesses (Parallel on 2 GPUs or Sequential on 1 GPU)
# -------------------------------------------------------------------------
print("\n" + "=" * 70)
if gpu_count > 1:
    print("⚡ Launching YOLO on GPU 0 and CNN on GPU 1 simultaneously...")
    p_yolo = subprocess.Popen([sys.executable, str(yolo_worker_script)], stdout=open(LOGS_DIR / "yolo.log", "w"), stderr=subprocess.STDOUT)
    p_cnn = subprocess.Popen([sys.executable, str(cnn_worker_script)], stdout=open(LOGS_DIR / "cnn.log", "w"), stderr=subprocess.STDOUT)

    while p_yolo.poll() is None or p_cnn.poll() is None:
        time.sleep(10)
        status_y = "Running" if p_yolo.poll() is None else f"Done ({p_yolo.returncode})"
        status_c = "Running" if p_cnn.poll() is None else f"Done ({p_cnn.returncode})"
        print(f"⏳ Training Status: YOLO [GPU 0] = {status_y} | CNN [GPU 1] = {status_c}")
else:
    print("⚙️ Single GPU detected: running sequentially (CNN first, then YOLO)...")
    subprocess.check_call([sys.executable, str(cnn_worker_script)])
    subprocess.check_call([sys.executable, str(yolo_worker_script)])

# -------------------------------------------------------------------------
# 5. Packaging into Single ZIP Files
# -------------------------------------------------------------------------
print("\n" + "=" * 70)
print("🗜️ Packaging Artifacts...")

# Package YOLO
yolo_zip = BASE_OUTPUT_DIR / "yolo_artifacts.zip"
with zipfile.ZipFile(yolo_zip, "w", zipfile.ZIP_DEFLATED) as zf:
    for f in YOLO_OUT_DIR.iterdir():
        if f.is_file():
            zf.write(f, arcname=f.name)
print(f"✅ Created YOLO Artifacts: {yolo_zip} ({yolo_zip.stat().st_size / 1e6:.2f} MB)")

# Package CNN
cnn_zip = BASE_OUTPUT_DIR / "cnn_artifacts.zip"
with zipfile.ZipFile(cnn_zip, "w", zipfile.ZIP_DEFLATED) as zf:
    for f in CNN_OUT_DIR.iterdir():
        if f.is_file():
            zf.write(f, arcname=f.name)
print(f"✅ Created CNN Artifacts: {cnn_zip} ({cnn_zip.stat().st_size / 1e6:.2f} MB)")

# Copy zips to /kaggle/working root for easiest Kaggle download
if os.path.exists("/kaggle/working"):
    shutil.copy2(yolo_zip, Path("/kaggle/working/yolo_artifacts.zip"))
    shutil.copy2(cnn_zip, Path("/kaggle/working/cnn_artifacts.zip"))
    print("📁 Copied zip files to /kaggle/working/ root for one-click download.")

print("=" * 70)
print("🎉 DUAL-MODEL TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
print("=" * 70)
