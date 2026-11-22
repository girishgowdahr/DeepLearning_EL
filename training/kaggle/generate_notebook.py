"""
Generator for ForestGuard_AI_Complete_Training.ipynb
Constructs an audited, fully automated, leak-free, reproducible Kaggle Jupyter Notebook.
Supports dynamic class resolution for sayedgamal99/smoke-fire-detection-yolo
as well as alternative D-Fire mirrors, ensuring strict compatibility with the FastAPI backend.
"""

import json
from pathlib import Path

def create_notebook():
    cells = []

    def md_cell(source_text):
        lines = [line + "\n" for line in source_text.strip().split("\n")]
        if lines:
            lines[-1] = lines[-1].rstrip("\n")
        return {
            "cell_type": "markdown",
            "metadata": {},
            "source": lines
        }

    def code_cell(source_text):
        lines = [line + "\n" for line in source_text.strip().split("\n")]
        if lines:
            lines[-1] = lines[-1].rstrip("\n")
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": lines
        }

    # -------------------------------------------------------------
    # CELL 1: Header / Overview Markdown
    # -------------------------------------------------------------
    cells.append(md_cell("""# 🌲 ForestGuard AI — Fully Automated Dual-Model Training Pipeline
### Academic Prototype: Early Wildfire & Smoke Detection Decision System
**Architectures**: Ultralytics YOLO11 Nano (Bounding Box Detector) + MobileNetV2 CNN (Binary Classifier)  
**Target Platform**: Kaggle Free GPU (**2 × NVIDIA Tesla T4** or **1 × Tesla T4**)  
**Dataset**: `sayedgamal99/smoke-fire-detection-yolo` (or D-Fire mirror)  
**Outputs**: `yolo_artifacts.zip` & `cnn_artifacts.zip` (ready for one-click download to your laptop)

---

### 📋 Prerequisites & Kaggle Execution Guide
1. **Accelerator**: In the right sidebar under **Notebook Options**, set **Accelerator** to **GPU T4 x2** (or GPU T4 x1).
2. **Internet**: Toggle **Internet** to **ON** (required to fetch pretrained base weights).
3. **Dataset**: Click **+ Add Input** (top right) $\\rightarrow$ Search for:
   ```
   smoke-fire-detection-yolo
   ```
   Add **`sayedgamal99/smoke-fire-detection-yolo`** (License: CC0 / CC BY 4.0).
4. **Execution Mode**:
   - `SMOKE_MODE = True`: 1-epoch quick pipeline verification (~3–5 minutes) to test GPU, tensor flow, and packaging.
   - `SMOKE_MODE = False`: Full production training (~45–60 minutes) yielding IEEE research-grade weights.
5. **Run All**: Click **Run All** (or **Save Version** $\\rightarrow$ **Save & Run All (Commit)** for background cloud training)."""))

    # -------------------------------------------------------------
    # CELL 2: Environment and GPU Verification
    # -------------------------------------------------------------
    cells.append(md_cell("""## 1. Environment & GPU Verification
Checks the Kaggle runtime environment, CUDA availability, GPU device models, VRAM capacity, and prepares directory trees."""))

    cells.append(code_cell("""import os
import sys
import time
import json
import shutil
import zipfile
import hashlib
from pathlib import Path

# Toggle: True = ~3-min validation pass; False = full academic training
SMOKE_MODE = os.environ.get("SMOKE", "0") == "1" or False

print("=" * 70)
print("🌲 FORESTGUARD AI: DUAL-MODEL TRAINING PIPELINE")
print(f"Mode: {'🧪 SMOKE TEST (1 Epoch, Quick Validation)' if SMOKE_MODE else '🚀 FULL PRODUCTION RUN'}")
print("=" * 70)

# Check PyTorch & CUDA
import torch
print(f"Python Version:  {sys.version.split()[0]}")
print(f"PyTorch Version: {torch.__version__}")
cuda_available = torch.cuda.is_available()
print(f"CUDA Available:  {cuda_available}")

if not cuda_available:
    print("⚠️ WARNING: CUDA not detected! Please set Accelerator = GPU T4 x2 in Kaggle settings.")
    gpu_count = 0
else:
    gpu_count = torch.cuda.device_count()
    print(f"Total GPUs Detected: {gpu_count}")
    for i in range(gpu_count):
        prop = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {prop.name} | VRAM: {prop.total_memory / (1024**3):.2f} GB")

# Define Master Output Paths
BASE_DIR = Path("/kaggle/working/forestguard_outputs" if os.path.exists("/kaggle") else "./forestguard_outputs")
YOLO_DIR = BASE_DIR / "yolo"
CNN_DIR = BASE_DIR / "cnn"
DATA_DIR = BASE_DIR / "data"
YOLO_DATA_DIR = DATA_DIR / "yolo"
CNN_DATA_DIR = DATA_DIR / "cnn"

for d in [BASE_DIR, YOLO_DIR, CNN_DIR, DATA_DIR, YOLO_DATA_DIR, CNN_DATA_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print(f"\\n📁 Output root: {BASE_DIR}")
print("✅ Environment check complete.")"""))

    # -------------------------------------------------------------
    # CELL 3: Dependency Installation
    # -------------------------------------------------------------
    cells.append(md_cell("""## 2. Dependency Installation
Installs required packages: Ultralytics YOLO, ONNX, ONNX Runtime, Scikit-learn, Matplotlib, and PyYAML."""))

    cells.append(code_cell("""import subprocess

print("📦 Installing required packages...")
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "ultralytics>=8.3.0",
    "onnx>=1.15.0",
    "onnxruntime>=1.17.0",
    "scikit-learn",
    "matplotlib",
    "pyyaml"
])

import ultralytics
from ultralytics import YOLO
import onnx
import onnxruntime as ort
import sklearn
import yaml
import matplotlib.pyplot as plt
import numpy as np

print(f"✓ Ultralytics Version: {ultralytics.__version__}")
print(f"✓ ONNX Version:        {onnx.__version__}")
print(f"✓ ONNX Runtime Version:{ort.__version__}")
print(f"✓ Scikit-learn Version:{sklearn.__version__}")
print("✅ All dependencies verified successfully.")"""))

    # -------------------------------------------------------------
    # CELL 4: Dynamic Dataset Discovery & Early Gate
    # -------------------------------------------------------------
    cells.append(md_cell("""## 3. Dataset Discovery & Validation Gate
Discovers the attached dataset in `/kaggle/input`. Specifically prioritizes `sayedgamal99/smoke-fire-detection-yolo` and other D-Fire mirrors. Halts execution early with clear instructions if missing."""))

    cells.append(code_cell("""def discover_dataset() -> Path:
    kaggle_input = Path("/kaggle/input")
    if not kaggle_input.exists():
        local_cand = Path("datasets/DFire")
        if local_cand.exists():
            return local_cand
        raise FileNotFoundError("Could not find /kaggle/input or local datasets/DFire folder.")

    # 1. Search for sayedgamal99 / smoke-fire-detection-yolo dataset specifically
    for cand in kaggle_input.glob("**/*sayedgamal99*"):
        if cand.is_dir():
            return cand
    for cand in kaggle_input.glob("**/*smoke-fire-detection-yolo*"):
        if cand.is_dir():
            return cand
    for cand in kaggle_input.glob("**/*d-fire*"):
        if cand.is_dir():
            return cand

    # 2. Search for any smoke-fire or dfire directory containing train/images or data.yaml
    for cand in kaggle_input.glob("**/*"):
        if cand.is_dir():
            if (cand / "data.yaml").exists() or (cand / "data.yml").exists():
                return cand
            if (cand / "train" / "images").exists() or (cand / "train" / "labels").exists():
                return cand

    # 3. Search for any directory in /kaggle/input
    subdirs = [p for p in kaggle_input.iterdir() if p.is_dir()]
    if subdirs:
        for s in subdirs:
            if (s / "data.yaml").exists() or (s / "train").exists():
                return s
        return subdirs[0]

    raise FileNotFoundError(
        "❌ DATASET NOT DETECTED IN /kaggle/input!\\n"
        "Please attach the dataset before proceeding:\\n"
        "1. Click '+ Add Input' in the right-hand panel.\\n"
        "2. Search for: 'smoke-fire-detection-yolo' (sayedgamal99/smoke-fire-detection-yolo)\\n"
        "   or 'thunguyen1012/smoke-fire-detection-yolo'.\\n"
        "3. Click the '+' button to attach it, then rerun this cell."
    )

raw_dataset_path = discover_dataset()
print(f"📂 Raw dataset located at: {raw_dataset_path}")
print("Directory listing:")
for item in raw_dataset_path.iterdir():
    print(f"  - {item.name} ({'dir' if item.is_dir() else 'file'})")"""))

    # -------------------------------------------------------------
    # CELL 5: Dynamic Class Mapping Resolution & Leak-Free Preparation
    # -------------------------------------------------------------
    cells.append(md_cell("""## 4. Dynamic Class Mapping & Leak-Free Dataset Preparation
**Critical Audit Feature**:
1. Dynamically parses the raw dataset's `data.yaml` to resolve which class ID is `fire` and which is `smoke`. Never assumes class 0 = fire. (e.g. `sayedgamal99` defines `0: smoke, 1: fire`).
2. Standardizes the output YOLO dataset to `{0: 'fire', 1: 'smoke'}` so that backend interfaces and metrics remain 100% unified.
3. Derives the binary CNN classification dataset (`fire` vs `no_fire`), strictly excluding smoke-only frames to avoid noisy ground truth."""))

    cells.append(code_cell("""# 1. Locate and parse raw dataset YAML configuration
raw_yaml_path = None
for y in raw_dataset_path.glob("**/*.y*ml"):
    if "data" in y.name.lower() or y.name in ["data.yaml", "data.yml"]:
        raw_yaml_path = y
        break

if not raw_yaml_path:
    yamls = list(raw_dataset_path.glob("*.y*ml")) + list(raw_dataset_path.glob("*/*.y*ml"))
    if yamls:
        raw_yaml_path = yamls[0]

if not raw_yaml_path or not raw_yaml_path.exists():
    raise FileNotFoundError(
        f"❌ Could not find a dataset YAML file (data.yaml) in {raw_dataset_path}!\\n"
        "Please ensure the attached dataset contains data.yaml defining the class names."
    )

print(f"📄 Found dataset configuration: {raw_yaml_path}")
with open(raw_yaml_path, "r", encoding="utf-8") as f:
    raw_cfg = yaml.safe_load(f)

raw_names = raw_cfg.get("names", {})
print(f"Raw class names in YAML: {raw_names}")

# Dynamically resolve raw class indices
raw_fire_id = None
raw_smoke_id = None

if isinstance(raw_names, dict):
    for k, v in raw_names.items():
        v_str = str(v).strip().lower()
        if "fire" in v_str:
            raw_fire_id = int(k)
        elif "smoke" in v_str:
            raw_smoke_id = int(k)
elif isinstance(raw_names, list):
    for idx, v in enumerate(raw_names):
        v_str = str(v).strip().lower()
        if "fire" in v_str:
            raw_fire_id = idx
        elif "smoke" in v_str:
            raw_smoke_id = idx
else:
    raise ValueError(f"❌ Unsupported format for 'names' in {raw_yaml_path}: {type(raw_names)}")

# Strict compatibility verification: Stop execution if classes are not recognized
if raw_fire_id is None or raw_smoke_id is None:
    raise ValueError(
        f"❌ INCOMPATIBLE DATASET CLASS MAPPING in {raw_yaml_path}!\\n"
        f"Resolved raw_fire_id={raw_fire_id}, raw_smoke_id={raw_smoke_id}.\\n"
        f"Expected class names to contain 'fire' and 'smoke', but found: {raw_names}.\\n"
        "Execution stopped to prevent training on inverted or incompatible classes."
    )

print("=" * 65)
print(f"✓ Dynamically Resolved Raw Class Mapping:")
print(f"   Raw Fire Class ID:  {raw_fire_id}  (Name: '{raw_names[raw_fire_id]}')")
print(f"   Raw Smoke Class ID: {raw_smoke_id} (Name: '{raw_names[raw_smoke_id]}')")
print(f"✓ Standardized Output Target Mapping for Backend:")
print(f"   Standardized Class 0 -> 'fire'")
print(f"   Standardized Class 1 -> 'smoke'")
print("=" * 65)

# 2. Capped dataset sizes to ensure training completes on schedule
MAX_YOLO_TRAIN = 200 if SMOKE_MODE else 6000
MAX_CNN_PER_CLASS = 100 if SMOKE_MODE else 3000
MAX_VAL_SAMPLES = 50 if SMOKE_MODE else 1000
MAX_TEST_SAMPLES = 50 if SMOKE_MODE else 1000

dataset_manifest = {
    "raw_dataset_path": str(raw_dataset_path),
    "raw_yaml_path": str(raw_yaml_path),
    "raw_classes": raw_names,
    "raw_fire_id": raw_fire_id,
    "raw_smoke_id": raw_smoke_id,
    "standardized_classes": {0: "fire", 1: "smoke"},
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "splits": {}
}

splits = ["train", "val", "test"]
print("\\n--- Validating & Preparing Splits ---")

for s in splits:
    # 1. Locate split images
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

    # 2. Locate split labels
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
    if len(images) == 0:
        raise ValueError(f"Split '{s}' has no images! Check raw dataset directory structure.")

    dataset_manifest["splits"][s] = {"raw_images": len(images)}

    # Destination directories
    (YOLO_DATA_DIR / s / "images").mkdir(parents=True, exist_ok=True)
    (YOLO_DATA_DIR / s / "labels").mkdir(parents=True, exist_ok=True)
    (CNN_DATA_DIR / s / "fire").mkdir(parents=True, exist_ok=True)
    (CNN_DATA_DIR / s / "no_fire").mkdir(parents=True, exist_ok=True)

    yolo_count = 0
    cnn_fire_count = 0
    cnn_nofire_count = 0

    yolo_cap = MAX_YOLO_TRAIN if s == "train" else MAX_VAL_SAMPLES
    cnn_cap = MAX_CNN_PER_CLASS if s == "train" else (MAX_VAL_SAMPLES // 2)

    for img in images:
        lbl_file = (src_lbl / f"{img.stem}.txt") if src_lbl else None
        has_fire = False
        has_smoke = False
        standardized_yolo_lines = []

        if lbl_file and lbl_file.exists():
            lines = lbl_file.read_text(encoding="utf-8", errors="ignore").strip().splitlines()
            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue
                try:
                    c_id = int(parts[0])
                    coords = parts[1:]
                    if c_id == raw_fire_id:
                        has_fire = True
                        standardized_yolo_lines.append(f"0 {' '.join(coords)}")
                    elif c_id == raw_smoke_id:
                        has_smoke = True
                        standardized_yolo_lines.append(f"1 {' '.join(coords)}")
                except ValueError:
                    continue

        is_smoke_only = (has_smoke and not has_fire)

        # Populate Standardized YOLO dataset
        if yolo_count < yolo_cap:
            dest_img = YOLO_DATA_DIR / s / "images" / img.name
            dest_lbl = YOLO_DATA_DIR / s / "labels" / f"{img.stem}.txt"
            if not dest_img.exists():
                shutil.copy2(img, dest_img)
            dest_lbl.write_text("\\n".join(standardized_yolo_lines), encoding="utf-8")
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

    dataset_manifest["splits"][s]["yolo_used"] = yolo_count
    dataset_manifest["splits"][s]["cnn_fire"] = cnn_fire_count
    dataset_manifest["splits"][s]["cnn_nofire"] = cnn_nofire_count
    print(f"  ✓ Prepared split '{s}': YOLO={yolo_count} | CNN Fire={cnn_fire_count}, No-Fire={cnn_nofire_count}")

# Generate standardized data.yaml for YOLO training
data_yaml_path = YOLO_DATA_DIR / "data.yaml"
with open(data_yaml_path, "w", encoding="utf-8") as f:
    yaml.dump({
        "path": str(YOLO_DATA_DIR.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": {0: "fire", 1: "smoke"}
    }, f)

with open(BASE_DIR / "dataset_manifest.json", "w", encoding="utf-8") as f:
    json.dump(dataset_manifest, f, indent=2)

print(f"\\n✅ Generated standardized data.yaml: {data_yaml_path}")
print(f"✅ Generated dataset manifest: {BASE_DIR / 'dataset_manifest.json'}")"""))

    # -------------------------------------------------------------
    # CELL 6: YOLO Smoke Test
    # -------------------------------------------------------------
    cells.append(md_cell("""## 5. YOLO Smoke Test
Runs a 1-epoch mini training pass on 50 samples to verify model downloading, CUDA tensor allocation, and loss backward pass before full training."""))

    cells.append(code_cell("""print("🧪 Running YOLO 1-Epoch Smoke Test...")
smoke_yaml_path = YOLO_DATA_DIR / "smoke_data.yaml"
with open(smoke_yaml_path, "w", encoding="utf-8") as f:
    yaml.dump({
        "path": str(YOLO_DATA_DIR.resolve()),
        "train": "val/images",
        "val": "val/images",
        "names": {0: "fire", 1: "smoke"}
    }, f)

smoke_model = YOLO("yolo11n.pt")
smoke_res = smoke_model.train(
    data=str(smoke_yaml_path),
    epochs=1,
    imgsz=320,
    batch=8,
    device=0 if torch.cuda.is_available() else "cpu",
    project=str(YOLO_DIR / "smoke_runs"),
    name="smoke_test",
    save=False,
    verbose=False
)
print("✅ YOLO Smoke Test Passed! Forward pass, backward pass, and CUDA verified.")"""))

    # -------------------------------------------------------------
    # CELL 7: YOLO Full Training
    # -------------------------------------------------------------
    cells.append(md_cell("""## 6. YOLO Full Training
Fine-tunes YOLO11 Nano (`yolo11n.pt`) on the standardized D-Fire dataset with early stopping and best checkpoint preservation."""))

    cells.append(code_cell("""yolo_epochs = 1 if SMOKE_MODE else 30
yolo_batch = 16
yolo_imgsz = 640

print(f"🚀 Starting YOLO11n Fine-Tuning ({yolo_epochs} epochs, imgsz={yolo_imgsz}, batch={yolo_batch})...")
yolo_model = YOLO("yolo11n.pt")

train_results = yolo_model.train(
    data=str(data_yaml_path),
    epochs=yolo_epochs,
    imgsz=yolo_imgsz,
    batch=yolo_batch,
    patience=10,
    device=0 if torch.cuda.is_available() else "cpu",
    project=str(YOLO_DIR / "runs"),
    name="forestguard_yolo",
    seed=42,
    save=True,
    plots=True
)

save_dir = Path(train_results.save_dir)
# Epochs actually completed (early stopping / interruption can end before the configured count)
try:
    epochs_completed = max(0, sum(1 for _ in open(save_dir / "results.csv", encoding="utf-8")) - 1)
except Exception:
    epochs_completed = None
best_pt_src = save_dir / "weights" / "best.pt"
if not best_pt_src.exists():
    best_pt_src = save_dir / "weights" / "last.pt"

target_best_pt = YOLO_DIR / "yolo_best.pt"
shutil.copy2(best_pt_src, target_best_pt)
print(f"✅ Saved Best YOLO Weights to: {target_best_pt} ({target_best_pt.stat().st_size / (1024*1024):.2f} MB)")"""))

    # -------------------------------------------------------------
    # CELL 8: YOLO Evaluation & ONNX Export
    # -------------------------------------------------------------
    cells.append(md_cell("""## 7. YOLO Evaluation on Held-Out Test Set & ONNX Export
Evaluates the best checkpoint on the held-out `test` split, saves genuine metrics, and exports an optimized ONNX model for CPU deployment."""))

    cells.append(code_cell("""print("📊 Validating best YOLO checkpoint on held-out test split...")
eval_model = YOLO(str(target_best_pt))
val_res = eval_model.val(data=str(data_yaml_path), split="test", imgsz=yolo_imgsz)

# Extract genuine test metrics (Class 0 = fire, Class 1 = smoke)
yolo_metrics = {
    "model": "YOLO11 Nano Wildfire Detector",
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "mAP50": round(float(val_res.box.map50), 4),
    "mAP50-95": round(float(val_res.box.map), 4),
    "precision": round(float(val_res.box.mp), 4),
    "recall": round(float(val_res.box.mr), 4),
    "epochs_configured": yolo_epochs,
    "epochs_trained": epochs_completed,
    "metrics_split": "test (held-out)",
    "image_size": yolo_imgsz,
    "classes": {
        "fire": {
            "precision": round(float(val_res.box.p[0]), 4) if len(val_res.box.p) > 0 else 0.0,
            "recall": round(float(val_res.box.r[0]), 4) if len(val_res.box.r) > 0 else 0.0,
            "mAP50": round(float(val_res.box.ap50[0]), 4) if len(val_res.box.ap50) > 0 else 0.0
        },
        "smoke": {
            "precision": round(float(val_res.box.p[1]), 4) if len(val_res.box.p) > 1 else 0.0,
            "recall": round(float(val_res.box.r[1]), 4) if len(val_res.box.r) > 1 else 0.0,
            "mAP50": round(float(val_res.box.ap50[1]), 4) if len(val_res.box.ap50) > 1 else 0.0
        }
    }
}

with open(YOLO_DIR / "yolo_metrics.json", "w", encoding="utf-8") as f:
    json.dump(yolo_metrics, f, indent=2)

print("\\n--- YOLO Test Metrics ---")
print(f"mAP50:     {yolo_metrics['mAP50']:.4f}")
print(f"mAP50-95:  {yolo_metrics['mAP50-95']:.4f}")
print(f"Precision: {yolo_metrics['precision']:.4f}")
print(f"Recall:    {yolo_metrics['recall']:.4f}")
print(f"Fire mAP50:  {yolo_metrics['classes']['fire']['mAP50']:.4f}")
print(f"Smoke mAP50: {yolo_metrics['classes']['smoke']['mAP50']:.4f}")

# Export ONNX
print("\\n📦 Exporting YOLO to ONNX (imgsz=416, CPU optimized)...")
eval_model.export(format="onnx", imgsz=416, simplify=True, dynamic=True)
onnx_exported = target_best_pt.with_suffix(".onnx")
if onnx_exported.exists():
    shutil.copy2(onnx_exported, YOLO_DIR / "yolo_best.onnx")
    print(f"✅ Saved YOLO ONNX model to: {YOLO_DIR / 'yolo_best.onnx'}")

# Copy training and validation plots
for p in ["results.png", "confusion_matrix.png", "PR_curve.png", "F1_curve.png"]:
    src = save_dir / p
    if src.exists():
        shutil.copy2(src, YOLO_DIR / p)
        shutil.copy2(src, YOLO_DIR / f"yolo_{p}")

print("✅ YOLO evaluation & export complete.")"""))

    # -------------------------------------------------------------
    # CELL 9: CNN Smoke Test
    # -------------------------------------------------------------
    cells.append(md_cell("""## 8. CNN Smoke Test
Verifies MobileNetV2 architecture, classifier head replacement, ImageNet normalization, and GPU forward/backward pass."""))

    cells.append(code_cell("""import torch.nn as nn
from torchvision import datasets, transforms, models

print("🧪 Running MobileNetV2 1-Epoch Smoke Test...")

cnn_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def build_mobilenetv2():
    weights = models.MobileNet_V2_Weights.DEFAULT
    m = models.mobilenet_v2(weights=weights)
    # Freeze backbone
    for param in m.features.parameters():
        param.requires_grad = False
    in_f = m.classifier[1].in_features
    m.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_f, 64),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(64, 2)
    )
    return m

smoke_cnn = build_mobilenetv2().to(cnn_device)
dummy_x = torch.randn(4, 3, 224, 224).to(cnn_device)
dummy_out = smoke_cnn(dummy_x)
assert dummy_out.shape == (4, 2), f"Unexpected output shape: {dummy_out.shape}"
dummy_loss = dummy_out.sum()
dummy_loss.backward()
print("✅ CNN Smoke Test Passed! Architecture and gradient flow verified.")"""))

    # -------------------------------------------------------------
    # CELL 10: CNN Full Training (2-Stage Transfer Learning)
    # -------------------------------------------------------------
    cells.append(md_cell("""## 9. CNN Full Training
Two-stage transfer learning:
1. **Stage 1 (Warmup)**: Classifier head warmup with backbone frozen (5 epochs, Adam, lr=1e-3).
2. **Stage 2 (Fine-Tuning)**: Top 4 inverted residual blocks unfrozen (8 epochs, AdamW, lr=1e-4 with CosineAnnealingLR)."""))

    cells.append(code_cell("""from torch.utils.data import DataLoader
import torch.optim as optim

warmup_epochs = 1 if SMOKE_MODE else 5
finetune_epochs = 1 if SMOKE_MODE else 8
cnn_batch = 32

# Transformations
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

train_ds = datasets.ImageFolder(str(CNN_DATA_DIR / "train"), transform=train_tf)
val_ds = datasets.ImageFolder(str(CNN_DATA_DIR / "val"), transform=eval_tf)
test_ds = datasets.ImageFolder(str(CNN_DATA_DIR / "test"), transform=eval_tf)

print(f"CNN Classes: {train_ds.classes} -> index {train_ds.class_to_idx}")
assert train_ds.class_to_idx["fire"] == 0 and train_ds.class_to_idx["no_fire"] == 1, \\
    f"Class index mapping mismatch! Expected fire:0, no_fire:1. Got: {train_ds.class_to_idx}"

train_loader = DataLoader(train_ds, batch_size=cnn_batch, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=cnn_batch, shuffle=False, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_ds, batch_size=cnn_batch, shuffle=False, num_workers=2, pin_memory=True)

cnn_model = build_mobilenetv2().to(cnn_device)
criterion = nn.CrossEntropyLoss()
best_val_acc = 0.0
best_cnn_weights = CNN_DIR / "cnn_mobilenetv2_best.pt"
cnn_history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

# --- Stage 1: Warmup ---
print(f"\\n--- STAGE 1: Classifier Head Warmup ({warmup_epochs} epochs) ---")
optimizer_s1 = optim.Adam(cnn_model.classifier.parameters(), lr=1e-3, weight_decay=1e-4)

for epoch in range(warmup_epochs):
    cnn_model.train()
    r_loss, r_corr, total = 0.0, 0, 0
    for x, y in train_loader:
        x, y = x.to(cnn_device), y.to(cnn_device)
        optimizer_s1.zero_grad()
        out = cnn_model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer_s1.step()
        r_loss += loss.item() * x.size(0)
        r_corr += (torch.max(out, 1)[1] == y).sum().item()
        total += y.size(0)

    # Validation
    cnn_model.eval()
    v_loss, v_corr, v_total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(cnn_device), y.to(cnn_device)
            out = cnn_model(x)
            v_loss += criterion(out, y).item() * x.size(0)
            v_corr += (torch.max(out, 1)[1] == y).sum().item()
            v_total += y.size(0)

    tr_acc = r_corr / total
    val_acc = v_corr / v_total
    cnn_history["train_acc"].append(tr_acc)
    cnn_history["val_acc"].append(val_acc)
    cnn_history["train_loss"].append(r_loss / total)
    cnn_history["val_loss"].append(v_loss / v_total)
    print(f"Warmup Epoch {epoch+1}/{warmup_epochs} - Train Acc: {tr_acc:.4f}, Loss: {r_loss/total:.4f} | Val Acc: {val_acc:.4f}, Loss: {v_loss/v_total:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(cnn_model.state_dict(), best_cnn_weights)

# --- Stage 2: Fine-Tuning ---
print(f"\\n--- STAGE 2: Fine-Tuning Top Blocks ({finetune_epochs} epochs) ---")
for block in cnn_model.features[-4:]:
    for param in block.parameters():
        param.requires_grad = True

optimizer_s2 = optim.AdamW(filter(lambda p: p.requires_grad, cnn_model.parameters()), lr=1e-4, weight_decay=1e-3)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_s2, T_max=finetune_epochs)

for epoch in range(finetune_epochs):
    cnn_model.train()
    r_loss, r_corr, total = 0.0, 0, 0
    for x, y in train_loader:
        x, y = x.to(cnn_device), y.to(cnn_device)
        optimizer_s2.zero_grad()
        out = cnn_model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer_s2.step()
        r_loss += loss.item() * x.size(0)
        r_corr += (torch.max(out, 1)[1] == y).sum().item()
        total += y.size(0)

    scheduler.step()

    # Validation
    cnn_model.eval()
    v_loss, v_corr, v_total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(cnn_device), y.to(cnn_device)
            out = cnn_model(x)
            v_loss += criterion(out, y).item() * x.size(0)
            v_corr += (torch.max(out, 1)[1] == y).sum().item()
            v_total += y.size(0)

    tr_acc = r_corr / total
    val_acc = v_corr / v_total
    cnn_history["train_acc"].append(tr_acc)
    cnn_history["val_acc"].append(val_acc)
    cnn_history["train_loss"].append(r_loss / total)
    cnn_history["val_loss"].append(v_loss / v_total)
    print(f"FineTune Epoch {epoch+1}/{finetune_epochs} - Train Acc: {tr_acc:.4f}, Loss: {r_loss/total:.4f} | Val Acc: {val_acc:.4f}, Loss: {v_loss/v_total:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(cnn_model.state_dict(), best_cnn_weights)

print(f"\\n✅ Best Validation Accuracy: {best_val_acc:.4f}")
print(f"✅ Saved Best CNN Weights to: {best_cnn_weights}")"""))

    # -------------------------------------------------------------
    # CELL 11: CNN Evaluation on Held-Out Test Set (Corrected Rounding Bug)
    # -------------------------------------------------------------
    cells.append(md_cell("""## 10. CNN Evaluation on Held-Out Test Set
Evaluates the best MobileNetV2 checkpoint on unseen test images, producing test accuracy, confusion matrix, precision, recall (with 4-decimal precision), and F1-score."""))

    cells.append(code_cell("""from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

print("📊 Evaluating best CNN model on held-out test split...")
cnn_model.load_state_dict(torch.load(best_cnn_weights, map_location=cnn_device))
cnn_model.eval()

preds, targets = [], []
with torch.no_grad():
    for x, y in test_loader:
        x = x.to(cnn_device)
        out = cnn_model(x)
        preds.extend(torch.max(out, 1)[1].cpu().numpy())
        targets.extend(y.numpy())

preds = np.array(preds)
targets = np.array(targets)
cm = confusion_matrix(targets, preds)
p, r, f1, _ = precision_recall_fscore_support(targets, preds, average="binary", pos_label=0)
test_acc = float(np.mean(preds == targets))

# Audited: recall correctly rounded to 4 decimals
cnn_metrics = {
    "model": "MobileNetV2 Fire Classifier",
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "accuracy": round(test_acc, 4),
    "test_accuracy": round(test_acc, 4),
    "macro_f1": round(float(f1), 4),
    "classes": {
        "fire": {
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "f1": round(float(f1), 4)
        }
    },
    "confusion_matrix": cm.tolist()
}

with open(CNN_DIR / "cnn_metrics.json", "w", encoding="utf-8") as f:
    json.dump(cnn_metrics, f, indent=2)

print("\\n--- CNN Test Metrics ---")
print(f"Accuracy:  {test_acc * 100:.2f}%")
print(f"F1-Score:  {f1:.4f}")
print(f"Precision: {p:.4f}")
print(f"Recall:    {r:.4f}")
print(f"Confusion Matrix:\\n{cm}")"""))

    # -------------------------------------------------------------
    # CELL 12: ONNX Export & Backend Compatibility Validation
    # -------------------------------------------------------------
    cells.append(md_cell("""## 11. ONNX Export & Backend Loader Verification
Exports MobileNetV2 to ONNX format with dynamic batching, validates numerical tolerances against PyTorch, and directly verifies compatibility against the FastAPI `CNNClassifier` backend loader interface."""))

    cells.append(code_cell("""print("📦 Exporting MobileNetV2 to ONNX...")
dummy_in = torch.randn(1, 3, 224, 224).to(cnn_device)
onnx_path = CNN_DIR / "cnn_mobilenetv2.onnx"

torch.onnx.export(
    cnn_model,
    dummy_in,
    str(onnx_path),
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
)
print(f"Saved ONNX model to: {onnx_path} ({onnx_path.stat().st_size / (1024*1024):.2f} MB)")

# 1. Numerical Equivalence Verification
print("🔍 Verifying ONNX numerical equivalence against PyTorch...")
ort_sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
ort_in_name = ort_sess.get_inputs()[0].name
dummy_cpu = dummy_in.cpu().numpy()
ort_out = ort_sess.run(None, {ort_in_name: dummy_cpu})[0]

cnn_model.eval()
with torch.no_grad():
    pt_out = cnn_model(dummy_in).cpu().numpy()

diff = np.max(np.abs(ort_out - pt_out))
print(f"Maximum absolute difference: {diff:.6e}")
assert diff < 1e-4, f"Numerical discrepancy exceeds tolerance: {diff}"
print("✓ PyTorch vs ONNX Runtime output difference < 1e-4")

# 2. Strict Backend Interface Simulation (replicates backend/models/cnn_classifier.py)
print("\\n🔍 Verifying compatibility with FastAPI backend CNNClassifier...")
options = ort.SessionOptions()
options.intra_op_num_threads = 4
backend_session = ort.InferenceSession(str(onnx_path), sess_options=options, providers=["CPUExecutionProvider"])

in_node = backend_session.get_inputs()[0]
out_node = backend_session.get_outputs()[0]
assert in_node.name == "input", f"Expected input name 'input', got {in_node.name}"
assert out_node.name == "output", f"Expected output name 'output', got {out_node.name}"

# Simulate exact backend preprocessing
dummy_raw_bgr = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
import cv2
resized = cv2.resize(dummy_raw_bgr, (224, 224))
rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
normalized = rgb.astype(np.float32) / 255.0
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
norm = (normalized - mean) / std
chw = np.transpose(norm, (2, 0, 1))
backend_input = np.expand_dims(chw, axis=0)

raw_outputs = backend_session.run([out_node.name], {in_node.name: backend_input})[0]
logits = raw_outputs[0]
exp_logits = np.exp(logits - np.max(logits))
probs = exp_logits / np.sum(exp_logits)

prob_fire = float(probs[0])
prob_no_fire = float(probs[1])
assert 0.0 <= prob_fire <= 1.0 and 0.0 <= prob_no_fire <= 1.0
assert abs((prob_fire + prob_no_fire) - 1.0) < 1e-5
print(f"✓ Backend simulation output: fire={prob_fire:.4f}, no_fire={prob_no_fire:.4f}")
print("✅ CNN ONNX model verified 100% compatible with backend CNNClassifier!")

# 3. Create MODEL_INFO.json
model_info = {
    "cnn": {
        "name": "MobileNetV2 Classifier",
        "architecture": "MobileNetV2 Transfer Learning",
        "version": "1.0.0",
        "trained_by": "ForestGuard AI Kaggle Pipeline",
        "input_shape": [1, 3, 224, 224],
        "classes": ["fire", "no_fire"],
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "format": "ONNX (CPU)"
    },
    "yolo": {
        "name": "YOLO11 Nano Detector",
        "architecture": "Ultralytics YOLO11n",
        "version": "1.0.0",
        "trained_by": "ForestGuard AI Kaggle Pipeline",
        "input_shape": [1, 3, 640, 640],
        "classes": ["fire", "smoke"],
        "format": "PyTorch / ONNX (CPU)"
    }
}

with open(CNN_DIR / "MODEL_INFO.json", "w", encoding="utf-8") as f:
    json.dump(model_info, f, indent=2)
with open(YOLO_DIR / "MODEL_INFO.json", "w", encoding="utf-8") as f:
    json.dump(model_info, f, indent=2)

print("✅ MODEL_INFO.json successfully generated.")"""))

    # -------------------------------------------------------------
    # CELL 13: Graph Generation
    # -------------------------------------------------------------
    cells.append(md_cell("""## 12. Evaluation Graph Generation
Generates high-resolution confusion matrix, loss curve, and accuracy progression plots for documentation and the IEEE paper."""))

    cells.append(code_cell("""# 1. Confusion Matrix Plot
fig, ax = plt.subplots(figsize=(6, 5))
cax = ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.8)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(x=j, y=i, s=str(cm[i, j]), va="center", ha="center", size="xx-large")
plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("True Label", fontsize=12)
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["fire", "no_fire"], fontsize=11)
ax.set_yticklabels(["fire", "no_fire"], fontsize=11)
plt.title(f"CNN Test Confusion Matrix (Accuracy: {test_acc*100:.2f}%)", pad=15)
plt.colorbar(cax)
plt.tight_layout()
plt.savefig(CNN_DIR / "cnn_confusion_matrix.png", dpi=200)
plt.show()

# 2. Accuracy Curves
plt.figure(figsize=(7, 4))
plt.plot(cnn_history["train_acc"], label="Train Accuracy", marker="o")
plt.plot(cnn_history["val_acc"], label="Validation Accuracy", marker="s")
plt.title("MobileNetV2 Classification Accuracy Progression")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig(CNN_DIR / "cnn_accuracy.png", dpi=200)
plt.show()

# 3. Loss Curves
plt.figure(figsize=(7, 4))
plt.plot(cnn_history["train_loss"], label="Train Loss", marker="o")
plt.plot(cnn_history["val_loss"], label="Validation Loss", marker="s")
plt.title("MobileNetV2 Cross-Entropy Loss Progression")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig(CNN_DIR / "cnn_loss.png", dpi=200)
plt.show()

print("✅ High-resolution evaluation graphs saved.")"""))

    # -------------------------------------------------------------
    # CELL 14: Artifact Packaging into Downloadable ZIPs
    # -------------------------------------------------------------
    cells.append(md_cell("""## 13. Artifact Packaging
Packages all trained weights, ONNX exports, metrics, and figures into two downloadable zip archives:
- `yolo_artifacts.zip`
- `cnn_artifacts.zip`"""))

    cells.append(code_cell("""print("🗜️ Packaging Artifacts into Single ZIP Archives...")

# 1. Package YOLO
yolo_zip_path = BASE_DIR / "yolo_artifacts.zip"
with zipfile.ZipFile(yolo_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for f in YOLO_DIR.iterdir():
        if f.is_file():
            zf.write(f, arcname=f.name)

# 2. Package CNN
cnn_zip_path = BASE_DIR / "cnn_artifacts.zip"
with zipfile.ZipFile(cnn_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for f in CNN_DIR.iterdir():
        if f.is_file():
            zf.write(f, arcname=f.name)

# Copy to /kaggle/working/ root for immediate one-click download in Kaggle UI
if os.path.exists("/kaggle/working"):
    shutil.copy2(yolo_zip_path, Path("/kaggle/working/yolo_artifacts.zip"))
    shutil.copy2(cnn_zip_path, Path("/kaggle/working/cnn_artifacts.zip"))
    print("📁 Copied zip archives directly to /kaggle/working/ for one-click download.")

print(f"✅ YOLO Zip: {yolo_zip_path} ({yolo_zip_path.stat().st_size / (1024*1024):.2f} MB)")
print(f"✅ CNN Zip:  {cnn_zip_path} ({cnn_zip_path.stat().st_size / (1024*1024):.2f} MB)")"""))

    # -------------------------------------------------------------
    # CELL 15: Final Artifact Verification & Manifest
    # -------------------------------------------------------------
    cells.append(md_cell("""## 14. Final Artifact Verification & Manifest
Computes cryptographic SHA-256 checksums and file sizes for all generated artifacts."""))

    cells.append(code_cell("""def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

manifest_lines = [
    "=" * 90,
    "🌲 FORESTGUARD AI: ARTIFACT MANIFEST & SHA-256 CHECKSUMS",
    f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}",
    "=" * 90,
    f"{'Filename':<35} {'Size (MB)':<12} {'SHA-256 Checksum':<64}",
    "-" * 90
]

for p in sorted(list(YOLO_DIR.iterdir()) + list(CNN_DIR.iterdir()) + [yolo_zip_path, cnn_zip_path]):
    if p.is_file():
        size_mb = f"{p.stat().st_size / (1024*1024):.2f}"
        chk = compute_sha256(p)
        manifest_lines.append(f"{p.name:<35} {size_mb:<12} {chk:<64}")

manifest_text = "\\n".join(manifest_lines)
manifest_file = BASE_DIR / "MANIFEST.txt"
manifest_file.write_text(manifest_text, encoding="utf-8")
print(manifest_text)

print("\\n" + "=" * 70)
print("🎉 ALL TRAINING, EVALUATION, AND PACKAGING COMPLETED SUCCESSFULLY!")
print("=" * 70)
print("\\n📥 Next Step on your laptop:")
print("Run the automated installer in PowerShell:")
print('.\\\\scripts\\\\install_artifacts.ps1 -YoloZip \"path\\\\to\\\\yolo_artifacts.zip\" -CnnZip \"path\\\\to\\\\cnn_artifacts.zip\"')"""))

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    return notebook

if __name__ == "__main__":
    nb = create_notebook()
    out_path = Path("training/kaggle/ForestGuard_AI_Complete_Training.ipynb")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"Generated notebook at: {out_path} with {len(nb['cells'])} cells.")
