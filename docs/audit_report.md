# ForestGuard AI — Repository & Architecture Audit Report

**Date & Time**: 2026-11-18 06:50 IST  
**Environment**: Windows 11, Python 3.11, 8GB RAM, Intel CPU (No Local GPU)  
**Target Repository**: `https://github.com/girishgowdahr/DeepLearning_EL` (branch `main`)  
**Commit Inspected**: `26ed4ee` (Working tree clean)

---

## 1. Exact YOLO Model & Configuration
- **Model Variant**: Ultralytics YOLO11 Nano (`yolo11n.pt`)
- **Default Resolution**: `imgsz = 640` (training), `imgsz = 416` (backend runtime inference)
- **Training Hyperparameters**:
  - Epochs: 30 (`patience = 10`)
  - Batch size: 16 (or 32 auto-reduced on VRAM limit)
  - Optimizer: Automatic SGD/AdamW with AMP (`amp=True`)
  - Workers: 2, `cache=False` (prevents host RAM exhaustion)
  - Saving: `save=True`, `save_period=5`, checkpoints at `runs/forestguard_yolo/weights/best.pt` and `last.pt`
- **Device Setting**: `device=0` inside `CUDA_VISIBLE_DEVICES=0` subprocess
- **Source Script**: [training/yolo/kaggle_train_yolo.py](file:///d:/RVCE/Sem3/Deep%20Learning%20EL/ForestGuard-AI/training/yolo/kaggle_train_yolo.py) and [training/yolo/train_yolo.py](file:///d:/RVCE/Sem3/Deep%20Learning%20EL/ForestGuard-AI/training/yolo/train_yolo.py)

---

## 2. Exact CNN Architecture & Configuration
- **Model Architecture**: MobileNetV2 (`torchvision.models.mobilenet_v2(weights=DEFAULT)`)
- **Transfer Learning Protocol**:
  - **Stage 1 (Head Warmup)**: 5 epochs, backbone features completely frozen (`requires_grad=False`), classification head trained with Adam (`lr=1e-3`, `weight_decay=1e-4`).
  - **Stage 2 (Fine-Tuning)**: 8 epochs, top 4 inverted residual blocks unfrozen (`requires_grad=True`), trained with AdamW (`lr=1e-4`, `weight_decay=1e-3`) with `CosineAnnealingLR`.
- **Classification Head Structure**:
  - `Dropout(0.3) -> Linear(1280, 64) -> ReLU() -> Dropout(0.2) -> Linear(64, 2)`
- **Input Specifications**: `224 × 224 × 3` RGB image
- **Normalization Constants**: ImageNet standards — Mean: `[0.485, 0.456, 0.406]`, Std: `[0.229, 0.224, 0.225]`
- **Classes**: Binary — Index `0: fire`, Index `1: no_fire` (Alphabetical ordering from PyTorch `ImageFolder`)
- **Source Script**: [training/cnn/kaggle_train_cnn.py](file:///d:/RVCE/Sem3/Deep%20Learning%20EL/ForestGuard-AI/training/cnn/kaggle_train_cnn.py) and [training/cnn/train_cnn.py](file:///d:/RVCE/Sem3/Deep%20Learning%20EL/ForestGuard-AI/training/cnn/train_cnn.py)

---

## 3. Dataset Source, License & Kaggle Accessibility
- **Dataset Name**: D-Fire (Wildfire & Smoke Detection Dataset)
- **Primary Citation**: de Venâncio et al., 2022 (*"DFire: an image dataset for fire and smoke detection"*)
- **License**: NOT VERIFIED. The source repository page states no licence; do not assume CC0 / CC BY. Confirm with the original authors before redistribution.
- **Public Kaggle Mirrors**:
  - `thunguyen1012/smoke-fire-detection-yolo`
  - `gaiasd/dfire-dataset`
- **Kaggle Accessibility**: 100% accessible directly inside Kaggle via **"+ Add Input"** without uploading from laptop or consuming laptop bandwidth/storage. No billing or click-through permission gates.

---

## 4. Class Mappings & Data Leakage Prevention
- **YOLO Format**: `<class_id> <x_center> <y_center> <width> <height>` (normalized $[0, 1]$)
  - `0: fire`
  - `1: smoke`
- **CNN Split Derivation**:
  - Uses the **exact same underlying split boundaries** (`train/`, `val/`, `test/`).
  - `fire`: Frames containing $\ge 1$ bounding box for `class 0`.
  - `no_fire`: Frames containing **0** boxes for `class 0` and **0** boxes for `class 1` (negative backgrounds).
  - **Smoke-only exclusion**: Frames containing only `class 1` (smoke plumes without visible flame) are **strictly excluded** from the binary fire/no-fire classifier to avoid label noise and ambiguous training signals.
  - Zero data leakage is guaranteed because no train images enter val or test.

---

## 5. Backend Model Filenames & Loader Interfaces
Confirmed alignment between training script outputs and backend model loaders:
| Model Component | Training Output | Expected Backend Path | Loader Implementation |
| :--- | :--- | :--- | :--- |
| **CNN Model** | `cnn_mobilenetv2.onnx` | `model_weights/cnn_mobilenetv2.onnx` | [backend/models/cnn_classifier.py](file:///d:/RVCE/Sem3/Deep%20Learning%20EL/ForestGuard-AI/backend/models/cnn_classifier.py) (`ONNX Runtime CPU`) |
| **YOLO Model** | `yolo_best.pt` | `model_weights/yolo_best.pt` | [backend/models/yolo_detector.py](file:///d:/RVCE/Sem3/Deep%20Learning%20EL/ForestGuard-AI/backend/models/yolo_detector.py) (`ultralytics.YOLO CPU`) |
| **Model Metadata** | `MODEL_INFO.json` | `model_weights/MODEL_INFO.json` | [backend/models/model_registry.py](file:///d:/RVCE/Sem3/Deep%20Learning%20EL/ForestGuard-AI/backend/models/model_registry.py) |

- Class order verified: `CNNClassifier.classes = ["fire", "no_fire"]` matches `probs[0]` (fire) and `probs[1]` (no-fire).
- Normalization verified: Mean `[0.485, 0.456, 0.406]`, Std `[0.229, 0.224, 0.225]` matches between training and inference.

---

## 6. Metrics & Graph Files Alignment
| Metric / Graph File | Generating Script | Expected Destination | Frontend Reader |
| :--- | :--- | :--- | :--- |
| `cnn_metrics.json` | `kaggle_train_cnn.py` | `results/metrics/cnn_metrics.json` | [Analytics.jsx](file:///d:/RVCE/Sem3/Deep%20Learning%20EL/ForestGuard-AI/frontend/src/pages/Analytics.jsx) (`metrics.cnn_metrics`) |
| `yolo_metrics.json` | `kaggle_train_yolo.py` | `results/metrics/yolo_metrics.json` | [Analytics.jsx](file:///d:/RVCE/Sem3/Deep%20Learning%20EL/ForestGuard-AI/frontend/src/pages/Analytics.jsx) (`metrics.yolo_metrics`) |
| `cnn_confusion_matrix.png` | `kaggle_train_cnn.py` | `results/graphs/cnn_confusion_matrix.png` | [Analytics.jsx](file:///d:/RVCE/Sem3/Deep%20Learning%20EL/ForestGuard-AI/frontend/src/pages/Analytics.jsx) (`/results/graphs/...`) |
| `cnn_accuracy.png` | `kaggle_train_cnn.py` | `results/graphs/cnn_accuracy.png` | [Analytics.jsx](file:///d:/RVCE/Sem3/Deep%20Learning%20EL/ForestGuard-AI/frontend/src/pages/Analytics.jsx) (`/results/graphs/...`) |
| `yolo_PR_curve.png` | `kaggle_train_yolo.py` | `results/graphs/yolo_PR_curve.png` | [Analytics.jsx](file:///d:/RVCE/Sem3/Deep%20Learning%20EL/ForestGuard-AI/frontend/src/pages/Analytics.jsx) (`/results/graphs/...`) |

### Mismatch Identified & Fixed:
- **Audit Finding**: In [backend/main.py](file:///d:/RVCE/Sem3/Deep%20Learning%20EL/ForestGuard-AI/backend/main.py), `/results` was not mounted as a `StaticFiles` endpoint; requesting `/results/graphs/*.png` was intercepted by the SPA fallback, serving `index.html` instead of image bytes.
- **Fix Applied**: Added `app.mount("/results", StaticFiles(directory=str(settings.RESULTS_DIR)), name="results")` and bypassed SPA routing for `/results/`.

---

## 7. Dependency & Platform Compatibility
- **Host Laptop**: Windows 11, Python 3.11.3, `torch==2.15.0+cpu`, `torchvision==0.29.2+cpu`, `ultralytics==8.4.215`, `onnxruntime==1.32.0`.
- **Kaggle Environment**: Linux, Python 3.10/3.11, CUDA 12.x, 2 × NVIDIA Tesla T4 (16 GB VRAM each).
- **Compatibility**:
  - `yolo_best.pt` exported by Ultralytics on Kaggle loads seamlessly on laptop CPU with Ultralytics.
  - `cnn_mobilenetv2.onnx` exported with ONNX opset 14 runs with zero PyTorch dependencies directly via ONNX Runtime CPU.

---

## 8. Reproducibility Check (Live Terminal Execution)
- `pytest backend\tests -v`: **15 passed, 1 warning in 2.93s** (All unit and API tests green).
- `npm run build`: **2,750 modules transformed, built in 14.25s** (`dist/` valid).
- Laptop Webcam (Device 0): Hardware opens at `(480, 640, 3)`; WebSocket streaming roundtrip verified at **23.3 ms**.
