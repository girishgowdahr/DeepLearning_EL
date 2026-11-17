# ForestGuard AI — Project Progress & Live Checklist

## Phase 0: Preflight Verification (Completed)
- [x] Workspace initialization & repository structure created (`frontend/`, `backend/`, `training/`, `datasets/`, `model_weights/`, `results/`, `scripts/`, `docs/`)
- [x] `.gitignore` created to protect secrets, venv, large weights, datasets, and local database
- [x] Git repository initialized (`git init`)
- [x] System environment audited:
  - **Node.js**: `v24.21.0` (Installed)
  - **npm**: `11.19.0` (Installed)
  - **Git**: `2.54.0.windows.1` (Installed)
  - **Python**: `Python 3.11.3` (Installed at `C:\Users\admin\AppData\Local\Programs\Python\Python311\python.exe`)
  - **cloudflared**: Not currently installed. (Self-signed HTTPS fallback and IP-Webcam/LAN available; winget approval requested if tunnel desired)
- [x] Free-tier and platform verification (2026-11-17):
  - **Ultralytics Platform**: Free tier provides cloud storage, promotional GPU credits, and direct export/download of `best.pt` in ONNX and PT formats.
  - **Kaggle Notebooks**: Free 30 hours/week NVIDIA T4 x2 GPU (requires phone verification). Excellent reliability for running batch training scripts and downloading artifacts.
  - **Roboflow Universe**: Free tier includes 10 credits/month, supports public dataset exports via API/CLI in YOLOv8/YOLO11 format.
  - **D-Fire Dataset**: Available on GitHub (`https://github.com/gaiasd/DFireDataset`) and Kaggle mirrors under CC0 1.0 Universal / CC BY 4.0. Over 21,000 images with `fire` (0) and `smoke` (1) classes in YOLO format, plus negative backgrounds.
- [x] Training platform recommendation: **Kaggle Notebooks (T4 GPU)** as primary workhorse for reproducible headless training script execution; **Ultralytics Platform** as alternative for YOLO fine-tuning if preferred.

---

## Live Checklist

### Track A: Training & Data
- [x] Create `datasets/DATASET_INFO.md`
- [x] CNN dataset derivation script (`training/cnn/prepare_cnn_dataset.py`)
- [x] CNN training script (`training/cnn/train_cnn.py`) MobileNetV2 transfer learning
- [x] CNN evaluation script (`training/cnn/evaluate_cnn.py`)
- [x] CNN ONNX export script (`training/cnn/export_onnx.py`)
- [x] YOLO dataset preparation script (`training/yolo/prepare_yolo_dataset.py`) + `data.yaml`
- [x] YOLO fine-tuning script (`training/yolo/train_yolo.py`) (nano model)
- [x] Cloud GPU training instructions (`training/cnn/README.md`, `training/yolo/README.md`)
- [ ] Execute cloud training run on Kaggle/Ultralytics & download weights into `model_weights/`

### Track B: Backend (FastAPI)
- [x] `backend/requirements.txt` & `backend/.env.example` & `backend/.env`
- [x] `backend/config.py` (settings & env vars)
- [x] `backend/database/schema.sql` & `backend/database/db.py` (SQLite)
- [x] `backend/models/cnn_classifier.py` (ONNX runtime CPU)
- [x] `backend/models/yolo_detector.py` (Ultralytics CPU)
- [x] `backend/models/model_registry.py` (status & metadata)
- [x] `backend/services/decision_engine.py` (fusion logic, temporal sliding window)
- [x] `backend/services/annotate.py` (OpenCV visualization banner & bounding boxes)
- [x] `backend/services/pipeline.py` (latest-frame-wins inference pipeline)
- [x] `backend/services/notifiers/` (base, dashboard, telegram stub, esp32 stub)
- [x] `backend/services/alert_manager.py` (cooldown, event management, sound trigger)
- [x] `backend/services/camera_worker.py` (OpenCV RTSP/MJPEG/webcam background worker)
- [x] `backend/api/` (routes for health, models, detect image/video, camera, data, settings, ws)
- [x] `backend/main.py` (FastAPI app, CORS, static mounting)
- [x] `backend/tests/test_decision_engine.py` (9/9 unit tests passing 100%)

### Track C: Frontend (React 18 + Vite)
- [x] Setup React 18 + Vite JSX project in `frontend/`
- [x] Hand-written dark control-room theme (`frontend/src/styles/theme.css`)
- [x] Web Audio API siren oscillator (`frontend/src/utils/audio.js`)
- [x] Global WebSocket hook (`frontend/src/hooks/useEvents.js`)
- [x] API Client (`frontend/src/api/client.js`)
- [x] Sidebar, Header with status chips, and Layout
- [x] Route 1: Overview (`/`)
- [x] Route 2: Live Camera (`/live`) with annotated video stream & QR code
- [x] Route 3: Image Detection (`/image`)
- [x] Route 4: Video Detection (`/video`)
- [x] Route 5: CNN Classification (`/cnn`)
- [x] Route 6: YOLO Detection (`/yolo`)
- [x] Route 7: Detection History (`/history`)
- [x] Route 8: Analytics & Performance (`/analytics`)
- [x] Route 9: Model Information (`/models`)
- [x] Route 10: Settings (`/settings`)
- [x] Mobile Phone Camera page (`/camera`) with `getUserMedia` & backpressure WS stream

### Track D: DevOps, Testing & Documentation
- [x] PowerShell scripts (`scripts/setup_windows.ps1`, `scripts/run_dev.ps1`, `scripts/run_demo.ps1`)
- [x] Self-signed certificate generator (`scripts/gen_selfsigned_cert.py`)
- [x] Automated smoke test script (`scripts/smoke_test.py`)
- [x] Documentation (`docs/architecture.md`, `docs/methodology.md`, `docs/testing.md`, `docs/dataset.md`, `docs/demo_guide.md`, `docs/academic_report_outline.md`, `docs/esp32_buzzer.md`)
