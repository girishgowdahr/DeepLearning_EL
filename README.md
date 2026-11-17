# 🌲 ForestGuard AI

**Forest Fire Detection Using Convolutional Neural Networks (CNN) and You Only Look Once (YOLO)**

An academic deep-learning and edge computer-vision prototype that turns an Android smartphone into a remote wildfire camera, detects **fire and smoke** simultaneously using **YOLO** (bounding-box spatial localization) and **MobileNetV2 CNN** (full-frame binary classification), fuses both with a transparent rule-based decision engine, and displays live telemetry on an interactive web dashboard with audio sirens and persistent event logging.

> ⚠️ **Academic Prototype Notice**: This system is designed for engineering research and academic demonstration. It is not certified for life-safety critical operations.

---

## 1. System Pipeline

```
Android Phone Camera ──► HTTPS Stream / WebSocket (/ws/stream) ──► FastAPI Backend (CPU)
       ──► YOLO Nano (fire/smoke boxes) + MobileNetV2 CNN (fire/no-fire prob)
       ──► Decision Fusion Engine (Temporal confirmation & cooldown)
       ──► React 18 Control-Room Dashboard ──► Web Audio API Siren Alarm
                                            └── Future: Telegram Bot & ESP32 Buzzer stubs
```

| Component | Architecture & Technology | Role in System |
|---|---|---|
| **Object Detection** | Ultralytics YOLO Nano (`yolo11n.pt` / `best.pt`, 416×416) | Spatial bounding-box localization of fire fronts & smoke plumes |
| **Classification** | MobileNetV2 (ImageNet Transfer Learning, 224×224, ONNX) | Full-frame verification of fire visual signatures |
| **Decision Engine** | Rule-based matrix + $M=6, N=3$ sliding window | Prevents false alarms; eliminates model disagreement |
| **Edge Capture** | HTML5 `getUserMedia` → Canvas → WebSocket backpressure | Mobile Android Chrome stream over HTTPS |
| **Backend & Storage** | Python 3.11, FastAPI, Uvicorn, SQLite (`forestguard.db`), OpenCV | Single-port API & static frontend hosting on port 8000 |
| **Frontend** | React 18, Vite, Recharts, Lucide Icons, pure hand-written CSS | Mission control dashboard with dark control-room theme |
| **Auditory Alert** | HTML5 Web Audio API synthesizer (880/660 Hz alternating oscillator) | Client-side alarm siren without audio file dependencies |

---

## 2. Why Both CNN and YOLO?

- **CNN** answers: *"Does this panoramic camera view contain fire characteristics?"* (Binary classification probability).
- **YOLO** answers: *"Where specifically are the fire fronts and smoke columns located?"* (Bounding box coordinates and object confidences).

Because classification probabilities and detection confidences represent different probabilistic distributions, **they are never averaged together**. The transparent Decision Engine evaluates them mutually:

| Condition | Decision Status | Alert Action |
|---|---|---|
| YOLO Fire $\ge 0.40$ **AND** CNN Fire $\ge 0.60$ ($\ge 3/6$ frames) | 🔴 **FIRE CONFIRMED** | Emergency Dashboard Alert + Browser Audio Siren |
| YOLO Smoke $\ge 0.45$ ($\ge 3/6$ frames) | 🟠 **SMOKE CONFIRMED** | Warning Alert + Double-Beep Tone |
| CNN Fire $\ge 0.90$ with no YOLO box | 🟡 **POSSIBLE FIRE** | Warning Alert (No siren; unlocalized) |
| Models Disagree (e.g. YOLO 0.70 vs CNN 0.15) | 🟡 **UNCERTAIN** | Logged as Uncertain; Siren Suppressed |
| No detections | 🟢 **NORMAL (NO FIRE)** | Routine Monitoring |

---

## 3. Quick Start (Windows PowerShell)

### Prerequisites
- Windows 10/11 (8 GB RAM, standard CPU, no GPU required)
- Python 3.11 or 3.12
- Node.js 18+ and npm
- Git

### One-Click Setup & Launch
```powershell
# 1. Clone repository
git clone <your-repo-url> ForestGuard-AI
cd ForestGuard-AI

# 2. Run automated setup (creates .venv, installs dependencies, builds frontend)
.\scripts\setup_windows.ps1

# 3. Start demo server
.\scripts\run_demo.ps1
```

Once running:
1. Open **`http://localhost:8000`** in your laptop browser.
2. Click **Enable Sound** in the top navigation bar to unlock the audio siren.
3. Open the **Live Camera** page (`/live`), scan the QR code with your Android phone, and tap **Start Camera** to stream live video!

---

## 4. Cloud Training Instructions

Models are fine-tuned on free cloud GPUs (Kaggle T4 / Ultralytics Platform) to prevent laptop overheating:
- **MobileNetV2 CNN**: Follow copy-paste steps in [`training/cnn/README.md`](file:///training/cnn/README.md).
- **YOLO Nano**: Follow copy-paste steps in [`training/yolo/README.md`](file:///training/yolo/README.md).

Trained weights are placed into `model_weights/`:
- `cnn_mobilenetv2.onnx`
- `yolo_best.pt`
- `MODEL_INFO.json`

*(The system starts gracefully even if models are absent, displaying clear "Model Unavailable" status badges).*

---

## 5. Verification & Testing

ForestGuard AI includes automated test suites and a 12-scenario operational test matrix:
```powershell
# Run Decision Engine unit tests
.venv\Scripts\python.exe -m pytest backend\tests

# Run end-to-end API smoke tests
.venv\Scripts\python.exe scripts\smoke_test.py
```
Detailed test results and latency figures are recorded in [`docs/testing.md`](file:///docs/testing.md).

---

## 6. Academic Deliverables & Documentation

- [System Architecture](file:///docs/architecture.md)
- [Deep Learning Methodology](file:///docs/methodology.md)
- [D-Fire Dataset & Zero-Leakage Protocol](file:///docs/dataset.md)
- [Testing & Matrix Validation Report](file:///docs/testing.md)
- [Academic Demo Presentation Guide](file:///docs/demo_guide.md)
- [Academic Report Outline](file:///docs/academic_report_outline.md)
- [ESP32 IoT Buzzer Guide](file:///docs/esp32_buzzer.md)
- [Live Progress & Milestone Checklist](file:///docs/progress.md)

---

## 7. License & Credits
- Built with PyTorch, Ultralytics YOLO, ONNX Runtime, FastAPI, React, and Recharts.
- Trained on the D-Fire Dataset (CC0 / CC BY 4.0).
