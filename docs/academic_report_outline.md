# Academic Project Report Outline
## Forest Fire Detection Using Convolutional Neural Networks (CNN) and You Only Look Once (YOLO)

---

### 1. Title Page & Abstract
- **Title**: Forest Fire Detection Using Convolutional Neural Networks (CNN) and You Only Look Once (YOLO)
- **Course**: Deep Learning Experiential Learning (Sem 3)
- **Abstract**:
  Early detection of forest fires is critical to mitigating catastrophic environmental and economic loss. While traditional lookout towers and satellite observation suffer from latency, cloud obstruction, and coarse resolution, edge computer vision offers low-latency localized threat detection. This project presents **ForestGuard AI**, a dual-model edge vision system combining a fine-tuned lightweight convolutional neural network (**MobileNetV2**) for binary scene classification and an optimized single-stage object detector (**YOLO nano**) for bounding-box localization of fire fronts and smoke columns. Operating over real-time video streamed from edge mobile cameras, a rule-based decision fusion engine synthesizes predictions across a sliding temporal confirmation window ($M=6$ frames) with cooldown suppression to eliminate transient false alarms. The entire inference pipeline operates on consumer CPU hardware (8 GB RAM) achieving 3.5–5.0 FPS with full-stack observability via a React control-room dashboard.

---

### 2. Introduction & Problem Statement
- **2.1 Background**: Wildfires consume millions of hectares annually, releasing megatons of carbon emissions and destroying ecosystems.
- **2.2 Problem Statement**: Existing computer vision methods either rely on single-task classifiers (which detect smoke/fire presence but cannot localize coordinates for firefighting teams) or lone object detectors (which suffer high false-positive rates when encountering sunsets, dust, or vehicular reflections). Furthermore, most DL models require expensive cloud GPUs, making rapid field deployment challenging.
- **2.3 Objectives**:
  1. Fine-tune a lightweight CNN (MobileNetV2) for fast full-frame fire classification.
  2. Fine-tune a nano object detector (YOLO) for spatial bounding-box localization of fire and smoke.
  3. Guarantee zero data-leakage by deriving both training sets from identical train/val/test split boundaries.
  4. Develop a transparent decision fusion engine with temporal confirmation and alarm fatigue suppression.
  5. Deploy a real-time full-stack prototype accessible from mobile phones and web browsers without GPU dependencies.

---

### 3. Existing System vs. Proposed System
- **3.1 Existing Systems**:
  - Satellite remote sensing (MODIS, VIIRS): Low revisit frequency (hours to days), high cloud occlusion.
  - Thermal sensors: Limited spatial range, high hardware deployment costs.
  - Single-model vision systems: High false-alarm rates during golden hour lighting.
- **3.2 Proposed ForestGuard AI System**:
  - Dual-model mutual verification: CNN validates YOLO proposals; YOLO localizes CNN alerts.
  - Zero cloud GPU dependency during inference: Runs efficiently on standard CPU hardware.
  - Live edge camera streaming over secure WebSockets (HTTPS/WSS).
  - Immediate audio-visual siren alerting with event cooldown.

---

### 4. Literature Review Outline
*(Topics & Academic Query Terms for Literature Review)*:
- *Wildfire smoke detection in UAV imagery using deep convolutional neural networks.*
- *Comparison of single-stage (YOLOv8/11) vs two-stage (Faster R-CNN) detectors for disaster response.*
- *Lightweight inverted residual architectures (MobileNetV2) for resource-constrained edge inference.*
- *Temporal voting and sliding window filtering in real-time video anomaly detection.*
- *Multimodal sensor fusion strategies in forest surveillance.*

---

### 5. System Architecture & Methodology
- **5.1 Ingestion Tier**: HTML5 `getUserMedia`, canvas frame grabber, backpressure over `/ws/stream`, OpenCV RTSP/MJPEG worker.
- **5.2 Inference Tier**:
  - **MobileNetV2 (CNN)**: 224×224 input, ImageNet weights, two-stage transfer learning, exported to ONNX Runtime (CPU).
  - **YOLO Nano**: 416×416 input, classes `0: fire`, `1: smoke`, Ultralytics CPU runtime.
- **5.3 Decision Fusion Tier**:
  - Formal evaluation rules: FIRE, SMOKE, POSSIBLE_FIRE, UNCERTAIN, NO_FIRE.
  - Sliding temporal window ($M=6, N=3$).
  - Event lifecycle: 10s clear time, 30s cooldown.
- **5.4 Storage & Notification Tier**:
  - SQLite persistent audit logging (`forestguard.db`).
  - Web Audio API synthesizer (880/660 Hz alternating oscillator).
  - Modular stubs for Telegram Bot API and ESP32 IoT hardware buzzer.

---

### 6. Dataset & Training Procedure
- **6.1 Source Dataset**: D-Fire Wildfire & Smoke Dataset (~21,000 images, licence not verified).
- **6.2 Split Integrity**: Strictly preserving train/val/test splits across both tasks.
- **6.3 Preprocessing & Deduplication**: Difference hashing (dHash) cross-split purge, class balancing (1,500–3,000 per class), hue-preserving data augmentation.
- **6.4 Training Platform**: Free cloud GPU (Kaggle NVIDIA T4 x2 / Ultralytics Platform).

---

### 7. Experimental Results & Performance Analysis
*(To be populated with exact metrics from `results/metrics/*.json`)*:
- **CNN Metrics**:
  - Test Accuracy: *TO BE FILLED AFTER TRAINING*
  - Fire Precision / Recall / F1: *TO BE FILLED AFTER TRAINING*
  - Macro F1-Score: *TO BE FILLED AFTER TRAINING*
- **YOLO Metrics**:
  - mAP@50: *TO BE FILLED AFTER TRAINING*
  - mAP@50-95: *TO BE FILLED AFTER TRAINING*
  - Per-class AP (Fire / Smoke): *TO BE FILLED AFTER TRAINING*
- **System Telemetry**:
  - Average CPU latency: ~195 ms per frame.
  - Effective streaming frame rate: 3.5–5.0 FPS on 8 GB RAM CPU laptop.
  - RAM footprint: ~1.25 GB.

---

### 8. Limitations & Future Enhancements
- **Limitations**:
  - Sunset and low-altitude golden hour glare mimic fire hues.
  - Dispersed low-altitude fog can resemble thin smoke.
  - Nighttime detection relies on visible flame illumination.
- **Future Work**:
  - Multi-camera PTZ synchronization and triangulation.
  - Thermal infrared (FLIR) optical camera fusion.
  - Active Telegram Bot dispatch with compressed snapshots.
  - ESP32 LoRa wireless siren deployment across remote watchtowers.
  - GPS drone telemetry integration.

---

### 9. References (Verified Sources)
1. D-Fire Dataset Official Repository: https://github.com/gaiasd/DFireDataset
2. Ultralytics YOLO Architecture & Documentation: https://docs.ultralytics.com
3. Sandler, M., et al. "MobileNetV2: Inverted Residuals and Linear Bottlenecks." CVPR 2018. https://arxiv.org/abs/1801.04381
4. ONNX Runtime CPU Optimization Engine: https://onnxruntime.ai
5. FastAPI High-Performance Framework: https://fastapi.tiangolo.com
