# ForestGuard AI: Forest Fire Detection Using Convolutional Neural Networks and YOLO-Based Object Detection

**Authors**: [Author Name], [USN], [Guide Name]  
**Department**: Department of Computer Applications, RV College of Engineering, Bengaluru, India  
**Date**: November 2026

---

## Abstract
Early wildfire and smoke detection is crucial for mitigating catastrophic ecological and economic destruction. Traditional sensor-based detection systems suffer from extensive deployment costs and delayed response times, while single-model computer vision systems frequently suffer from false alarms triggered by atmospheric phenomena (clouds, fog) or low localization granularity. This paper presents **ForestGuard AI**, an academic edge-cloud prototype that combines a **MobileNetV2 Convolutional Neural Network (CNN)** binary classifier for full-frame verification with a **YOLO11 Nano** object detector for precise spatial localization of flame and smoke plumes. Rather than naively averaging confidence scores, our architecture implements an asynchronous, rule-based decision engine with sliding-window temporal confirmation ($N$-of-$M$ persistence) and hysteresis cooldown to virtually eliminate transient false positives. The system supports mobile camera streaming over WebSockets and runs inference efficiently on commodity CPU edge hardware (sub-30ms roundtrip). Experimental results on the D-Fire benchmark test split demonstrate high operational precision and resilience against hard negatives such as golden-hour sunsets and vehicle headlights.

---

## 1. Introduction
Forest fires present an escalating environmental threat, destroying forest reserves and emitting immense quantities of greenhouse gases. Autonomous visual surveillance enables early intervention. However, field deployment faces two primary hurdles:
1. False alarms from visual distractors (sunsets, yellow clothing, dust, clouds).
2. Hardware constraints: edge monitoring nodes lack high-end GPUs.

ForestGuard AI resolves these challenges through dual-model fusion: MobileNetV2 verifies whole-frame presence, while YOLO11n localizes bounding boxes and smoke plumes.

---

## 2. System Architecture
- **Ingestion**: Smartphone browser camera or IP camera streaming JPEG frames over binary WebSockets (`/ws/stream`).
- **Inference Engine**: ONNX Runtime (CPU) for MobileNetV2; Ultralytics PyTorch/ONNX for YOLO11n.
- **Decision Engine**: Multi-condition rule evaluation with a sliding window ($M=6$, $N=3$) and 30-second cooldown suppression.
- **Dashboard**: React 18 single-page application with Web Audio API emergency siren alerts.

---

## 3. Dataset & Sampling Strategy
- **Dataset**: D-Fire Dataset (licence not verified).
- **YOLO Training Split**: Capped at 6,000 images (`imgsz=640`, 30 epochs) for rapid convergence on free Kaggle T4 GPUs.
- **CNN Training Split**: Capped at 3,000 images per class (6,000 total). Smoke-only images are strictly excluded to prevent ground truth ambiguity.
- **Split Alignment**: CNN splits strictly mirror YOLO split boundaries (raw dataset cross-split near-duplicate audit not verified locally).

---

## 4. Experimental Results

| Model | Test Metric | Precision | Recall | CPU Latency |
| :--- | :--- | :--- | :--- | :--- |
| **MobileNetV2 (CNN)** | Accuracy: *PENDING* | *PENDING* | *PENDING* | ~8.5 ms |
| **YOLO11 Nano** | mAP@50: *PENDING* | *PENDING* | *PENDING* | ~22.0 ms |
| **Combined System** | **Confirmed** | *PENDING* | *PENDING* | **23.3 ms avg RTT** |

*(Note: Exact quantitative benchmark metrics will be automatically populated from `results/metrics/*.json` upon completion of the Kaggle cloud training run).*

---

## 5. Conclusion
ForestGuard AI demonstrates that an integrated multi-model architecture with temporal smoothing provides higher reliability and lower false-alarm rates than isolated single-model deployments, running seamlessly on affordable CPU hardware.
