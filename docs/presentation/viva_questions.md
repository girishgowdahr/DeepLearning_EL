# ForestGuard AI — Deep Learning Viva Q&A Guide (30+ Questions)

Comprehensive technical answers tailored specifically to the ForestGuard AI architecture and implementation.

---

### Section 1: Core Architecture & Decision Fusion

#### Q1: Why use both a CNN and a YOLO detector instead of just YOLO?
**Answer**: YOLO is an object detector optimized for localization (drawing bounding boxes), but it can produce false-positive boxes on ambiguous textures (like clouds, dust, or bright vehicle reflections). MobileNetV2 is an image classifier that evaluates the global receptive field of the full frame to answer: *"Is there fire context anywhere in this scene?"* By requiring agreement between classification and detection, false alarms drop dramatically.

#### Q2: Why don't you simply average the confidence scores of CNN and YOLO?
**Answer**: Averaging confidences mathematically masks critical disagreement. If YOLO is 85% confident of a small yellow box (e.g. reflective vest) but CNN is 95% confident the scene is pure forest canopy (No-Fire), an average would produce 0.90, falsely triggering a siren! In our rule-based engine, this scenario triggers `UNCERTAIN`, preventing false alarms while flagging for review.

#### Q3: What is Temporal Confirmation ($N$-of-$M$), and why is it used?
**Answer**: Real-world video feeds experience camera sensor noise, compression artifacts, and passing reflections. A single frame spike must not wake up emergency services. Our decision engine maintains a sliding window of $M=6$ frames and requires $N=3$ candidate fire detections before escalating to `FIRE_CONFIRMED`.

#### Q4: What happens if one of the models fails or its weight file is missing?
**Answer**: ForestGuard AI implements graceful degradation:
- If CNN is unavailable: YOLO functions in standalone mode (flags `(CNN unavailable)` in log).
- If YOLO is unavailable: CNN functions in unlocalized warning mode (`POSSIBLE_FIRE`).
- If both are missing: System defaults to `NO_FIRE` with an administrative alert, keeping the camera stream alive.

---

### Section 2: MobileNetV2 & Transfer Learning

#### Q5: Why did you choose MobileNetV2 over ResNet-50 or VGG-16?
**Answer**: MobileNetV2 uses depthwise separable convolutions and inverted residual bottlenecks, reducing parameters from ~25M (ResNet-50) to ~3.5M. This enables sub-10ms inference on commodity laptop CPUs without needing an external GPU.

#### Q6: Explain your two-stage transfer learning procedure for CNN.
**Answer**:
1. **Stage 1 (Warmup, 5 epochs)**: The pre-trained ImageNet backbone is frozen (`requires_grad=False`). Only the newly initialized classification head is trained with Adam (`lr=1e-3`) to prevent large random gradients from destroying pre-trained feature weights.
2. **Stage 2 (Fine-tuning, 8 epochs)**: The top 4 inverted residual blocks are unfrozen and trained at a lower learning rate (`lr=1e-4`) using `AdamW` and cosine annealing to adapt high-level visual features to fire and smoke textures.

#### Q7: Why were smoke-only images excluded from CNN training?
**Answer**: A scene containing only smoke without visible flames is neither pure "Fire" nor pure "No-Fire". Labeling it as Fire creates confusion when flames are absent; labeling it as No-Fire penalizes the model when smoke is visible. We strictly exclude smoke-only frames from CNN binary classification, letting YOLO handle smoke localization.

---

### Section 3: YOLO11 Nano & Object Detection

#### Q8: What loss functions does YOLO use during training?
**Answer**: YOLO11 uses three primary loss components:
1. **Bounding Box Regression Loss**: CIoU (Complete Intersection over Union), which measures overlap, distance between box centers, and aspect ratio consistency.
2. **Distribution Focal Loss (DFL)**: Models bounding box coordinates as continuous probability distributions.
3. **Classification Loss**: Binary Cross-Entropy (BCE) with focal loss to handle class imbalance.

#### Q9: What is Non-Maximum Suppression (NMS) and IoU?
**Answer**: 
- **IoU (Intersection over Union)**: Ratio of the area of overlap between predicted and ground truth boxes to their union area.
- **NMS**: A post-processing algorithm that eliminates redundant overlapping bounding boxes for the same object by keeping the highest-confidence box and discarding boxes with IoU $> 0.45$.

#### Q10: Why did you train on Kaggle Dual-T4 instead of locally?
**Answer**: Our laptop has an Intel CPU with 8 GB RAM and no dedicated NVIDIA GPU. Training YOLO and CNN locally would take 15+ hours and risk out-of-memory crashes. Kaggle provides 2 × NVIDIA Tesla T4 GPUs (16 GB VRAM each), completing training in under 60 minutes with mixed-precision acceleration (`amp=True`).

---

### Section 4: Metrics, Evaluation & Optimization

#### Q11: What is mAP@50 and mAP@50-95?
**Answer**:
- **mAP@50**: Mean Average Precision calculated at an IoU threshold of 0.50.
- **mAP@50-95**: The official COCO metric, which averages mAP across IoU thresholds from 0.50 to 0.95 in steps of 0.05. It tests localization precision at strict overlap boundaries.

#### Q12: Why is Recall critical in wildfire detection?
**Answer**: In life-safety applications, a False Negative (missing a real wildfire) is catastrophic, whereas a False Positive (investigating a false alarm) is merely inconvenient. High recall ensures every potential ignition is detected.

#### Q13: How does ONNX Runtime improve inference speed?
**Answer**: ONNX Runtime optimizes computation graphs through constant folding, kernel fusion (e.g. combining Conv+BatchNorm+ReLU into a single CPU operation), and multi-threaded parallel execution (`CPUExecutionProvider`), achieving 2–3× speedup over raw PyTorch CPU.

---

### Section 5: Real-World Deployment & Edge Systems

#### Q14: How does the phone camera stream to the server?
**Answer**: Using the HTML5 `navigator.mediaDevices.getUserMedia` API, the phone accesses its rear camera, captures video frames into an HTML Canvas at 640×480, compresses them to JPEG, and sends them via WebSocket (`/ws/stream`) to the FastAPI backend.

#### Q15: Why is HTTPS required for phone camera streaming?
**Answer**: Modern mobile web browsers (Chrome, Safari) strictly block camera and microphone access on insecure HTTP connections. Only `localhost` and origins secured with valid or self-signed HTTPS certificates can access `getUserMedia`.

#### Q16: What is the purpose of the 30-second cooldown in the decision engine?
**Answer**: Once an alert is confirmed, repeating sirens on every subsequent video frame overwhelms operators. The cooldown suppresses duplicate sirens during an active event while keeping detection logs updated.

---

*(Questions 17–32 covering augmentation, class imbalance, confusion matrix analysis, and edge deployment are documented in full across the viva reference manual).*
