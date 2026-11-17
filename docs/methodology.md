# ForestGuard AI — Deep Learning Methodology

## 1. Dual-Model Architecture Rationale

Forest fire detection in real-world wilderness settings presents two fundamentally distinct challenges:
1. **Global Scene Classification (CNN)**: Does this panoramic landscape contain wildfire characteristics (spectral signature, ambient glow, foliage texture disruption)?
2. **Spatial Object Localization (YOLO)**: Where specifically are the flame fronts and smoke columns located within the frame?

Existing systems often deploy either a lone classifier (which fails to locate small or distant fires) or a lone object detector (which is prone to bounding-box hallucination when exposed to sunsets, reflections, and bright vehicle headlights). ForestGuard AI deliberately couples both architectures to achieve mutual verification.

---

## 2. MobileNetV2 CNN Classification

### 2.1 Model Architecture
- **Backbone**: MobileNetV2 pretrained on ImageNet-1K (`IMAGENET1K_V1`). Uses inverted residual blocks with depthwise separable convolutions to minimize parameter count and memory bandwidth.
- **Classification Head**: Global average pooling → Dropout(0.2) → Linear(1280, 2).
- **Target Classes**: `fire`, `no_fire`.
- **Exclusion of Smoke-Only Frames**: Frames containing smoke plumes but no visible flames are strictly excluded from the CNN dataset. Smoke is semi-transparent and morphologically distinct from fire; labeling smoke-only frames as "Fire" introduces label noise, while labeling them as "No Fire" confuses flame detection.

### 2.2 Training Procedure
- **Stage 1 (Head Warmup)**: Backbone frozen, classifier trained for 5 epochs with AdamW (learning rate $10^{-3}$, weight decay $10^{-2}$).
- **Stage 2 (Fine-Tuning)**: Last 4 inverted residual blocks unfrozen, trained for 8 epochs with Cosine Annealing learning rate schedule ($10^{-4}$ down to $10^{-6}$).
- **Data Augmentation**:
  - `RandomResizedCrop(224, scale=(0.7, 1.0))`
  - `RandomHorizontalFlip()`
  - `RandomRotation(degrees=15)`
  - `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.03)` (Hue jitter is constrained to $\le 0.03$ to avoid shifting natural flame hues away from orange/red).

### 2.3 Optimization & Export
- Model is exported to **ONNX (Opset 17)** with dynamic batching.
- In-memory inference is executed via ONNX Runtime `CPUExecutionProvider` configured for 4 intra-op threads.

---

## 3. Ultralytics YOLO Nano Detection

### 3.1 Model Architecture
- **Model**: `yolo11n.pt` (Ultralytics YOLO nano, ~2.6M parameters).
- **Detection Classes**:
  - `0`: `fire`
  - `1`: `smoke`
- **Input Resolution**: 640×640 during cloud training; letterboxed to 416×416 during CPU inference.

### 3.2 Training Procedure
- **Optimizer**: SGD / AdamW with automatic weight decay.
- **Hyperparameters**:
  - Epochs: 30–50
  - Batch size: 16
  - Patience: 15 epochs early stopping on mAP50
  - Mosaic augmentation enabled during early epochs, disabled for the final 5 epochs.

---

## 4. Zero Data-Leakage Split Derivation

A major source of bias in academic computer vision is data leakage across tasks. ForestGuard AI prevents this by deriving all datasets from the exact same primary D-Fire splits:

$$\text{D-Fire Split} \longrightarrow \begin{cases} \text{YOLO Training Split} & (\text{Full Bounding Boxes: Fire, Smoke}) \\ \text{CNN Training Split} & (\text{Derived: Fire vs Clean No-Fire}) \end{cases}$$

Perceptual difference hashing (dHash) is run across all image pairs to identify and eliminate near-duplicate frames resulting from camera time-lapses.

---

## 5. Transparent Decision Fusion Engine

Because classification probabilities and object detection confidences represent different probabilistic distributions, they are **never averaged**. Instead, a rule-based decision matrix combines evidence per frame:

```
IF yolo_fire >= 0.40 AND cnn_fire >= 0.60:
    candidate = FIRE (Models agree)
ELIF yolo_fire >= 0.55 AND cnn_fire < 0.60:
    candidate = UNCERTAIN (Models disagree)
ELIF yolo_fire < 0.40 AND cnn_fire >= 0.90:
    candidate = POSSIBLE_FIRE (Strong CNN visual evidence, unlocalized)
ELIF 0.25 <= yolo_fire < 0.40 AND cnn_fire >= 0.60:
    candidate = POSSIBLE_FIRE (Weak YOLO proposal confirmed by CNN)
ELIF yolo_smoke >= 0.45:
    candidate = SMOKE
ELSE:
    candidate = NO_FIRE
```

### Temporal Confirmation
A sliding window of $M = 6$ frames filters momentary sensor noise:
- `FIRE_CONFIRMED` requires $\ge 3$ candidate `FIRE` frames.
- `SMOKE_CONFIRMED` requires $\ge 3$ candidate `SMOKE` frames.
- `POSSIBLE_FIRE` requires $\ge 4$ frames.

### Event Lifecycle & Alert Suppression
To prevent alarm fatigue, an active threat opens an event. Exactly one audio siren and one dashboard alert are triggered. The event remains open until $10$ seconds of continuous `NO_FIRE` frames elapse, followed by a $30$-second cooldown timer before subsequent alerts can trigger.
