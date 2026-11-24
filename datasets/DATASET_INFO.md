# ForestGuard AI — Dataset Information & Provenance

## 1. Primary Dataset: D-Fire (Wildfire & Smoke Detection)
- **Official Repository**: [gaia-solutions-on-demand/DFireDataset](https://github.com/gaiasd/DFireDataset)
- **Kaggle Mirror**: `sayedgamal99/smoke-fire-detection-yolo` / `thunguyen1012/smoke-fire-detection-yolo` / `gaiasd/dfire-dataset`
- **License**: NOT VERIFIED. The source repository page states no licence; do not assume CC0 / CC BY. Confirm with the original authors before redistribution.
- **Task Types**: Object Detection (YOLO format) & Image Classification (Derived)
- **Total Raw Images**: ~21,000+ frames (surveillance, aerial, time-lapse, nighttime, wilderness)
- **Classes (YOLO Model)**:
  - `0`: `smoke` (plumes, wildfire smoke columns)
  - `1`: `fire` (flames, bonfires, forest fires)
  - Negatives: background scenes without fire and without smoke (forest canopy, mountains, clouds, sky)
- **Classes (CNN Classifier)**:
  - `0`: `fire`
  - `1`: `no_fire` (smoke-only frames excluded)

---

## 2. Training Time-Boxing & Sampling Caps
To guarantee that training finishes reliably tonight on Kaggle's free GPU (NVIDIA T4) within strict limits and without exhausting free quotas:

| Model | Raw Available | Capped Training Size | Target Epochs | Batch Size | Img Size | Est. Runtime (T4) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **MobileNetV2 (CNN)** | ~21,000 | **3,000 per class** (6,000 total) | 5 warmup + 8 fine-tune | 32 | 224 | **~15–20 mins** |
| **YOLO11 Nano** | ~21,000 | **6,000 images** (capped) | 30 (patience=10) | 16 | 640 | **~45–60 mins** |

Both models include an initial **1-epoch smoke test (200 images)** before full training to verify GPU memory, CUDA tensors, loss calculation, and ONNX export within 30 seconds.

---

## 3. Split Derivation & Data Leakage Prevention Strategy
To guarantee **zero data leakage** between the object detection (YOLO) and image classification (MobileNetV2 CNN) tasks, both models strictly use the **identical underlying split boundaries**:

```
D-Fire Raw Split (Train / Val / Test)
  │
  ├─► YOLO Split (Direct usage)
  │     ├── train/ (images + labels with fire and smoke boxes, capped at 6,000)
  │     ├── val/   (capped at 1,000)
  │     └── test/  (capped at 1,000)
  │
  └─► CNN Split (Derived directly from corresponding split)
        ├── train/
        │     ├── fire/     (images with >= 1 fire box, capped at 3,000)
        │     └── no_fire/  (images with NO fire and NO smoke boxes, capped at 3,000)
        ├── val/
        │     ├── fire/
        │     └── no_fire/
        └── test/
              ├── fire/
              └── no_fire/
```

### Critical Derivation Rules:
1. **Smoke-Only Image Exclusion for CNN**: Images containing *only smoke* and no fire are explicitly excluded from CNN training and evaluation. A smoke-only scene is neither pure Fire nor pure No-Fire; classifying it as either would introduce label noise.
2. **Class Balancing**: Undersampling is applied to the majority class to balance `fire` and `no_fire` ratios (3,000 images per class for fast, balanced convergence).
3. **Hard Negatives Benchmark**: Background forest images, sunsets/sunrises, and low-altitude clouds are included in the No-Fire evaluation set.
