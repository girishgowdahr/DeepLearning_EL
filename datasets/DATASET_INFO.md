# ForestGuard AI — Dataset Information & Provenance

## 1. Primary Dataset: D-Fire (Wildfire & Smoke Detection)
- **Official Repository**: [gaia-solutions-on-demand/DFireDataset](https://github.com/gaiasd/DFireDataset)
- **Kaggle Mirror**: `gaiasd/dfire-dataset` / `smoke-fire-detection-yolo`
- **License**: CC0 1.0 Universal / Creative Commons Attribution (CC BY 4.0)
- **Task Types**: Object Detection (YOLO format) & Image Classification (Derived)
- **Total Raw Images**: ~21,000+ frames (surveillance, aerial, time-lapse, nighttime, wilderness)
- **Classes**:
  - `0`: `fire` (flames, bonfires, forest fires)
  - `1`: `smoke` (plumes, wildfire smoke columns)
  - Negatives: background scenes without fire and without smoke (forest canopy, mountains, clouds, sky)

---

## 2. Split Derivation & Data Leakage Prevention Strategy
To guarantee **zero data leakage** between the object detection (YOLO) and image classification (MobileNetV2 CNN) tasks, both models strictly use the **identical underlying split boundaries**:

```
D-Fire Raw Split (Train / Val / Test)
  │
  ├─► YOLO Split (Direct usage)
  │     ├── train/ (images + labels with fire and smoke boxes)
  │     ├── val/
  │     └── test/
  │
  └─► CNN Split (Derived directly from corresponding split)
        ├── train/
        │     ├── fire/     (images with >= 1 fire box)
        │     └── no_fire/  (images with NO fire and NO smoke boxes)
        ├── val/
        │     ├── fire/
        │     └── no_fire/
        └── test/
              ├── fire/
              └── no_fire/
```

### Critical Derivation Rules:
1. **Smoke-Only Image Exclusion for CNN**: Images containing *only smoke* and no fire are explicitly excluded from CNN training and evaluation. A smoke-only scene is neither pure Fire nor pure No-Fire; classifying it as either would introduce label noise.
2. **Duplicate Detection**: Perceptual hashing (dHash) is run across splits. Any duplicate or near-identical frames found across train and test are purged from the training set.
3. **Class Balancing**: Undersampling is applied to the majority class to balance `fire` and `no_fire` ratios (~1,500 to 3,000 images per class for fast, balanced convergence).

---

## 3. Hard Negatives Evaluation Set
To honestly benchmark real-world false-positive rates, a specialized validation check set is included:
- **Sunsets & Sunrises** (golden hour light resembling flames)
- **Red & Orange Objects** (vehicles, clothing, reflective gear)
- **Low-hanging Fog & Cumulus Clouds** (distinguished from smoke plumes)
- **Nighttime Vehicle Headlights & Streetlamps**

All performance reports distinguish between standard test accuracy and hard-negative resilience.
