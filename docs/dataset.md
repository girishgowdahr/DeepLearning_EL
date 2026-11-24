# D-Fire Dataset Specification & Split Partitioning Protocol

## 1. Overview
- **Dataset**: D-Fire (Dilation & Fire/Smoke Benchmark)
- **Source**: [https://github.com/gaiasd/DFireDataset](https://github.com/gaiasd/DFireDataset)
- **License**: NOT VERIFIED. The source repository page states no licence; do not assume CC0 / CC BY. Confirm with the original authors before redistribution.
- **Total Images**: 21,000+ surveillance, forest lookout, aerial, and ground camera frames.
- **Classes (YOLO Model)**:
  - `0`: `smoke`
  - `1`: `fire`
  - Negatives: Forest canopies, skies, mountains, and plains containing neither fire nor smoke.
- **Classes (CNN Classifier)**:
  - `0`: `fire`
  - `1`: `no_fire` (smoke-only frames excluded)

## 2. Split Partitioning & Anti-Leakage Protocol
To ensure valid scientific benchmarking across both models:
1. **Identical Partitions**: The derived MobileNetV2 classification dataset strictly adheres to the exact train/val/test splits of the YOLO dataset.
2. **Smoke-Only Frame Isolation**: Images containing only smoke (without flames) are excluded from the binary fire classifier to prevent ambiguous ground-truth labels.
3. **Partition Integrity**: Training splits mirror each other directly. Note that because raw dataset images reside in cloud storage, complete absence of consecutive time-lapse near-duplicate frames across splits has not been verified on local raw imagery.
