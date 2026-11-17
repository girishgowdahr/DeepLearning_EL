# D-Fire Dataset Specification & Zero-Leakage Splitting

## 1. Overview
- **Dataset**: D-Fire (Dilation & Fire/Smoke Benchmark)
- **Source**: [https://github.com/gaiasd/DFireDataset](https://github.com/gaiasd/DFireDataset)
- **License**: CC0 1.0 Universal (Public Domain Dedication) / CC BY 4.0
- **Total Images**: 21,000+ surveillance, forest lookout, aerial, and ground camera frames.
- **Classes**:
  - `0`: `fire`
  - `1`: `smoke`
  - Negatives: Forest canopies, skies, mountains, and plains containing neither fire nor smoke.

## 2. Zero Data-Leakage Split Protocol
To ensure valid scientific benchmarking across both models:
1. **Identical Partitions**: The derived MobileNetV2 classification dataset strictly adheres to the exact train/val/test splits of the YOLO dataset.
2. **Smoke-Only Frame Isolation**: Images containing only smoke (without flames) are excluded from the binary fire classifier to prevent ambiguous ground-truth labels.
3. **Deduplication**: Images with identical perceptual dHash values are deduplicated across splits.
