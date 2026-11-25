# ForestGuard AI — Dataset Leakage & Annotation Integrity Audit Report

**Audit Protocol Version**: 1.0.0
**Audit Timestamp**: November 25, 2026
**Auditor**: Antigravity Empirical Research Suite
**Audit Script**: `scripts/audit_dataset_leakage.py` (Untracked in working tree; pending commit approval)
**Dataset Source**: Kaggle (`sayedgamal99/smoke-fire-detection-yolo`)
**Actual Dataset Audit Status**: `PENDING_OFFLINE_DATASET` (Full image dataset resides on Kaggle)

---

## 1. Executive Summary & Verdict

> [!IMPORTANT]
> **Actual Dataset Audit Status**: **PENDING FULL RAW DATASET EXTRACTION**
> The full raw dataset (over 20,000 high-resolution images, >2 GB) was processed and trained on Kaggle cloud instances and is not stored inside the local Git repository (compliant with `.gitignore` standards for binary assets). Consequently, **no claim of zero dataset leakage is made** in project documentation or academic publications until direct pixel-level execution of this audit suite is conducted on the complete raw image archive.

A reproducible audit tool (`scripts/audit_dataset_leakage.py`) has been authored and placed in the working tree to allow automated, deterministic verification whenever the raw dataset is mounted.

---

## 2. Dataset Architecture & Split Boundaries

The ForestGuard AI dual-model system is trained on imagery derived from the D-Fire wildland surveillance dataset:
- **YOLO11 Nano**: Spatial object detector requiring 2-class bounding boxes (`0: smoke, 1: fire`).
- **MobileNetV2 CNN**: Frame-level binary classifier (`0: fire, 1: no_fire`), excluding smoke-only imagery to avoid ambiguous ground truth.

### Configured Split Limits (from Kaggle Training Pipeline):
- **Train Split**: Capped at 6,000 images for YOLO; 3,000 per class for CNN.
- **Validation Split**: Capped at 1,000 samples for YOLO; 500 per class for CNN.
- **Held-Out Test Split**: Capped at 1,000 samples for YOLO; 500 per class for CNN.

---

## 3. Audit Methodology

The audit tool implements three orthogonal tiers of dataset verification:

### Tier 1: Cryptographic Duplicate Detection (SHA-256)
- Computes streaming 64KB-chunked SHA-256 hashes for all image files across all splits.
- Compares hash sets between splits ($\text{Train} \cap \text{Val}$, $\text{Train} \cap \text{Test}$, $\text{Val} \cap \text{Test}$) to detect exact bit-for-bit file duplicates across train and evaluation partitions.
- Scans intra-split duplicate counts to measure dataset redundancy.

### Tier 2: Perceptual Hash Analysis (Difference Hash / dHash)
In wildfire video surveillance, consecutive frames from a fixed camera tower may differ by slight compression artifacts or camera sensor noise while depicting the identical physical tree canopy or fire column. Bitwise SHA-256 fails to detect these near-duplicates.
- **dHash Algorithm**:
  1. Grayscale conversion.
  2. Lanczos resampling to $(8+1) \times 8 = 9 \times 8$ pixels.
  3. Row-wise adjacent gradient comparison: $P[x, y] > P[x+1, y]$.
  4. Bit packing into a 64-bit unsigned integer hash.
- **Near-Duplicate Metric**: Hamming distance:
  $$\text{dist}(h_1, h_2) = \sum_{i=0}^{63} (h_1 \oplus h_2)_i$$
  Pairs with $\text{dist} \le 5$ bits are flagged as potential perceptual duplicates.

### Tier 3: Annotation Integrity & Bounding Box Sanity
- **Label Alignment**: Validates that every image has an associated YOLO `.txt` file.
- **Empty Labels**: Recognizes zero-byte files as valid negative (background) forest samples.
- **Coordinate Sanity**: Validates that normalized YOLO coordinates satisfy:
  $$x_c, y_c \in [0.0, 1.0], \quad w, h \in (0.0, 1.0]$$
- **Class Identifier Range**: Confirms all annotations use valid class IDs ($0$ or $1$).

---

## 4. Synthetic Tool Validation Results (Algorithm Sanity Test Only)

> [!NOTE]
> The numbers below are from an isolated **synthetic unit test** (`scratch/verify_leakage_tool.py`) created solely to verify that `audit_dataset_leakage.py` detects duplicates and near-duplicates correctly. **These figures do NOT reflect the actual Kaggle dataset**, which has not yet been audited locally.

The auditing script was verified using synthetic mock image-label vectors:
- Created mock directories containing known identical images, known background negatives, and labeled objects.
- **Synthetic Test Result**:
  - Exact SHA-256 cross-split duplicates detected: **100% precision (1/1 detected)**.
  - Perceptual near-duplicates detected: **100% precision (2/2 detected)**.
  - Negative background images parsed: **100% precision**.
  - Bounding box out-of-bounds detection: Verified.

---

## 5. Current Local Repository Findings

Running `python scripts/audit_dataset_leakage.py` against the local repository yields:

```json
{
  "timestamp": "2026-11-24 15:57:02 UTC",
  "status": "PENDING_OFFLINE_DATASET",
  "dataset_directory": null,
  "hamming_threshold": 5,
  "images_audited": {},
  "exact_duplicates": {
    "intra_split": [],
    "inter_split_leakage": []
  },
  "perceptual_near_duplicates": {
    "inter_split_leakage": []
  },
  "annotation_integrity": {
    "missing_labels": [],
    "corrupted_labels": [],
    "out_of_bounds_coords": [],
    "invalid_classes": []
  },
  "class_distribution": {},
  "verdict": "AUDIT PENDING: Full raw image dataset is not stored in local Git clone (expected per .gitignore). Training dataset resides on Kaggle (sayedgamal99/smoke-fire-detection-yolo, 20,000+ images). No zero-leakage claim is made without direct execution of this script against full raw images."
}
```

---

## 6. Execution Instructions for Full Dataset (Kaggle or Local)

### Execution on Kaggle (Cloud Environment)
Inside a Kaggle notebook attached to `sayedgamal99/smoke-fire-detection-yolo`:
```bash
python scripts/audit_dataset_leakage.py \
  --dataset-dir /kaggle/input/smoke-fire-detection-yolo \
  --threshold 5 \
  --json-out results/dataset_leakage_audit_report.json \
  --md-out docs/dataset_leakage_audit.md
```

### Execution on Local Workstation (with Downloaded Dataset)
```bash
# 1. Download raw dataset from Kaggle
kaggle datasets download -d sayedgamal99/smoke-fire-detection-yolo
unzip smoke-fire-detection-yolo.zip -d datasets/raw_dfire

# 2. Run the audit script
python scripts/audit_dataset_leakage.py \
  --dataset-dir datasets/raw_dfire \
  --threshold 5
```

---

## 7. Academic Integrity & Paper Alignment

In accordance with strict scientific guidelines:
1. The project IEEE report outline (`docs/academic_report_outline.md`), paper draft (`docs/paper/forestguard_paper.md`), and LaTeX manuscript (`docs/paper/forestguard_paper.tex`) do **not** claim a "zero-leakage guarantee."
2. The paper explicitly documents that cross-split contamination risk is mitigated via fixed directory-level partition boundaries in the original dataset distribution, while acknowledging that camera sensor correlation in static towers represents an inherent challenge in natural wildfire datasets.
