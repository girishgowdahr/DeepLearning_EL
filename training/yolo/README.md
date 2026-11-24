# YOLO11 Nano Training — Kaggle Handoff Guide

This guide describes how to train the **YOLO11n fire & smoke bounding-box detector** on Kaggle's free GPU in **~45–60 minutes**.

---

## ⚡ Quick 5-Step Kaggle Checklist

1. **Open Kaggle & Create a New Notebook**:
   - Go to [kaggle.com/code](https://www.kaggle.com/code) and click **"New Notebook"**.
   - Make sure your Kaggle account is phone-verified (required to use free GPU).

2. **Configure Accelerator & Internet (Right Sidebar)**:
   - Click the Settings panel on the right.
   - **Accelerator**: Select **GPU T4 x2** (or **GPU T4**).
   - **Internet**: Toggle **Internet ON** (required for `ultralytics` to download base `yolo11n.pt`).

3. **Attach Dataset**:
   - Click **"+ Add Input"** (top right).
   - Search for `smoke-fire-detection-yolo`.
   - Select and add **`thunguyen1012/smoke-fire-detection-yolo`** (or `gaiasd/dfire-dataset`).
   - *License*: Licence NOT yet verified (see datasets/DATASET_INFO.md); check the original D-Fire source before redistribution.

4. **Paste and Run the Script**:
   - Copy the entire contents of [`kaggle_train_yolo.py`](kaggle_train_yolo.py).
   - Paste it into the notebook cell and run it (**Shift + Enter**).
   - *What happens automatically*:
     - Runs a **1-epoch smoke test on a mini dataset** to confirm GPU execution within 30 seconds.
     - Caps training set to **6,000 images** to guarantee completion within ~45–60 minutes on Kaggle T4.
     - Trains YOLO11n for 30 epochs with `imgsz=640`, `batch=16`, `patience=10`.
     - Evaluates test set metrics (mAP50, mAP50-95, Fire/Smoke PR curves).
     - Exports `yolo_best.onnx` with dynamic batching.
     - Packages everything into a single zip file: `/kaggle/working/yolo_artifacts.zip`.

5. **Download and Place Artifacts**:
   - In the right sidebar under **Output**, click the three dots next to `yolo_artifacts.zip` and select **Download**.
   - Unzip `yolo_artifacts.zip` into your local `ForestGuard-AI/` directory:
     - `yolo_best.pt` -> `model_weights/yolo_best.pt`
     - `yolo_best.onnx` -> `model_weights/yolo_best.onnx`
     - `results.png` -> `results/graphs/yolo_results.png`
     - `confusion_matrix.png` -> `results/graphs/yolo_confusion_matrix.png`
     - `PR_curve.png` -> `results/graphs/yolo_PR_curve.png`
     - `F1_curve.png` -> `results/graphs/yolo_F1_curve.png`
     - `metrics.json` -> `results/metrics/yolo_metrics.json`
