# MobileNetV2 CNN Training — Kaggle Handoff Guide

This guide describes how to train the **MobileNetV2 binary fire classifier** on Kaggle's free GPU in **~15–20 minutes**.

---

## ⚡ Quick 5-Step Kaggle Checklist

1. **Open Kaggle & Create a New Notebook**:
   - Go to [kaggle.com/code](https://www.kaggle.com/code) and click **"New Notebook"**.
   - Make sure your Kaggle account is phone-verified (required to use free GPU).

2. **Configure Accelerator & Internet (Right Sidebar)**:
   - Click the **Notebook options (three dots)** or the Settings panel on the right.
   - **Accelerator**: Select **GPU T4 x2** (or **GPU T4**).
   - **Internet**: Toggle **Internet ON** (required to download base PyTorch model weights).

3. **Attach Dataset**:
   - Click **"+ Add Input"** (top right).
   - Search for `smoke-fire-detection-yolo` or `dfire`.
   - Select and add **`thunguyen1012/smoke-fire-detection-yolo`** (or `gaiasd/dfire-dataset`).
   - *License*: Licence NOT yet verified; check the original D-Fire source before redistribution.

4. **Paste and Run the Script**:
   - Copy the entire contents of [`kaggle_train_cnn.py`](kaggle_train_cnn.py).
   - Paste it into the first notebook cell and run it (**Shift + Enter**).
   - *What happens automatically*:
     - Runs a **1-epoch smoke test on 200 images** to confirm GPU execution within 30 seconds.
     - Automatically derives binary Fire / No-Fire splits (excluding ambiguous smoke-only frames).
     - Caps training at 3,000 images per class for balanced, fast convergence.
     - Trains 5 warmup epochs (head) + 8 fine-tune epochs (top blocks).
     - Evaluates test set (Accuracy, Precision, Recall, F1).
     - Exports `cnn_mobilenetv2.onnx`.
     - Packages everything into a single zip file: `/kaggle/working/cnn_artifacts.zip`.

5. **Download and Place Artifacts**:
   - In the right sidebar under **Output**, click the three dots next to `cnn_artifacts.zip` and select **Download**.
   - Unzip `cnn_artifacts.zip` into your local `ForestGuard-AI/` directory:
     - `cnn_mobilenetv2.onnx` -> `model_weights/cnn_mobilenetv2.onnx`
     - `cnn_mobilenetv2_best.pt` -> `model_weights/cnn_mobilenetv2_best.pt`
     - `confusion_matrix.png` -> `results/graphs/cnn_confusion_matrix.png`
     - `training_history.png` -> `results/graphs/cnn_training_history.png`
     - `metrics.json` -> `results/metrics/cnn_metrics.json`
