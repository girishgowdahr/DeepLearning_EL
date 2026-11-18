# ForestGuard AI — Kaggle Dual-GPU Training Guide

Follow these numbered steps to train both models (**YOLO11n** and **MobileNetV2 CNN**) concurrently on Kaggle's free GPUs (**2 × NVIDIA Tesla T4**).

---

## ⏱️ Time Budget & Resources
- **Hardware**: 2 × NVIDIA Tesla T4 (Free on Kaggle)
- **Quota**: ~30 GPU-hours per week (this run uses ~1.5 GPU-hours total)
- **Total Duration**:
  - **Smoke Test Mode (`SMOKE=1`)**: **~4–6 minutes** (Run this first to verify GPU execution!)
  - **Full Production Training**: **~45–60 minutes** (Runs YOLO and CNN concurrently on separate GPUs)

---

## Step 1: Open Kaggle & Create Notebook
1. Log into [kaggle.com](https://www.kaggle.com) with your account.
2. Go to [kaggle.com/code](https://www.kaggle.com/code) and click **"New Notebook"**.

---

## Step 2: Configure Accelerator & Internet (Right Sidebar)
In the right-hand **Notebook Options / Settings** panel:
1. **Accelerator**: Select **GPU T4 x2** *(Note: phone verification on Kaggle is required for GPU access)*.
2. **Internet**: Toggle **Internet ON** *(Required to download base pretrained weights and packages)*.

---

## Step 3: Attach the D-Fire Dataset
1. Click **"+ Add Input"** (at top right).
2. Search for: `smoke-fire-detection-yolo`
3. Look for **`thunguyen1012/smoke-fire-detection-yolo`** (or `gaiasd/dfire-dataset`).
4. Click the **+** button to attach it to the notebook.
   - *License*: CC0 / CC BY 4.0. Completely free, no permission requests or payments needed.

---

## Step 4: Run Smoke Test First (~5 minutes)
In the first notebook cell, paste this command to test the pipeline on a mini-batch of 200 images:

```python
import os
os.environ["SMOKE"] = "1"

# Paste or run the forestguard_kaggle.py script
!curl -s https://raw.githubusercontent.com/girishgowdahr/DeepLearning_EL/main/training/kaggle/forestguard_kaggle.py -o forestguard_kaggle.py
!python forestguard_kaggle.py
```
*(Or copy-paste the entire contents of `training/kaggle/forestguard_kaggle.py` into the cell and set `SMOKE_MODE = True` at the top).*

- **Expected result**: Passes in ~5 minutes, creating test checkpoints and verifying CUDA on both GPUs.

---

## Step 5: Start Full Production Run (~45–60 minutes)
Once the smoke test passes:
1. Create a new cell or set `SMOKE_MODE = False` (or run without `SMOKE=1`):
   ```python
   import os
   os.environ["SMOKE"] = "0"
   !python forestguard_kaggle.py
   ```
2. **Recommended for Background Execution**:
   - Click **Save Version** (top right) $\rightarrow$ select **"Save & Run All (Commit)"**.
   - You can now safely close your browser or laptop! Kaggle's cloud servers will continue training until finished.

---

## Step 6: Download the Artifacts
Once training finishes:
1. Look at the right panel under **Output** (or under the saved version's Output section).
2. You will see two zip files:
   - `yolo_artifacts.zip` (~15–20 MB)
   - `cnn_artifacts.zip` (~10–15 MB)
3. Click the three dots next to each file and select **Download**.

---

## Step 7: Place Artifacts on Your Laptop
On your laptop, run the automated integration script in PowerShell:
```powershell
.\scripts\install_artifacts.ps1 -YoloZip "C:\path\to\yolo_artifacts.zip" -CnnZip "C:\path\to\cnn_artifacts.zip"
```
Or manually extract:
- `yolo_best.pt` $\rightarrow$ `model_weights/yolo_best.pt`
- `yolo_best.onnx` $\rightarrow$ `model_weights/yolo_best.onnx`
- `cnn_mobilenetv2.onnx` $\rightarrow$ `model_weights/cnn_mobilenetv2.onnx`
- `cnn_mobilenetv2_best.pt` $\rightarrow$ `model_weights/cnn_mobilenetv2_best.pt`
- `MODEL_INFO.json` $\rightarrow$ `model_weights/MODEL_INFO.json`
- `yolo_metrics.json` & `cnn_metrics.json` $\rightarrow$ `results/metrics/`
- Plots (`*.png`) $\rightarrow$ `results/graphs/`
