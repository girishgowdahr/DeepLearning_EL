# ForestGuard AI — Kaggle Training Guide

This directory contains the automated training pipeline for **ForestGuard AI**:
- **Notebook**: [`ForestGuard_AI_Complete_Training.ipynb`](file:///d:/RVCE/Sem3/Deep%20Learning%20EL/ForestGuard-AI/training/kaggle/ForestGuard_AI_Complete_Training.ipynb) (Recommended: 1-click Run All)
- **Standalone Orchestration Script**: `forestguard_kaggle.py` (CLI / cURL alternative)
- **Generator**: `generate_notebook.py` (Builds & validates the notebook)

---

## ⏱️ Computational Budget & Hardware Requirements

| Resource | Specification | Notes |
| :--- | :--- | :--- |
| **Hardware** | **2 × NVIDIA Tesla T4** (or 1 × T4) | Free on Kaggle (30 GPU hours/week) |
| **Internet** | **Enabled** | Required to fetch base weights (`yolo11n.pt`, ImageNet) |
| **Primary Dataset** | `sayedgamal99/smoke-fire-detection-yolo` | CC0 Public Domain / CC BY 4.0 |
| **Alternative Mirrors**| `thunguyen1012/smoke-fire-detection-yolo` | Fully supported via dynamic class mapping |
| **Smoke Test Runtime** | **~3–5 minutes** | Quick verification with `SMOKE_MODE = True` |
| **Full Production Runtime**| **~45–60 minutes** | 30 epochs YOLO + 13 epochs CNN (`SMOKE_MODE = False`) |

---

## 🧠 Dynamic Class Mapping & Architecture Compatibility

Different Kaggle mirrors of D-Fire define class orders differently:
* `sayedgamal99/smoke-fire-detection-yolo` defines `0: smoke, 1: fire`.
* Other mirrors define `0: fire, 1: smoke`.

The notebook **dynamically inspects `data.yaml` at runtime** to resolve raw class IDs:
1. Identifies the exact raw IDs for `fire` and `smoke`.
2. Standardizes the YOLO dataset annotations so that output weights always map `0 -> fire` and `1 -> smoke`.
3. Derives the binary CNN dataset (`fire` vs `no_fire`), strictly excluding smoke-only frames.
4. Generates ONNX models directly matching the FastAPI backend loader interfaces (`CNNClassifier` and `YOLODetector`).

---

## 🚀 Step-by-Step Kaggle Execution Guide

### Step 1: Upload the Notebook to Kaggle
1. Log into [kaggle.com](https://www.kaggle.com).
2. Go to **[kaggle.com/code](https://www.kaggle.com/code)** and click **"New Notebook"**.
3. In the notebook top menu, click **File $\rightarrow$ Import Notebook**.
4. Select the **Upload** tab and drag-and-drop:
   ```
   training/kaggle/ForestGuard_AI_Complete_Training.ipynb
   ```

---

### Step 2: Configure Accelerator & Internet (Right Sidebar)
In the right-hand **Notebook Options / Settings** panel:
1. **Accelerator**: Select **GPU T4 x2** *(requires free phone verification on Kaggle)*.
2. **Internet**: Toggle **Internet ON**.

---

### Step 3: Attach the D-Fire Dataset
1. In the top right panel, click **"+ Add Input"**.
2. In the search box, type:
   ```
   smoke-fire-detection-yolo
   ```
3. Locate **`sayedgamal99/smoke-fire-detection-yolo`**.
4. Click the **"+"** button to attach it to the notebook.
   *The notebook automatically detects the dataset root in `/kaggle/input`, parses `data.yaml`, and verifies train/val/test splits.*

---

### Step 4: Choose Execution Mode & Run

In **Cell 1**:
- For a quick validation test (~3 minutes):
  ```python
  SMOKE_MODE = True
  ```
- For full academic training (~45–60 minutes):
  ```python
  SMOKE_MODE = False
  ```

#### Option A: Interactive Run
Click **"Run All"** from the top menu toolbar. Monitor the cell outputs in real time.

#### Option B: Background Run (Recommended)
1. Click **"Save Version"** (top right).
2. Select **"Save & Run All (Commit)"**.
3. You can safely close your browser or turn off your laptop! Kaggle's cloud servers will continue training until finished.

---

### Step 5: Download the Trained Artifacts
Once training completes:
1. Open the notebook's **Output** panel on the right sidebar (or open the saved version's Output section).
2. You will see two zip archives:
   - `yolo_artifacts.zip` (~15–20 MB): Contains `yolo_best.pt`, `yolo_best.onnx`, `yolo_metrics.json`, PR curves, confusion matrix.
   - `cnn_artifacts.zip` (~10–15 MB): Contains `cnn_mobilenetv2.onnx`, `cnn_mobilenetv2_best.pt`, `cnn_metrics.json`, loss/accuracy curves.
3. Click the three dots next to each file $\rightarrow$ **Download**.

---

### Step 6: Install Artifacts on Your Laptop
Move the downloaded zip files to your laptop and execute the automated installer in PowerShell:
```powershell
.\scripts\install_artifacts.ps1 -YoloZip "C:\path\to\yolo_artifacts.zip" -CnnZip "C:\path\to\cnn_artifacts.zip"
```

The script extracts the weights to `model_weights/`, metrics to `results/metrics/`, plots to `results/graphs/`, and prints SHA-256 hashes matching the notebook's `MANIFEST.txt`.

Next, run the validation suite:
```powershell
python scripts\validate_models.py
```
Both models will now report **`AVAILABLE`**, and the live detection dashboard will run real neural network inference!
