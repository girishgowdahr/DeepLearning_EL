# MobileNetV2 CNN Training Guide — ForestGuard AI

This guide describes how to train the MobileNetV2 Fire/No-Fire classifier on a free cloud GPU (Kaggle or Google Colab) and transfer the resulting weights to the laptop backend.

---

## Option 1: Free Cloud Training on Kaggle (Recommended)
1. Log in to [Kaggle](https://www.kaggle.com) and create a **New Notebook**.
2. Under **Notebook options** (right panel):
   - Turn **Accelerator** to **GPU T4 x2**.
   - Ensure **Internet** is toggled **On**.
3. In the first code cell, attach or clone the dataset and training code:
   ```python
   !git clone https://github.com/gaiasd/DFireDataset.git /kaggle/working/dfire
   !pip install torch torchvision onnx onnxruntime matplotlib
   ```
4. Copy and run the contents of `training/cnn/prepare_cnn_dataset.py`:
   ```bash
   python prepare_cnn_dataset.py --source_dir /kaggle/working/dfire --out_dir /kaggle/working/datasets/cnn
   ```
5. Run the two-stage training script:
   ```bash
   python train_cnn.py --data_dir /kaggle/working/datasets/cnn --epochs_stage1 5 --epochs_stage2 8 --out_dir /kaggle/working/model_weights
   ```
6. Evaluate and export ONNX:
   ```bash
   python evaluate_cnn.py --test_dir /kaggle/working/datasets/cnn/test --weights /kaggle/working/model_weights/cnn_mobilenetv2.pt
   python export_onnx.py --weights /kaggle/working/model_weights/cnn_mobilenetv2.pt --out_onnx /kaggle/working/model_weights/cnn_mobilenetv2.onnx
   ```
7. Download the following artifacts and place them into your local `model_weights/` and `results/` folders:
   - `model_weights/cnn_mobilenetv2.onnx` -> `ForestGuard-AI/model_weights/`
   - `model_weights/cnn_mobilenetv2.pt` -> `ForestGuard-AI/model_weights/`
   - `cnn_metrics.json` and `cnn_history.json` -> `ForestGuard-AI/results/metrics/`
   - `cnn_confusion_matrix.png`, `cnn_accuracy.png`, `cnn_loss.png` -> `ForestGuard-AI/results/graphs/`

---

## Option 2: CPU Fallback on Laptop (If Cloud is Blocked)
If cloud GPU access is blocked or unavailable, you can run a reduced training session directly on CPU:
```powershell
.venv\Scripts\python.exe training\cnn\train_cnn.py --epochs_stage1 2 --epochs_stage2 3 --batch_size 16
.venv\Scripts\python.exe training\cnn\export_onnx.py
```
*(Note: CPU training will be clearly labeled as "Trained on CPU with reduced dataset" in MODEL_INFO.json).*
