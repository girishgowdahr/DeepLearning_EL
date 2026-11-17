# YOLO Nano Fine-Tuning Guide — ForestGuard AI

This guide explains how to fine-tune **YOLO11n** (classes: `0: fire`, `1: smoke`) on a free cloud GPU (Kaggle or Ultralytics Platform) and integrate the weights into the ForestGuard AI laptop backend.

---

## Method 1: Kaggle Notebook with Free T4 GPU (Recommended)
1. Open a new notebook on [Kaggle](https://www.kaggle.com) with **GPU T4 x2** enabled and **Internet On**.
2. Run in the first cell:
   ```bash
   !pip install ultralytics
   !git clone https://github.com/gaiasd/DFireDataset.git /kaggle/working/dfire
   ```
3. Prepare dataset structure and `data.yaml`:
   ```bash
   python prepare_yolo_dataset.py --dataset_root /kaggle/working/dfire --out_yaml /kaggle/working/data.yaml
   ```
4. Start training:
   ```bash
   python train_yolo.py --data /kaggle/working/data.yaml --model yolo11n.pt --epochs 30 --imgsz 640 --batch 16
   ```
5. Download the trained artifacts:
   - `runs/forestguard/weights/best.pt` -> rename/place to `ForestGuard-AI/model_weights/yolo_best.pt`
   - `runs/forestguard/weights/best.onnx` -> `ForestGuard-AI/model_weights/yolo_best.onnx`
   - Metrics file `yolo_metrics.json` -> `ForestGuard-AI/results/metrics/`
   - Curve images (`yolo_results.png`, `yolo_confusion_matrix.png`) -> `ForestGuard-AI/results/graphs/`
6. Restart the backend to automatically load the new weights!

---

## Method 2: Ultralytics Platform
1. Log in to [Ultralytics Platform](https://platform.ultralytics.com/).
2. Create a new Project named `ForestGuard-AI`.
3. Upload the D-Fire dataset zip or link.
4. Select `yolo11n` and train using free promotional cloud GPU credits.
5. In the model dashboard, click **Export** -> select **ONNX** or download `best.pt`.
6. Place `best.pt` into `ForestGuard-AI/model_weights/yolo_best.pt`.
