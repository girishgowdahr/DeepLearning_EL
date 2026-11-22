# Finalization runbook (Kaggle -> install -> evaluate -> Android)

Run everything from the repo root on YOUR laptop. Nothing here trains on the laptop; Kaggle's free GPU does the training.
Status of each step is only "done" when the command output says so. Anything that failed stays failed in the report.

## Phase 1 - Kaggle (needs your Kaggle account; I cannot do this part for you)
One-time, by you (these are approval steps that must not be automated):
1. Kaggle account: verify your phone number (unlocks GPU). 2. Open `https://www.kaggle.com/datasets/sayedgamal99/smoke-fire-detection-yolo`
   once and check the licence/terms shown on the page; record them in `datasets/DATASET_INFO.md`.
3. Kaggle -> Settings -> API -> create a token yourself. Set it in your shell; never paste it into chat or commit it:
   `$env:KAGGLE_API_TOKEN = "<token>"` (or follow the CLI's own `kaggle` auth instructions). Then `pip install kaggle`.

```powershell
.venv\Scripts\python.exe scripts\kaggle_run.py --owner YOUR_KAGGLE_USERNAME --smoke     # ~5 min: proves GPU + pipeline + packaging
.venv\Scripts\python.exe scripts\kaggle_run.py --owner YOUR_KAGGLE_USERNAME             # full run, monitors until Kaggle says COMPLETE
# if your window closed while it ran:
.venv\Scripts\python.exe scripts\kaggle_run.py --owner YOUR_KAGGLE_USERNAME --attach-only
```
Success = Kaggle status COMPLETE **and** both `yolo_artifacts.zip` / `cnn_artifacts.zip` verified. The smoke run produces
pipeline-check weights, not trained models. Output: `run\kaggle_output\`. Free quota is limited (~30 GPU hours/week); a failed push
or a timeout consumes nothing you did not already start.

## Phase 2 - install and verify
```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\finalize_forestguard.ps1 -Strict `
  -YoloZip .\run\kaggle_output\yolo_artifacts.zip -CnnZip .\run\kaggle_output\cnn_artifacts.zip
```
Stop on the first failure; read `results\final_verification_report.txt`.

## Phase 3 - evaluation on held-out test data
`scripts\evaluate_models.py` reads `datasets\cnn\test\{fire,no_fire}` and `datasets\yolo\data.yaml`. Those folders are NOT in git and are
NOT in the artifact zips, so a local re-evaluation needs the D-Fire test split on the laptop and a `data.yaml` with local paths (the
Kaggle-written one points at `/kaggle/working/...`). Until you have that, the genuine held-out numbers are the ones the Kaggle notebook
computed on its test split and wrote to `cnn_metrics.json` / `yolo_metrics.json` inside the artifacts (check `"metrics_split"`).
```powershell
.venv\Scripts\python.exe scripts\evaluate_models.py --strict
```
If it says SKIPPED, that is the truth: do not fill the paper with other numbers. `scripts\update_paper_metrics.py` then syncs the paper.
A `metrics_split` starting with `VALIDATION` means the notebook did not finish its test evaluation: do not report it as test performance.

## Phase 4 - Android (USB debugging on, phone unlocked, laptop app running with an HTTPS tunnel)
```powershell
.\scripts\run_demo.ps1                                              # window 1
.venv\Scripts\python.exe scripts\android_check.py                    # window 2: launches Chrome on the phone, watches the server
```
You must tap **Start Camera** and **Allow** on the phone and confirm the items it lists under "REQUIRES HUMAN CONFIRMATION".
