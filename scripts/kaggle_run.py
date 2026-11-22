"""
ForestGuard AI — Kaggle submit / monitor / download automation.

Pushes training/kaggle/ForestGuard_AI_Complete_Training.ipynb to Kaggle as a private GPU notebook
(T4 x2), polls until Kaggle reports a final status, downloads the output, and verifies that BOTH
yolo_artifacts.zip and cnn_artifacts.zip exist and are valid zip files.

Success is only reported when Kaggle itself says COMPLETE *and* the artifacts verify. Credentials are
read by the kaggle CLI from its own config/env; this script never prints or stores them.

Usage (from the repo root, with the kaggle CLI installed and authenticated on YOUR machine):
    python scripts/kaggle_run.py --owner YOUR_KAGGLE_USERNAME --smoke      # ~5 min pipeline check
    python scripts/kaggle_run.py --owner YOUR_KAGGLE_USERNAME              # full run
    python scripts/kaggle_run.py --owner YOUR_KAGGLE_USERNAME --attach-only --slug forestguard-ai-training
        (skip the push; just monitor an already running/finished notebook and download its output)
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "training" / "kaggle" / "ForestGuard_AI_Complete_Training.ipynb"
DATASET = "sayedgamal99/smoke-fire-detection-yolo"
FINAL_OK = ("COMPLETE",)
FINAL_BAD = ("ERROR", "CANCEL", "FAIL")
ARTIFACTS = ("yolo_artifacts.zip", "cnn_artifacts.zip")


def run(cmd, check=False):
    p = subprocess.run(cmd, capture_output=True, text=True)
    out = (p.stdout or "") + (p.stderr or "")
    if check and p.returncode != 0:
        raise SystemExit(f"Command failed ({p.returncode}): {' '.join(cmd[:3])} ...\n{out[-800:]}")
    return p.returncode, out


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def credentials_present() -> bool:
    """Presence check only. Values are never read into output."""
    if os.environ.get("KAGGLE_API_TOKEN") or (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")):
        return True
    try:
        home = Path.home() / ".kaggle"
        return any((home / n).exists() for n in ("kaggle.json", "credentials.json", "access_token"))
    except OSError:
        return False


def resolve_kaggle_cmd(kaggle_bin: str = "kaggle", python_bin: str | None = None) -> list[str] | None:
    """
    Resolve Kaggle command into an executable argv list.
    Supports Windows-compatible path resolution and falls back to
    [sys.executable, "-m", "kaggle"] when a standalone executable is not found.
    """
    py = python_bin or sys.executable

    # 1. Direct path or system PATH lookup
    if kaggle_bin:
        found = shutil.which(kaggle_bin)
        if found:
            return [found]
        p = Path(kaggle_bin)
        if p.is_file():
            return [str(p.resolve())]
        if sys.platform.startswith("win"):
            for ext in (".exe", ".cmd", ".bat"):
                p_ext = Path(f"{kaggle_bin}{ext}")
                if p_ext.is_file():
                    return [str(p_ext.resolve())]
                found_ext = shutil.which(f"{kaggle_bin}{ext}")
                if found_ext:
                    return [found_ext]

    # 2. Check python executable's directory (e.g. .venv/Scripts/kaggle.exe on Windows, bin/kaggle on POSIX)
    if py:
        py_dir = Path(py).parent
        cand = py_dir / kaggle_bin
        if cand.is_file():
            return [str(cand.resolve())]
        if sys.platform.startswith("win"):
            for ext in (".exe", ".cmd", ".bat"):
                cand_ext = py_dir / f"{kaggle_bin}{ext}"
                if cand_ext.is_file():
                    return [str(cand_ext.resolve())]

    # 3. Module fallback: [sys.executable, "-m", "kaggle"]
    if kaggle_bin in ("kaggle", "kaggle.exe"):
        try:
            if py == sys.executable:
                import importlib.util
                if importlib.util.find_spec("kaggle") is not None:
                    return [py, "-m", "kaggle"]
            else:
                proc = subprocess.run([py, "-m", "kaggle", "--version"], capture_output=True, text=True, timeout=5)
                if proc.returncode == 0:
                    return [py, "-m", "kaggle"]
        except Exception:
            pass

    return None


def build_kernel_folder(dest: Path, owner: str, slug: str, smoke: bool) -> Path:
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    text = NOTEBOOK.read_text(encoding="utf-8")
    flag = '"SMOKE_MODE = os.environ.get(\\"SMOKE\\", \\"0\\") == \\"1\\" or False\\n"'
    if smoke:
        if flag not in text:
            raise SystemExit("Cannot enable smoke mode: SMOKE_MODE line not found in the notebook (it changed). Aborting.")
        text = text.replace("or False\\n\"", "or True\\n\"", 1)
    nb_name = NOTEBOOK.name
    (dest / nb_name).write_text(text, encoding="utf-8")
    meta = {
        "id": f"{owner}/{slug}",
        "title": slug.replace("-", " ").title(),
        "code_file": nb_name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [DATASET],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (dest / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return dest


def parse_status(text: str) -> str:
    m = re.search(r"(COMPLETE|ERROR|CANCEL\w*|FAIL\w*|RUNNING|QUEUED|NEW)", text.upper())
    return m.group(1) if m else "UNKNOWN"


def verify_artifacts(folder: Path) -> list:
    problems, report = [], []
    for name in ARTIFACTS:
        matches = list(folder.rglob(name))
        if not matches:
            problems.append(f"{name}: NOT FOUND in downloaded output")
            continue
        z = matches[0]
        try:
            with zipfile.ZipFile(z) as zf:
                bad = zf.testzip()
                n = len(zf.namelist())
            if bad:
                problems.append(f"{name}: corrupt member {bad}")
            elif n == 0 or z.stat().st_size < 1024:
                problems.append(f"{name}: empty archive")
            else:
                report.append(f"{name}  {z.stat().st_size/1e6:.2f} MB  {n} files  sha256={sha256(z)}  -> {z}")
        except zipfile.BadZipFile:
            problems.append(f"{name}: not a valid zip")
    for line in report:
        print("  OK  " + line)
    return problems


def log_has_traceback(folder: Path) -> bool:
    for lg in list(folder.rglob("*.log")) + list(folder.rglob("*.json")):
        try:
            if "Traceback (most recent call last)" in lg.read_text(encoding="utf-8", errors="ignore"):
                return True
        except Exception:
            pass
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--owner", required=True, help="your Kaggle username")
    ap.add_argument("--slug", default="forestguard-ai-training")
    ap.add_argument("--smoke", action="store_true", help="1-epoch pipeline check instead of the full run")
    ap.add_argument("--timeout-min", type=int, default=240, help="give up monitoring after this many minutes")
    ap.add_argument("--poll-sec", type=int, default=60)
    ap.add_argument("--kaggle-bin", default="kaggle")
    ap.add_argument("--out", default=str(ROOT / "run" / "kaggle_output"))
    ap.add_argument("--attach-only", action="store_true", help="do not push; monitor/download an existing notebook")
    ap.add_argument("--dry-run", action="store_true", help="build the kernel folder and stop (no network)")
    a = ap.parse_args()

    kernel_ref = f"{a.owner}/{a.slug}"
    print(f"Notebook: {kernel_ref}   mode: {'SMOKE (1 epoch)' if a.smoke else 'FULL'}   GPU: T4 x2 only")

    if not NOTEBOOK.exists():
        print(f"FAILED: {NOTEBOOK} not found. Run from a full checkout.")
        return 2
    folder = build_kernel_folder(ROOT / "run" / "kaggle_kernel", a.owner, a.slug, a.smoke)
    print(f"Kernel folder prepared: {folder}")
    if a.dry_run:
        print("Dry run: stopping before any network call.")
        return 0

    kaggle_cmd = resolve_kaggle_cmd(a.kaggle_bin)
    if not kaggle_cmd:
        print("FAILED: kaggle CLI not found. Install: pip install kaggle   (then authenticate; see docs/finalization_runbook.md)")
        return 2
    if not credentials_present():
        print("FAILED: no Kaggle credentials found (KAGGLE_API_TOKEN, or KAGGLE_USERNAME+KAGGLE_KEY, or ~/.kaggle).")
        print("        Create an API token in your Kaggle account settings yourself; this script will not do it for you.")
        return 2

    if not a.attach_only:
        print("Pushing notebook (private, GPU T4 x2, internet on) ...")
        code, out = run(kaggle_cmd + ["kernels", "push", "-p", str(folder), "--accelerator", "NvidiaTeslaT4"])
        print(out.strip()[-600:])
        if code != 0 or re.search(r"error|invalid|forbidden|unauthori", out, re.I):
            print("FAILED: Kaggle rejected the push. Nothing was trained. Typical causes: dataset not accessible to your account "
                  "(open the dataset page and accept its terms once), phone verification missing (needed for GPU), or an invalid slug.")
            return 1

    deadline = time.time() + a.timeout_min * 60
    status, last = "UNKNOWN", None
    while time.time() < deadline:
        code, out = run(kaggle_cmd + ["kernels", "status", kernel_ref])
        status = parse_status(out)
        if status != last:
            print(f"[{time.strftime('%H:%M:%S')}] Kaggle status: {status}")
            last = status
        if status in FINAL_OK or any(status.startswith(b) for b in FINAL_BAD):
            break
        time.sleep(a.poll_sec)
    else:
        print(f"TIMED OUT after {a.timeout_min} min (last status: {status}). The notebook may still be running on Kaggle; "
              f"re-run with --attach-only to resume monitoring. NOT reporting success.")
        return 3

    if status not in FINAL_OK:
        print(f"FAILED: Kaggle reports {status}. No successful training can be claimed. Open the notebook log on Kaggle.")
        return 1

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Kaggle reports COMPLETE. Downloading output to {out_dir} ...")
    code, out = run(kaggle_cmd + ["kernels", "output", kernel_ref, "-p", str(out_dir), "-o", "-q"])
    if code != 0:
        print("FAILED: output download failed:\n" + out[-500:])
        return 1

    print("Verifying artifacts:")
    problems = verify_artifacts(out_dir)
    if log_has_traceback(out_dir):
        problems.append("notebook log contains a Python traceback although Kaggle says COMPLETE - inspect it before trusting the weights")
    if problems:
        print("\nVERIFICATION FAILED:")
        for p in problems:
            print("  - " + p)
        return 1
    if a.smoke:
        print("\nSMOKE run verified: the pipeline works end to end. These weights are NOT trained models - run again without --smoke.")
        return 0
    print("\nVERIFIED: Kaggle COMPLETE and both artifact zips are valid.")
    print(f"Next: powershell .\\scripts\\finalize_forestguard.ps1 -Strict -YoloZip <path>\\yolo_artifacts.zip -CnnZip <path>\\cnn_artifacts.zip")
    return 0


if __name__ == "__main__":
    sys.exit(main())
