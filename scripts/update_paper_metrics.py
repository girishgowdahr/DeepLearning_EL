"""
ForestGuard AI — Paper Metrics Synchronizer & LaTeX Finalizer
=============================================================
Safely updates docs/paper/forestguard_paper.tex and forestguard_paper.md
with genuine evaluation metrics from results/metrics/ once available.
Compiles the LaTeX document to PDF if compiler tools (pdflatex/xelatex/latexmk) are installed.
Keeps unavailable metrics marked PENDING. Never fabricates results.
"""

import sys
import os
import json
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def try_compile_latex(tex_path: Path) -> bool:
    """Attempts to compile the LaTeX paper if a compiler is found in PATH."""
    compiler = shutil.which("pdflatex") or shutil.which("xelatex") or shutil.which("latexmk")
    if not compiler:
        print("ℹ Notice: LaTeX compiler (pdflatex / xelatex / latexmk) not installed on local host.")
        print("   LaTeX source forestguard_paper.tex is synchronized and ready for Overleaf or cloud compilation.")
        return False

    try:
        print(f"📄 Compiling LaTeX paper with {Path(compiler).name}...")
        cmd = [compiler, "-interaction=nonstopmode", f"-output-directory={tex_path.parent}", str(tex_path)]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        pdf_path = tex_path.with_suffix(".pdf")
        if pdf_path.exists():
            print(f"✅ Successfully compiled IEEE paper to PDF: {pdf_path}")
            return True
        else:
            print(f"⚠️ Notice: LaTeX compilation finished without generating PDF output.")
            return False
    except Exception as e:
        print(f"⚠️ LaTeX compilation error: {e}")
        return False

def main():
    yolo_metrics_path = ROOT / "results" / "metrics" / "yolo_metrics.json"
    cnn_metrics_path = ROOT / "results" / "metrics" / "cnn_metrics.json"
    paper_tex_path = ROOT / "docs" / "paper" / "forestguard_paper.tex"
    paper_md_path = ROOT / "docs" / "paper" / "forestguard_paper.md"

    if not yolo_metrics_path.exists() or not cnn_metrics_path.exists():
        print("⚠️ Notice: Genuine training metrics not found in results/metrics/.")
        print("   forestguard_paper.tex remains marked as PENDING.")
        print("   Metrics will be populated automatically when Kaggle artifacts are installed.")
        # Attempt compilation of draft if requested
        if "--compile" in sys.argv:
            try_compile_latex(paper_tex_path)
        return 0

    print("📄 Found genuine model evaluation metrics:")
    print(f"   YOLO: {yolo_metrics_path}")
    print(f"   CNN:  {cnn_metrics_path}")

    with open(yolo_metrics_path, "r", encoding="utf-8") as f:
        yolo = json.load(f)
    with open(cnn_metrics_path, "r", encoding="utf-8") as f:
        cnn = json.load(f)

    # Extract genuine metrics
    cnn_acc = f"{float(cnn.get('test_accuracy', cnn.get('accuracy', 0.0))) * 100:.1f}\\%"
    cnn_prec = f"{float(cnn.get('classes', {}).get('fire', {}).get('precision', cnn.get('precision', 0.0))):.3f}"
    cnn_rec = f"{float(cnn.get('classes', {}).get('fire', {}).get('recall', cnn.get('recall', 0.0))):.3f}"

    test_metrics = yolo.get('test', {}) if isinstance(yolo.get('test'), dict) else {}
    yolo_map = f"{float(yolo.get('mAP50', test_metrics.get('mAP50', 0.0))):.3f}"
    yolo_prec = f"{float(yolo.get('precision', test_metrics.get('precision', 0.0))):.3f}"
    yolo_rec = f"{float(yolo.get('recall', test_metrics.get('recall', 0.0))):.3f}"

    print(f"   CNN Metrics:  Acc={cnn_acc}, Prec={cnn_prec}, Recall={cnn_rec}")
    print(f"   YOLO Metrics: mAP50={yolo_map}, Prec={yolo_prec}, Recall={yolo_rec}")

    # Update LaTeX document
    if paper_tex_path.exists():
        tex_content = paper_tex_path.read_text(encoding="utf-8")
        
        # Replace CNN row
        tex_content = re.sub(
            r"MobileNetV2 \(CNN\)\s*&\s*[^&]+\s*&\s*[^&]+\s*&\s*[^&]+\s*&\s*\\sim8\.5 ms",
            f"MobileNetV2 (CNN) & {cnn_acc} & {cnn_prec} & {cnn_rec} & $\\\\sim$8.5 ms",
            tex_content
        )
        # Replace YOLO row
        tex_content = re.sub(
            r"YOLO11n \(Detector\)\s*&\s*[^&]+\s*&\s*[^&]+\s*&\s*[^&]+\s*&\s*\\sim22\.0 ms",
            f"YOLO11n (Detector) & {yolo_map} & {yolo_prec} & {yolo_rec} & $\\\\sim$22.0 ms",
            tex_content
        )
        
        paper_tex_path.write_text(tex_content, encoding="utf-8")
        print(f"✅ Updated LaTeX paper: {paper_tex_path}")

    # Update Markdown document
    if paper_md_path.exists():
        md_content = paper_md_path.read_text(encoding="utf-8")
        md_content = re.sub(
            r"\|\s*MobileNetV2 \(CNN\)\s*\|\s*[^|]+\|\s*[^|]+\|\s*[^|]+\|\s*~8\.5 ms\s*\|",
            f"| MobileNetV2 (CNN) | {cnn_acc} | {cnn_prec} | {cnn_rec} | ~8.5 ms |",
            md_content
        )
        md_content = re.sub(
            r"\|\s*YOLO11n \(Detector\)\s*\|\s*[^|]+\|\s*[^|]+\|\s*[^|]+\|\s*~22\.0 ms\s*\|",
            f"| YOLO11n (Detector) | {yolo_map} | {yolo_prec} | {yolo_rec} | ~22.0 ms |",
            md_content
        )
        paper_md_path.write_text(md_content, encoding="utf-8")
        print(f"✅ Updated Markdown paper: {paper_md_path}")

    # Compile LaTeX if compiler is available
    if paper_tex_path.exists():
        try_compile_latex(paper_tex_path)

    return 0

if __name__ == "__main__":
    sys.exit(main())
