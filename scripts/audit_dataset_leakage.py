#!/usr/bin/env python3
"""
ForestGuard AI — Dataset Leakage & Annotation Integrity Audit Tool
==================================================================
Performs reproducible dataset audits:
1. Exact duplicate detection using SHA-256 cryptographic hashing.
2. Perceptual hashing (dHash/aHash) for visually identical/near-duplicate images.
3. Cross-split contamination analysis across train, validation, and test splits.
4. Image-label pairing, boundary checks, and annotation integrity auditing.
5. Split-wise class distribution and representation balance analysis.
6. Graceful fallback reporting when full dataset is hosted remotely (e.g. Kaggle).

Usage:
  python scripts/audit_dataset_leakage.py [--dataset-dir PATH] [--threshold INT]
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


def compute_sha256(path: Path) -> str:
    """Computes SHA-256 digest of a file in streaming chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_dhash(image_path: Path, hash_size: int = 8) -> Optional[int]:
    """
    Computes difference hash (dHash) using Pillow.
    Returns 64-bit integer representing gradient direction across 9x8 grayscale image.
    """
    if not HAS_PIL:
        return None
    try:
        with Image.open(image_path) as img:
            # Convert to grayscale and resize to (width + 1, height)
            gray = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
            pixels = list(gray.getdata())

            diff = []
            for row in range(hash_size):
                row_offset = row * (hash_size + 1)
                for col in range(hash_size):
                    pixel_left = pixels[row_offset + col]
                    pixel_right = pixels[row_offset + col + 1]
                    diff.append(pixel_left > pixel_right)

            # Pack booleans into 64-bit integer
            decimal_value = 0
            for bit in diff:
                decimal_value = (decimal_value << 1) | (1 if bit else 0)
            return decimal_value
    except Exception:
        return None


def hamming_distance(hash1: int, hash2: int) -> int:
    """Calculates Hamming distance (bit count of XOR) between two integer hashes."""
    return bin(hash1 ^ hash2).count("1")


class DatasetLeakageAuditor:
    def __init__(self, dataset_dir: Optional[Path] = None, hamming_threshold: int = 5):
        self.dataset_dir = dataset_dir
        self.hamming_threshold = hamming_threshold
        self.splits = ["train", "val", "test"]
        self.results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "status": "UNKNOWN",
            "dataset_directory": str(dataset_dir) if dataset_dir else None,
            "hamming_threshold": hamming_threshold,
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
            "verdict": ""
        }

    def locate_splits(self) -> Dict[str, Tuple[Optional[Path], Optional[Path]]]:
        """Resolves (images_dir, labels_dir) for each split."""
        split_paths = {}
        if not self.dataset_dir or not self.dataset_dir.exists():
            return split_paths

        for s in self.splits:
            img_cand = None
            lbl_cand = None

            # Candidate image dirs
            cands = [
                self.dataset_dir / s / "images",
                self.dataset_dir / s,
                self.dataset_dir / f"{s}_images",
            ]
            if s == "val":
                cands.extend([
                    self.dataset_dir / "valid" / "images",
                    self.dataset_dir / "valid"
                ])

            for cand in cands:
                if cand.exists() and any(cand.glob("*.jp*g")) or any(cand.glob("*.png")):
                    img_cand = cand
                    break

            # Candidate label dirs
            lbl_cands = [
                self.dataset_dir / s / "labels",
                self.dataset_dir / "labels" / s,
                self.dataset_dir / f"{s}_labels",
            ]
            if s == "val":
                lbl_cands.extend([
                    self.dataset_dir / "valid" / "labels",
                    self.dataset_dir / "labels" / "valid"
                ])

            for cand in lbl_cands:
                if cand.exists():
                    lbl_cand = cand
                    break

            split_paths[s] = (img_cand, lbl_cand)
        return split_paths

    def run_audit(self) -> Dict:
        """Executes full leakage and integrity audit or reports pending status."""
        split_paths = self.locate_splits()
        has_data = any(img_dir is not None for img_dir, _ in split_paths.values())

        if not has_data:
            self.results["status"] = "PENDING_OFFLINE_DATASET"
            self.results["verdict"] = (
                "AUDIT PENDING: Full raw image dataset is not stored in local Git clone (expected per .gitignore). "
                "Training dataset resides on Kaggle (sayedgamal99/smoke-fire-detection-yolo, 20,000+ images). "
                "No zero-leakage claim is made without direct execution of this script against full raw images."
            )
            return self.results

        self.results["status"] = "COMPLETED"
        print(f"🌲 Commencing Dataset Leakage & Integrity Audit on: {self.dataset_dir}")
        print(f"   Perceptual Hash Threshold: Hamming distance <= {self.hamming_threshold}")

        split_images: Dict[str, List[Path]] = {}
        split_hashes: Dict[str, Dict[str, Path]] = {}     # sha256 -> Path
        split_phashes: Dict[str, Dict[int, Path]] = {}    # dhash -> Path

        # Phase 1: Ingest images & Compute cryptographic and perceptual hashes
        for s in self.splits:
            img_dir, lbl_dir = split_paths[s]
            images = []
            if img_dir and img_dir.exists():
                images = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.jpeg")) + list(img_dir.glob("*.png")))
            split_images[s] = images
            split_hashes[s] = {}
            split_phashes[s] = {}
            self.results["images_audited"][s] = len(images)
            print(f"  Split '{s}': Found {len(images)} images.")

            for img_path in images:
                # SHA-256
                sha = compute_sha256(img_path)
                if sha in split_hashes[s]:
                    self.results["exact_duplicates"]["intra_split"].append({
                        "split": s,
                        "file1": str(split_hashes[s][sha].name),
                        "file2": str(img_path.name),
                        "sha256": sha
                    })
                else:
                    split_hashes[s][sha] = img_path

                # Perceptual dHash
                if HAS_PIL:
                    dh = compute_dhash(img_path)
                    if dh is not None:
                        split_phashes[s][dh] = img_path

        # Phase 2: Detect Cross-Split Cryptographic Leakage (Train vs Val, Train vs Test, Val vs Test)
        split_names = list(split_hashes.keys())
        for i in range(len(split_names)):
            for j in range(i + 1, len(split_names)):
                s1, s2 = split_names[i], split_names[j]
                common_hashes = set(split_hashes[s1].keys()) & set(split_hashes[s2].keys())
                for c_hash in common_hashes:
                    self.results["exact_duplicates"]["inter_split_leakage"].append({
                        "split_a": s1,
                        "file_a": str(split_hashes[s1][c_hash].name),
                        "split_b": s2,
                        "file_b": str(split_hashes[s2][c_hash].name),
                        "sha256": c_hash
                    })

        # Phase 3: Detect Cross-Split Perceptual Near-Duplicates (Train vs Val/Test)
        if HAS_PIL and "train" in split_phashes:
            train_phashes = split_phashes["train"]
            for test_split in ["val", "test"]:
                if test_split not in split_phashes:
                    continue
                for t_hash, t_path in split_phashes[test_split].items():
                    for tr_hash, tr_path in train_phashes.items():
                        dist = hamming_distance(t_hash, tr_hash)
                        if dist <= self.hamming_threshold:
                            self.results["perceptual_near_duplicates"]["inter_split_leakage"].append({
                                "train_file": str(tr_path.name),
                                "eval_split": test_split,
                                "eval_file": str(t_path.name),
                                "hamming_distance": dist
                            })

        # Phase 4: Annotation Integrity & Class Distribution
        for s in self.splits:
            img_dir, lbl_dir = split_paths[s]
            class_counts = {"fire_boxes": 0, "smoke_boxes": 0, "total_boxes": 0, "negative_images": 0, "positive_images": 0}

            for img_path in split_images[s]:
                lbl_file = (lbl_dir / f"{img_path.stem}.txt") if lbl_dir else None
                if not lbl_file or not lbl_file.exists():
                    self.results["annotation_integrity"]["missing_labels"].append(f"{s}/{img_path.name}")
                    continue

                try:
                    lines = lbl_file.read_text(encoding="utf-8", errors="ignore").strip().splitlines()
                except Exception:
                    self.results["annotation_integrity"]["corrupted_labels"].append(f"{s}/{lbl_file.name}")
                    continue

                if not lines or (len(lines) == 1 and not lines[0].strip()):
                    class_counts["negative_images"] += 1
                    continue

                has_pos = False
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    try:
                        cid = int(parts[0])
                        coords = [float(x) for x in parts[1:5]]
                    except ValueError:
                        self.results["annotation_integrity"]["corrupted_labels"].append(f"{s}/{lbl_file.name}")
                        continue

                    if cid not in [0, 1]:
                        self.results["annotation_integrity"]["invalid_classes"].append({
                            "file": f"{s}/{lbl_file.name}",
                            "class_id": cid
                        })

                    # Bounding box coordinates sanity [0, 1]
                    x_c, y_c, w, h = coords
                    if not (0.0 <= x_c <= 1.0 and 0.0 <= y_c <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                        self.results["annotation_integrity"]["out_of_bounds_coords"].append({
                            "file": f"{s}/{lbl_file.name}",
                            "coords": coords
                        })

                    if cid == 0:
                        class_counts["smoke_boxes"] += 1
                        has_pos = True
                    elif cid == 1:
                        class_counts["fire_boxes"] += 1
                        has_pos = True
                    class_counts["total_boxes"] += 1

                if has_pos:
                    class_counts["positive_images"] += 1
                else:
                    class_counts["negative_images"] += 1

            self.results["class_distribution"][s] = class_counts

        # Final verdict formulation
        leak_count = len(self.results["exact_duplicates"]["inter_split_leakage"])
        near_leak_count = len(self.results["perceptual_near_duplicates"]["inter_split_leakage"])

        if leak_count == 0 and near_leak_count == 0:
            self.results["verdict"] = "ZERO_LEAKAGE_CONFIRMED: No cryptographic or perceptual duplicates detected across splits."
        else:
            self.results["verdict"] = (
                f"POTENTIAL_LEAKAGE_DETECTED: Found {leak_count} exact SHA-256 cross-split duplicates and "
                f"{near_leak_count} perceptual near-duplicates (Hamming dist <= {self.hamming_threshold})."
            )

        return self.results


def format_markdown_report(results: Dict) -> str:
    """Formats audit dictionary as clean, academic Markdown report."""
    md = []
    md.append("# ForestGuard AI — Dataset Leakage & Annotation Integrity Audit Report")
    md.append(f"**Audit Timestamp**: {results['timestamp']}  ")
    md.append(f"**Status**: `{results['status']}`  ")
    md.append(f"**Target Directory**: `{results['dataset_directory']}`  ")
    md.append(f"**Perceptual Hash Threshold**: Hamming distance $\\le {results['hamming_threshold']}$ bits  \n")
    md.append("---")

    md.append("## 1. Executive Summary & Verdict")
    md.append(f"> {results['verdict']}\n")

    md.append("## 2. Split Size & Image Inventory")
    if results["images_audited"]:
        md.append("| Split | Audited Image Count |")
        md.append("| :--- | :--- |")
        for s, count in results["images_audited"].items():
            md.append(f"| **{s}** | {count:,} images |")
    else:
        md.append("*No local image files audited. Dataset is hosted on Kaggle or pending download.*")
    md.append("")

    md.append("## 3. Cryptographic Duplicate Analysis (SHA-256)")
    exact_inter = results["exact_duplicates"]["inter_split_leakage"]
    exact_intra = results["exact_duplicates"]["intra_split"]
    md.append(f"- **Inter-Split Leakage (Cross-Contamination)**: **{len(exact_inter)}** instances found.")
    md.append(f"- **Intra-Split Duplicates (Redundancy within split)**: **{len(exact_intra)}** instances found.")
    if exact_inter:
        md.append("\n### Detected Inter-Split Cryptographic Leaks:")
        for item in exact_inter[:10]:
            md.append(f"- `{item['split_a']}/{item['file_a']}` $\\leftrightarrow$ `{item['split_b']}/{item['file_b']}` (`{item['sha256'][:16]}...`)")
        if len(exact_inter) > 10:
            md.append(f"- *...and {len(exact_inter) - 10} more.*")
    md.append("")

    md.append("## 4. Perceptual Hash Analysis (Visual Near-Duplicates)")
    near_leaks = results["perceptual_near_duplicates"]["inter_split_leakage"]
    md.append(f"- **Cross-Split Perceptual Near-Duplicates**: **{len(near_leaks)}** pairs identified.")
    if near_leaks:
        md.append("\n### Sample Perceptual Near-Duplicates:")
        for item in near_leaks[:10]:
            md.append(f"- Train `{item['train_file']}` $\\leftrightarrow$ {item['eval_split'].capitalize()} `{item['eval_file']}` (Hamming Distance: {item['hamming_distance']})")
        if len(near_leaks) > 10:
            md.append(f"- *...and {len(near_leaks) - 10} more.*")
    md.append("")

    md.append("## 5. Annotation Integrity & Bounding Box Validity")
    annot = results["annotation_integrity"]
    md.append(f"- **Missing Labels**: {len(annot['missing_labels'])}")
    md.append(f"- **Corrupted / Unparseable Label Files**: {len(annot['corrupted_labels'])}")
    md.append(f"- **Out-of-Bounds Bounding Box Coordinates**: {len(annot['out_of_bounds_coords'])}")
    md.append(f"- **Invalid Class IDs**: {len(annot['invalid_classes'])}")
    md.append("")

    md.append("## 6. Split-Wise Class Distribution")
    dist = results["class_distribution"]
    if dist:
        md.append("| Split | Positive Imgs | Negative Imgs | Fire Boxes | Smoke Boxes | Total Boxes |")
        md.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
        for s, counts in dist.items():
            md.append(f"| **{s}** | {counts.get('positive_images', 0):,} | {counts.get('negative_images', 0):,} | {counts.get('fire_boxes', 0):,} | {counts.get('smoke_boxes', 0):,} | {counts.get('total_boxes', 0):,} |")
    else:
        md.append("*Distribution analysis pending full dataset access.*")
    md.append("")

    md.append("## 7. Instructions for Offline or Kaggle Execution")
    md.append("To execute this audit against the full dataset on Kaggle or a local machine:")
    md.append("```bash")
    md.append("# On Kaggle:")
    md.append("python scripts/audit_dataset_leakage.py --dataset-dir /kaggle/input/smoke-fire-detection-yolo")
    md.append("")
    md.append("# On Local Machine (after downloading raw dataset):")
    md.append("python scripts/audit_dataset_leakage.py --dataset-dir path/to/extracted/dataset")
    md.append("```\n")

    return "\n".join(md)


def main():
    parser = argparse.ArgumentParser(description="ForestGuard AI Dataset Leakage & Integrity Auditor")
    parser.add_argument("--dataset-dir", type=str, default=None, help="Root path of image dataset (contains train, val, test)")
    parser.add_argument("--threshold", type=int, default=5, help="Hamming distance threshold for near-duplicates (default: 5)")
    parser.add_argument("--json-out", type=str, default="results/dataset_leakage_audit_report.json", help="Path to write JSON results")
    parser.add_argument("--md-out", type=str, default="docs/dataset_leakage_audit.md", help="Path to write Markdown report")
    args = parser.parse_args()

    ds_dir = Path(args.dataset_dir) if args.dataset_dir else None

    # Auto-detect if standard directories exist
    if ds_dir is None:
        for cand in [Path("datasets"), Path("forestguard_dataset"), Path("data")]:
            if cand.exists() and (cand / "train").exists():
                ds_dir = cand
                break

    auditor = DatasetLeakageAuditor(dataset_dir=ds_dir, hamming_threshold=args.threshold)
    results = auditor.run_audit()

    # Save JSON report
    os.makedirs(os.path.dirname(args.json_out), exist_ok=True)
    with open(args.json_out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"📝 Saved audit JSON: {args.json_out}")

    # Save Markdown report
    os.makedirs(os.path.dirname(args.md_out), exist_ok=True)
    md_content = format_markdown_report(results)
    with open(args.md_out, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"📄 Saved audit Markdown report: {args.md_out}")

    print("\n" + "=" * 65)
    print(f"AUDIT STATUS: {results['status']}")
    print(f"VERDICT:      {results['verdict']}")
    print("=" * 65)


if __name__ == "__main__":
    main()
