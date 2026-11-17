"""
CNN Dataset Preparation Script — ForestGuard AI
Derives clean MobileNetV2 (fire / no_fire) splits from D-Fire YOLO dataset.
Zero data leakage: preserves original train/val/test split boundaries.
"""

import os
import shutil
import argparse
import json
from pathlib import Path
from PIL import Image

def compute_dhash(image_path: Path, hash_size: int = 8) -> str:
    """Computes difference hash (dHash) to detect duplicate images."""
    try:
        with Image.open(image_path) as img:
            img = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
            pixels = list(img.getdata())
            diff = []
            for row in range(hash_size):
                for col in range(hash_size):
                    left = pixels[row * (hash_size + 1) + col]
                    right = pixels[row * (hash_size + 1) + col + 1]
                    diff.append("1" if left > right else "0")
            return "".join(diff)
    except Exception:
        return ""

def process_split(
    split_name: str,
    images_dir: Path,
    labels_dir: Path,
    out_dir: Path,
    max_per_class: int,
    seen_hashes: set
) -> dict:
    fire_dir = out_dir / split_name / "fire"
    no_fire_dir = out_dir / split_name / "no_fire"
    fire_dir.mkdir(parents=True, exist_ok=True)
    no_fire_dir.mkdir(parents=True, exist_ok=True)

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    img_files = [f for f in images_dir.iterdir() if f.suffix.lower() in image_extensions]

    fire_images = []
    no_fire_images = []
    skipped_smoke_only = 0
    skipped_duplicates = 0

    for img_path in img_files:
        label_path = labels_dir / f"{img_path.stem}.txt"
        
        has_fire = False
        has_smoke = False

        if label_path.exists():
            with open(label_path, "r", encoding="utf-8") as lf:
                for line in lf:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    cls_id = int(parts[0])
                    if cls_id == 0:
                        has_fire = True
                    elif cls_id == 1:
                        has_smoke = True

        # Perceptual hash for duplicate detection
        h = compute_dhash(img_path)
        if h in seen_hashes:
            skipped_duplicates += 1
            continue
        if h:
            seen_hashes.add(h)

        if has_fire:
            fire_images.append(img_path)
        elif not has_fire and not has_smoke:
            no_fire_images.append(img_path)
        else:
            # Smoke only: neither pure Fire nor pure No-Fire
            skipped_smoke_only += 1

    # Balance classes up to max_per_class
    import random
    random.seed(42)
    random.shuffle(fire_images)
    random.shuffle(no_fire_images)

    selected_fire = fire_images[:max_per_class]
    selected_no_fire = no_fire_images[:max_per_class]

    # Copy files
    for p in selected_fire:
        shutil.copy2(p, fire_dir / p.name)
    for p in selected_no_fire:
        shutil.copy2(p, no_fire_dir / p.name)

    return {
        "fire_count": len(selected_fire),
        "no_fire_count": len(selected_no_fire),
        "skipped_smoke_only": skipped_smoke_only,
        "skipped_duplicates": skipped_duplicates
    }

def main():
    parser = argparse.ArgumentParser(description="Prepare CNN Fire/No-Fire dataset from D-Fire")
    parser.add_argument("--source_dir", type=str, required=True, help="Path to raw D-Fire dataset directory")
    parser.add_argument("--out_dir", type=str, default="datasets/cnn", help="Target output directory")
    parser.add_argument("--max_per_class", type=int, default=3000, help="Max images per class per split")
    args = parser.parse_args()

    source = Path(args.source_dir)
    out = Path(args.out_dir)
    seen_hashes = set()

    report = {"splits": {}}

    for split in ["train", "val", "test"]:
        img_dir = source / split / "images"
        lbl_dir = source / split / "labels"
        
        # Fallback if structure is without images/labels subfolders
        if not img_dir.exists():
            img_dir = source / "images" / split
            lbl_dir = source / "labels" / split

        if img_dir.exists():
            print(f"Processing split '{split}' from {img_dir}...")
            stats = process_split(split, img_dir, lbl_dir, out, args.max_per_class, seen_hashes)
            report["splits"][split] = stats
            print(f"  Split '{split}': {stats['fire_count']} fire, {stats['no_fire_count']} no_fire")

    # Save dataset report
    report_path = Path("results/metrics/dataset_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Dataset preparation complete! Report saved to {report_path}")

if __name__ == "__main__":
    main()
