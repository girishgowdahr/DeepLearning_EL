"""
YOLO Dataset Preparation & Validation Script — ForestGuard AI
Prepares and validates D-Fire YOLO dataset, verifies bounding box coordinates,
and outputs data.yaml for Ultralytics YOLO training.
"""

import os
import argparse
from pathlib import Path

DATA_YAML_TEMPLATE = """# ForestGuard AI YOLO Dataset Configuration
path: {dataset_root}
train: train/images
val: val/images
test: test/images

# Classes
names:
  0: fire
  1: smoke
"""

def validate_labels(label_dir: Path) -> dict:
    total_boxes = 0
    class_counts = {0: 0, 1: 0}
    invalid_boxes = 0

    if not label_dir.exists():
        return {"total": 0, "classes": class_counts, "invalid": 0}

    for label_file in label_dir.glob("*.txt"):
        with open(label_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                try:
                    cls_id = int(parts[0])
                    x, y, w, h = [float(v) for v in parts[1:]]
                    # Verify normalized bounds
                    if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0:
                        total_boxes += 1
                        if cls_id in class_counts:
                            class_counts[cls_id] += 1
                    else:
                        invalid_boxes += 1
                except ValueError:
                    invalid_boxes += 1

    return {"total": total_boxes, "classes": class_counts, "invalid": invalid_boxes}

def main():
    parser = argparse.ArgumentParser(description="Prepare YOLO data.yaml and validate annotations")
    parser.add_argument("--dataset_root", type=str, default="datasets/yolo", help="Root directory of YOLO dataset")
    parser.add_argument("--out_yaml", type=str, default="datasets/yolo/data.yaml", help="Target path for data.yaml")
    args = parser.parse_args()

    root = Path(args.dataset_root).resolve()
    out_yaml_path = Path(args.out_yaml)
    out_yaml_path.parent.mkdir(parents=True, exist_ok=True)

    # Validate splits
    for split in ["train", "val", "test"]:
        lbl_dir = root / split / "labels"
        stats = validate_labels(lbl_dir)
        print(f"Split '{split}': {stats['total']} valid boxes (Fire: {stats['classes'][0]}, Smoke: {stats['classes'][1]}), Invalid: {stats['invalid']}")

    # Write data.yaml with forward slashes
    content = DATA_YAML_TEMPLATE.format(dataset_root=str(root).replace("\\", "/"))
    with open(out_yaml_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"data.yaml generated successfully at {out_yaml_path}")

if __name__ == "__main__":
    main()
