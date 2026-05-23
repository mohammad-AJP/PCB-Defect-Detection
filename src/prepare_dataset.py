"""
Step 1: Dataset preprocessing.

Reads COCO JSON annotations, splits into train/val/test, copies images,
converts annotations to YOLO format, creates dataset.yaml and metadata.csv.

COCO category_id 0 is the supercategory (skipped).
COCO category_ids 1-6 are mapped to YOLO class ids 0-5.

Output:
    data/processed/images/{train,val,test}/
    data/processed/labels/{train,val,test}/
    data/processed/dataset.yaml
    data/processed/metadata.csv

Usage:
    python src/prepare_dataset.py --config configs/default.yaml
    python src/prepare_dataset.py --config configs/default.yaml --raw-dir /path/to/dataset
"""

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_config


# ── helpers ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Prepare PCB dataset")
    p.add_argument("--config", required=True)
    p.add_argument("--raw-dir", default=None,
                   help="Override config raw_dir (path to PCB_Defect folder)")
    return p.parse_args()


def load_coco(json_path: Path) -> dict:
    if not json_path.exists():
        raise FileNotFoundError(f"COCO JSON not found: {json_path}")
    with open(json_path) as f:
        return json.load(f)


def build_category_map(categories: list) -> dict:
    """
    Returns {coco_id -> yolo_class_id} skipping the supercategory (id=0).
    The remaining categories are sorted by id and remapped to 0-based indices.
    """
    real_cats = sorted(
        [c for c in categories if c["id"] != 0],
        key=lambda c: c["id"]
    )
    return {c["id"]: i for i, c in enumerate(real_cats)}, real_cats


def coco_bbox_to_yolo(bbox, img_w, img_h):
    """Convert COCO [x,y,w,h] to YOLO [cx,cy,w,h] normalized."""
    x, y, w, h = bbox
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    wn = w / img_w
    hn = h / img_h
    # clamp to [0, 1]
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    wn = max(0.0, min(1.0, wn))
    hn = max(0.0, min(1.0, hn))
    return cx, cy, wn, hn


def split_images(image_ids: list, train_r, val_r, seed: int):
    """Deterministic stratified split by ratio."""
    rng = random.Random(seed)
    ids = list(image_ids)
    rng.shuffle(ids)
    n = len(ids)
    n_train = round(n * train_r)
    n_val = round(n * val_r)
    train = ids[:n_train]
    val = ids[n_train:n_train + n_val]
    test = ids[n_train + n_val:]
    return train, val, test


# ── main pipeline ─────────────────────────────────────────────────────────────

def process_coco(raw_dir: Path, processed_dir: Path, coco_json_rel: str,
                 train_r, val_r, test_r, seed: int):
    coco = load_coco(raw_dir / coco_json_rel)
    cat_map, real_cats = build_category_map(coco["categories"])
    class_names = [c["name"] for c in real_cats]

    # index images and annotations
    img_info = {img["id"]: img for img in coco["images"]}
    ann_by_img = {img_id: [] for img_id in img_info}
    skipped_anns = 0
    for ann in coco["annotations"]:
        if ann["category_id"] not in cat_map:
            skipped_anns += 1
            continue
        ann_by_img[ann["image_id"]].append(ann)
    if skipped_anns:
        print(f"  Skipped {skipped_anns} annotations with unmapped category_id")

    all_ids = list(img_info.keys())
    train_ids, val_ids, test_ids = split_images(all_ids, train_r, val_r, seed)
    split_map = (
        {i: "train" for i in train_ids}
        | {i: "val" for i in val_ids}
        | {i: "test" for i in test_ids}
    )

    print(f"  Split: {len(train_ids)} train / {len(val_ids)} val / {len(test_ids)} test")

    metadata_rows = []

    for img_id, split in split_map.items():
        info = img_info[img_id]
        src_img = raw_dir / "images" / info["file_name"]
        if not src_img.exists():
            print(f"  WARNING: image not found, skipping: {src_img}")
            continue

        img_w, img_h = info["width"], info["height"]

        # copy image
        dst_img_dir = processed_dir / "images" / split
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_img = dst_img_dir / info["file_name"]
        if not dst_img.exists():
            shutil.copy2(src_img, dst_img)

        # write YOLO label file
        dst_lbl_dir = processed_dir / "labels" / split
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(info["file_name"]).stem
        dst_lbl = dst_lbl_dir / f"{stem}.txt"

        anns = ann_by_img.get(img_id, [])
        lines = []
        for ann in anns:
            cls = cat_map[ann["category_id"]]
            cx, cy, wn, hn = coco_bbox_to_yolo(ann["bbox"], img_w, img_h)
            if wn <= 0 or hn <= 0:
                continue
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {wn:.6f} {hn:.6f}")

        dst_lbl.write_text("\n".join(lines) + ("\n" if lines else ""))

        metadata_rows.append({
            "image_id": img_id,
            "file_name": info["file_name"],
            "width": img_w,
            "height": img_h,
            "split": split,
            "n_annotations": len(anns),
        })

    return class_names, metadata_rows


def write_dataset_yaml(processed_dir: Path, class_names: list):
    yaml_path = processed_dir / "dataset.yaml"
    data = {
        "path": str(processed_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(class_names),
        "names": class_names,
    }
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    return yaml_path


def write_metadata(processed_dir: Path, rows: list):
    df = pd.DataFrame(rows)
    path = processed_dir / "metadata.csv"
    df.to_csv(path, index=False)
    return path


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # resolve paths relative to config file location
    config_dir = Path(args.config).parent.parent  # repo root
    raw_dir = Path(args.raw_dir) if args.raw_dir else config_dir / cfg["dataset"]["raw_dir"]
    processed_dir = config_dir / cfg["dataset"]["processed_dir"]

    if not raw_dir.exists():
        print(f"ERROR: raw_dir not found: {raw_dir}")
        sys.exit(1)

    fmt = cfg["dataset"]["format"].lower()
    if fmt != "coco":
        print(f"ERROR: only 'coco' format is supported in this dataset. Got: {fmt}")
        sys.exit(1)

    coco_json_rel = cfg["dataset"].get("coco_json", "annotation/_annotations.coco.json")
    seed = cfg["dataset"]["seed"]
    train_r = cfg["dataset"]["train_ratio"]
    val_r = cfg["dataset"]["val_ratio"]
    test_r = cfg["dataset"]["test_ratio"]

    if abs(train_r + val_r + test_r - 1.0) > 1e-6:
        print(f"ERROR: train+val+test ratios must sum to 1.0, got {train_r+val_r+test_r}")
        sys.exit(1)

    processed_dir.mkdir(parents=True, exist_ok=True)

    print(f"Raw dataset : {raw_dir}")
    print(f"Output dir  : {processed_dir}")
    print(f"Processing COCO annotations...")

    class_names, metadata_rows = process_coco(
        raw_dir, processed_dir, coco_json_rel,
        train_r, val_r, test_r, seed
    )

    yaml_path = write_dataset_yaml(processed_dir, class_names)
    meta_path = write_metadata(processed_dir, metadata_rows)

    print(f"\nDone.")
    print(f"  dataset.yaml : {yaml_path}")
    print(f"  metadata.csv : {meta_path}")
    print(f"  Classes      : {class_names}")

    # quick sanity check
    for split in ("train", "val", "test"):
        n_imgs = len(list((processed_dir / "images" / split).glob("*.jpg")))
        n_lbls = len(list((processed_dir / "labels" / split).glob("*.txt")))
        print(f"  {split:5s}: {n_imgs} images, {n_lbls} label files")


if __name__ == "__main__":
    main()
