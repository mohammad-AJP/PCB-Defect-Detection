"""
One-time setup for the HRIPCB second-dataset experiments.

HRIPCB already has YOLO labels and a train/val/test split, but its
directory layout  HRIPCB/{split}/{images,labels}/  is the reverse of
what the rest of the pipeline expects:  data/hripcb/{images,labels}/{split}/

This script creates that expected layout using SYMLINKS (no disk copies),
then writes metadata.csv and dataset.yaml so every downstream script
(prepare_tiles, train_detector, infer_full, infer_tiles, evaluate, …)
works with --config configs/hripcb.yaml without any further changes.

Usage:
    python src/prepare_hripcb.py --config configs/hripcb.yaml
"""

import argparse
import sys
from pathlib import Path

import cv2
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_config


def parse_args():
    p = argparse.ArgumentParser(description="Set up HRIPCB for the pipeline")
    p.add_argument("--config", required=True)
    return p.parse_args()


def main():
    args  = parse_args()
    cfg   = load_config(args.config)

    repo_root    = Path(args.config).parent.parent
    source_dir   = (repo_root / cfg["dataset"]["source_dir"]).resolve()
    processed_dir = (repo_root / cfg["dataset"]["processed_dir"]).resolve()
    class_names  = cfg["class_names"]
    out_dir      = (repo_root / cfg["outputs"]["dir"]).resolve()

    if not source_dir.exists():
        print(f"ERROR: source_dir not found: {source_dir}")
        sys.exit(1)

    print(f"Source  : {source_dir}")
    print(f"Output  : {processed_dir}")
    print()

    splits = ["train", "val", "test"]
    metadata_rows = []

    for split in splits:
        src_img_dir = source_dir / split / "images"
        src_lbl_dir = source_dir / split / "labels"

        if not src_img_dir.exists():
            print(f"  WARNING: {src_img_dir} not found, skipping {split}")
            continue

        # create target dirs
        dst_img_dir = processed_dir / "images" / split
        dst_lbl_dir = processed_dir / "labels" / split
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

        images = sorted(src_img_dir.glob("*.jpg")) + sorted(src_img_dir.glob("*.png"))
        n_boxes_total = 0

        for i, img_src in enumerate(images):
            # symlink image
            img_dst = dst_img_dir / img_src.name
            if not img_dst.exists():
                img_dst.symlink_to(img_src)

            # symlink label
            lbl_src = src_lbl_dir / f"{img_src.stem}.txt"
            lbl_dst = dst_lbl_dir / f"{img_src.stem}.txt"
            if lbl_src.exists() and not lbl_dst.exists():
                lbl_dst.symlink_to(lbl_src)

            # read image dimensions
            img = cv2.imread(str(img_src))
            if img is None:
                print(f"  WARNING: cannot read {img_src.name}")
                continue
            h, w = img.shape[:2]

            # count annotations
            n_boxes = 0
            if lbl_src.exists():
                n_boxes = sum(1 for l in lbl_src.read_text().strip().splitlines()
                              if l.strip())
            n_boxes_total += n_boxes

            metadata_rows.append({
                "image_id":      i + len(metadata_rows),
                "file_name":     img_src.name,
                "width":         w,
                "height":        h,
                "split":         split,
                "n_annotations": n_boxes,
            })

        print(f"  {split:5s}: {len(images):4d} images  {n_boxes_total:5d} boxes  "
              f"→ symlinked into {dst_img_dir}")

    # write metadata.csv
    meta_df = pd.DataFrame(metadata_rows)
    meta_path = processed_dir / "metadata.csv"
    meta_df.to_csv(meta_path, index=False)
    print(f"\nMetadata → {meta_path}  ({len(meta_df)} rows)")

    # write dataset.yaml  (points to absolute paths for Ultralytics)
    ds_yaml = {
        "path":  str(processed_dir),
        "train": "images/train",
        "val":   "images/val",
        "test":  "images/test",
        "nc":    len(class_names),
        "names": class_names,
    }
    ds_yaml_path = processed_dir / "dataset.yaml"
    with open(ds_yaml_path, "w") as f:
        yaml.dump(ds_yaml, f, default_flow_style=False, sort_keys=False)
    print(f"dataset.yaml → {ds_yaml_path}")

    # create output dirs
    for sub in ["predictions", "tables", "figures"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    (out_dir / "runs").mkdir(parents=True, exist_ok=True)
    print(f"Output dirs → {out_dir}/")

    # quick sanity
    print()
    print("Split summary:")
    for split in splits:
        n_img = len(list((processed_dir / "images" / split).glob("*.jpg")))
        n_lbl = len(list((processed_dir / "labels" / split).glob("*.txt")))
        print(f"  {split:5s}: {n_img} images  {n_lbl} labels")

    print("\nSetup complete. Next:")
    print("  python src/prepare_tiles.py   --config configs/hripcb.yaml")
    print("  python src/train_detector.py  --config configs/hripcb.yaml \\")
    print("      --model yolo12n --data data/hripcb_tiles/dataset.yaml --name train_hripcb")


if __name__ == "__main__":
    main()
