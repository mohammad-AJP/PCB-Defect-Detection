"""
Dataset verification and visualization tool.

Run after prepare_dataset.py to confirm the processed dataset is correct.

Checks:
  1. Image / label file count per split
  2. Every label file is parseable and boxes are in [0,1]
  3. Class distribution per split
  4. Box size distribution (original pixels)
  5. Resolution-collapse preview: how many boxes shrink to <16 px² at 640

Saves:
  outputs/figures/verify/sample_{split}_{n}.jpg  -- images with GT boxes
  outputs/figures/verify/stats.png               -- class & size distribution charts

Usage:
    python src/verify_dataset.py --config configs/default.yaml
    python src/verify_dataset.py --config configs/default.yaml --samples 6 --split train
"""

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_config, xywhn_to_xyxy

# ── colour palette for up to 6 classes ───────────────────────────────────────
PALETTE = [
    (220,  20,  60),   # crimson
    ( 30, 144, 255),   # dodger blue
    ( 50, 205,  50),   # lime green
    (255, 165,   0),   # orange
    (148,   0, 211),   # violet
    (  0, 206, 209),   # dark turquoise
]


# ── helpers ───────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Verify processed PCB dataset")
    p.add_argument("--config", required=True)
    p.add_argument("--samples", type=int, default=4,
                   help="Number of sample images to visualize per split")
    p.add_argument("--split", default=None,
                   help="Visualize only this split (train/val/test). Default: all")
    return p.parse_args()


def load_dataset_yaml(processed_dir: Path) -> dict:
    yp = processed_dir / "dataset.yaml"
    if not yp.exists():
        raise FileNotFoundError(f"dataset.yaml not found: {yp}\nRun prepare_dataset.py first.")
    with open(yp) as f:
        return yaml.safe_load(f)


def read_label_file(lbl_path: Path):
    """Returns list of (class_id, cx, cy, w, h) tuples."""
    boxes = []
    for line in lbl_path.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls, cx, cy, w, h = int(parts[0]), float(parts[1]), float(parts[2]), \
                              float(parts[3]), float(parts[4])
        boxes.append((cls, cx, cy, w, h))
    return boxes


def draw_boxes_on_image(img_bgr, boxes, class_names, line_thickness=2):
    """Draw YOLO boxes (normalized) on a copy of the image."""
    h, w = img_bgr.shape[:2]
    vis = img_bgr.copy()
    for cls, cx, cy, bw, bh in boxes:
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
        color = PALETTE[cls % len(PALETTE)]
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, line_thickness)
        label = class_names[cls] if cls < len(class_names) else str(cls)
        cv2.putText(vis, label, (x1, max(0, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return vis


# ── per-split check ───────────────────────────────────────────────────────────

def check_split(processed_dir: Path, split: str, class_names: list,
                n_samples: int, out_dir: Path):
    img_dir = processed_dir / "images" / split
    lbl_dir = processed_dir / "labels" / split

    img_files = sorted(img_dir.glob("*.jpg"))
    lbl_files = sorted(lbl_dir.glob("*.txt"))

    print(f"\n  [{split}]  {len(img_files)} images  |  {len(lbl_files)} label files")

    if len(img_files) != len(lbl_files):
        print(f"  WARNING: image/label count mismatch!")

    # validate labels
    errors = 0
    class_counts = Counter()
    box_areas_px = []          # absolute area in original image
    box_areas_640 = []         # area after resizing to 640
    box_areas_1280 = []

    for img_f in img_files:
        stem = img_f.stem
        lbl_f = lbl_dir / f"{stem}.txt"
        if not lbl_f.exists():
            print(f"  WARNING: missing label for {img_f.name}")
            errors += 1
            continue

        img = cv2.imread(str(img_f))
        if img is None:
            print(f"  WARNING: cannot read image {img_f.name}")
            errors += 1
            continue

        h, w = img.shape[:2]
        boxes = read_label_file(lbl_f)

        for cls, cx, cy, bw, bh in boxes:
            # sanity: all values in [0,1]
            for v in [cx, cy, bw, bh]:
                if not (0.0 <= v <= 1.0):
                    print(f"  ERROR: out-of-range box in {lbl_f.name}: "
                          f"cls={cls} cx={cx:.4f} cy={cy:.4f} w={bw:.4f} h={bh:.4f}")
                    errors += 1
            class_counts[cls] += 1

            # absolute area
            abs_w = bw * w
            abs_h = bh * h
            box_areas_px.append(abs_w * abs_h)

            # area at 640
            scale_640 = min(640 / w, 640 / h)
            box_areas_640.append((abs_w * scale_640) * (abs_h * scale_640))

            # area at 1280
            scale_1280 = min(1280 / w, 1280 / h)
            box_areas_1280.append((abs_w * scale_1280) * (abs_h * scale_1280))

    # class distribution
    print(f"  Class distribution:")
    for cls_id in sorted(class_counts):
        name = class_names[cls_id] if cls_id < len(class_names) else f"cls{cls_id}"
        print(f"    {cls_id}: {name:20s} {class_counts[cls_id]}")

    # box area summary
    if box_areas_px:
        arr = np.array(box_areas_px)
        arr_640 = np.array(box_areas_640)
        tiny_640 = (arr_640 < 16).sum()
        tiny_pct = 100 * tiny_640 / len(arr_640)
        print(f"  Box area (original px²): min={arr.min():.0f}  "
              f"median={np.median(arr):.0f}  max={arr.max():.0f}")
        print(f"  Boxes < 16 px² after resize to 640: "
              f"{tiny_640}/{len(arr_640)} ({tiny_pct:.1f}%) ← resolution-collapse risk")

    if errors:
        print(f"  ERRORS found: {errors}")
    else:
        print(f"  All labels valid.")

    # ── visualize samples ─────────────────────────────────────────────────────
    rng = np.random.default_rng(0)
    sample_idxs = rng.choice(len(img_files), size=min(n_samples, len(img_files)), replace=False)

    out_dir.mkdir(parents=True, exist_ok=True)
    for k, idx in enumerate(sample_idxs):
        img_f = img_files[idx]
        img = cv2.imread(str(img_f))
        if img is None:
            continue
        stem = img_f.stem
        lbl_f = lbl_dir / f"{stem}.txt"
        boxes = read_label_file(lbl_f) if lbl_f.exists() else []

        vis = draw_boxes_on_image(img, boxes, class_names)

        # scale down for saving (long edge → 1200px)
        h, w = vis.shape[:2]
        scale = min(1.0, 1200 / max(h, w))
        if scale < 1.0:
            vis = cv2.resize(vis, (int(w * scale), int(h * scale)))

        out_path = out_dir / f"sample_{split}_{k:02d}_{stem}.jpg"
        cv2.imwrite(str(out_path), vis)

    print(f"  Saved {len(sample_idxs)} sample images → {out_dir}/")

    return class_counts, box_areas_640


# ── statistics plot ───────────────────────────────────────────────────────────

def plot_stats(all_counts: dict, all_areas_640: dict, class_names: list, out_dir: Path):
    splits = list(all_counts.keys())
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 1. Class distribution per split
    ax = axes[0]
    x = np.arange(len(class_names))
    width = 0.25
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    for i, split in enumerate(splits):
        counts = [all_counts[split].get(cls_id, 0) for cls_id in range(len(class_names))]
        ax.bar(x + i * width, counts, width, label=split, color=colors[i % len(colors)])
    ax.set_xticks(x + width)
    ax.set_xticklabels([n.replace("_", "\n") for n in class_names], fontsize=8)
    ax.set_title("Class distribution per split")
    ax.set_ylabel("Count")
    ax.legend()

    # 2. Box area distribution at 640 (histogram, all splits combined)
    ax = axes[1]
    all_areas = []
    for v in all_areas_640.values():
        all_areas.extend(v)
    all_areas = np.array(all_areas)
    bins = [0, 4, 16, 64, 256, 1024, 4096, all_areas.max() + 1]
    counts_hist, _ = np.histogram(all_areas, bins=bins)
    labels = ["0-4", "4-16", "16-64", "64-256", "256-1k", "1k-4k", ">4k"]
    colors_hist = ["#d62728" if i < 2 else "#ff7f0e" if i < 3 else "#2ca02c"
                   for i in range(len(labels))]
    ax.bar(labels, counts_hist, color=colors_hist)
    ax.set_title("Box area (px²) after resize to 640")
    ax.set_xlabel("Area bin (px²)")
    ax.set_ylabel("Count")
    # add note about collapse zones
    red_patch = mpatches.Patch(color="#d62728", label="High collapse risk (<16 px²)")
    orange_patch = mpatches.Patch(color="#ff7f0e", label="Moderate risk (16-64 px²)")
    green_patch = mpatches.Patch(color="#2ca02c", label="Safe (>64 px²)")
    ax.legend(handles=[red_patch, orange_patch, green_patch], fontsize=8)

    # 3. CDF of box area at 640
    ax = axes[2]
    sorted_areas = np.sort(all_areas)
    cdf = np.arange(1, len(sorted_areas) + 1) / len(sorted_areas)
    ax.plot(sorted_areas, cdf, lw=2)
    for thresh, label in [(4, "4 px²"), (16, "16 px²"), (64, "64 px²")]:
        frac = (sorted_areas <= thresh).mean()
        ax.axvline(thresh, color="red", linestyle="--", alpha=0.6)
        ax.text(thresh + 2, frac, f"{frac*100:.1f}%\n@ {label}",
                fontsize=8, color="red", va="bottom")
    ax.set_xscale("log")
    ax.set_xlabel("Box area at 640 (px²) [log scale]")
    ax.set_ylabel("Cumulative fraction")
    ax.set_title("CDF of box area (resolution collapse analysis)")
    ax.set_xlim(left=0.5)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = out_dir / "stats.png"
    plt.savefig(str(out_path), dpi=150)
    plt.close()
    print(f"\nStats chart saved → {out_path}")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg = load_config(args.config)

    config_dir = Path(args.config).parent.parent
    processed_dir = config_dir / cfg["dataset"]["processed_dir"]
    out_dir = config_dir / cfg["outputs"]["dir"] / "figures" / "verify"

    ds = load_dataset_yaml(processed_dir)
    class_names = ds["names"]
    print(f"Dataset: {processed_dir}")
    print(f"Classes : {class_names}")

    splits_to_check = [args.split] if args.split else ["train", "val", "test"]

    all_counts = {}
    all_areas_640 = {}

    for split in splits_to_check:
        counts, areas = check_split(
            processed_dir, split, class_names, args.samples, out_dir
        )
        all_counts[split] = counts
        all_areas_640[split] = areas

    plot_stats(all_counts, all_areas_640, class_names, out_dir)

    # print metadata summary if available
    meta_path = processed_dir / "metadata.csv"
    if meta_path.exists():
        meta = pd.read_csv(meta_path)
        print(f"\nMetadata summary ({len(meta)} images):")
        print(meta.groupby("split")[["n_annotations", "width", "height"]].describe().to_string())


if __name__ == "__main__":
    main()
