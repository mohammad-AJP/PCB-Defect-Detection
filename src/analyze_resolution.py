"""
Step 8: Resolution-collapse analysis.

For every GT box in the test set, computes its apparent area after resizing
the full image to 640 and 1280, then links each GT to whether it was detected
by each inference method.  Produces:

  outputs/tables/size_recall.csv     — recall by area bin per method (already
                                        written by evaluate.py; this script adds
                                        richer per-box detail)
  outputs/figures/resolution/
    collapse_histogram.png           — distribution of GT areas at different scales
    recall_by_area.png               — recall curves vs resized area per method
    collapse_examples.png            — montage of defects that become tiny at 640

Usage:
    python src/analyze_resolution.py --config configs/default.yaml
"""

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_config, xywhn_to_xyxy, compute_iou


AREA_BINS   = [(0, 4), (4, 16), (16, 64), (64, float("inf"))]
AREA_LABELS = ["0–4", "4–16", "16–64", ">64"]
IOU_THRESH  = 0.5

METHOD_ORDER = [
    ("full_640",        "Full-640",               "#d62728"),
    ("full_1280",       "Full-1280",               "#ff7f0e"),
    ("tile_640",        "Tile-640 + NMS",          "#2ca02c"),
    ("tile_640_ov128",  "Tile-640 + Ov + NMS",     "#1f77b4"),
    ("tatm_640_ov128",  "Tile-640 + Ov + TA-TM",   "#9467bd"),
]


# ── helpers ───────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Resolution collapse analysis")
    p.add_argument("--config", required=True)
    p.add_argument("--split",  default="test")
    return p.parse_args()


def load_gt(processed_dir: Path, split: str, meta: pd.DataFrame) -> list[dict]:
    lbl_dir = processed_dir / "labels" / split
    boxes = []
    for _, row in meta[meta["split"] == split].iterrows():
        stem   = Path(row["file_name"]).stem
        lbl_f  = lbl_dir / f"{stem}.txt"
        img_w, img_h = int(row["width"]), int(row["height"])
        s640  = min(640  / img_w, 640  / img_h)
        s1280 = min(1280 / img_w, 1280 / img_h)
        if not lbl_f.exists():
            continue
        for line in lbl_f.read_text().strip().splitlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls_id = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:])
            x1, y1, x2, y2 = xywhn_to_xyxy(cx, cy, bw, bh, img_w, img_h)
            w_px, h_px = x2 - x1, y2 - y1
            boxes.append({
                "image_id": int(row["image_id"]),
                "file_name": row["file_name"],
                "img_w": img_w, "img_h": img_h,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "class_id": cls_id,
                "area_native":  w_px * h_px,
                "area_at_640":  (w_px * s640)  * (h_px * s640),
                "area_at_1280": (w_px * s1280) * (h_px * s1280),
            })
    return boxes


def load_pred(pred_path: Path) -> pd.DataFrame:
    df = pd.read_csv(pred_path).dropna(subset=["x1"])
    if "score_adj" in df.columns:
        df["conf"] = df["score_adj"]
    else:
        df["conf"] = df["score"]
    df["class_id"] = df["class_id"].astype(int)
    return df


def detected_gt_indices(gt_boxes: list[dict], pred_df: pd.DataFrame) -> set[int]:
    """Return set of GT indices matched by at least one prediction (IoU≥0.5)."""
    from collections import defaultdict
    gt_by_img_cls = defaultdict(list)
    for i, g in enumerate(gt_boxes):
        gt_by_img_cls[(g["image_id"], g["class_id"])].append((i, g))

    matched = set()
    for _, row in pred_df.sort_values("conf", ascending=False).iterrows():
        key = (int(row["image_id"]), int(row["class_id"]))
        for gi, g in gt_by_img_cls.get(key, []):
            if gi in matched:
                continue
            if compute_iou([row.x1, row.y1, row.x2, row.y2],
                           [g["x1"], g["y1"], g["x2"], g["y2"]]) >= IOU_THRESH:
                matched.add(gi)
                break
    return matched


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_collapse_histogram(gt_boxes: list[dict], out_path: Path):
    """Histogram of GT box areas at native, 640, and 1280 scales."""
    native  = np.array([g["area_native"]  for g in gt_boxes])
    at640   = np.array([g["area_at_640"]  for g in gt_boxes])
    at1280  = np.array([g["area_at_1280"] for g in gt_boxes])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # left: histogram of area at 640
    ax = axes[0]
    bins = [0, 4, 16, 64, 256, 1024, at640.max() + 1]
    counts, _ = np.histogram(at640, bins=bins)
    labels = ["<4", "4–16", "16–64", "64–256", "256–1k", ">1k"]
    colors = ["#d62728", "#d62728", "#ff7f0e", "#2ca02c", "#2ca02c", "#2ca02c"]
    ax.bar(labels, counts, color=colors, edgecolor="white")
    ax.set_title("GT box area distribution at 640px input")
    ax.set_xlabel("Area (px²)")
    ax.set_ylabel("Count")
    red   = mpatches.Patch(color="#d62728", label="High collapse risk (<16 px²)")
    orng  = mpatches.Patch(color="#ff7f0e", label="Moderate (16–64 px²)")
    green = mpatches.Patch(color="#2ca02c", label="Safe (>64 px²)")
    ax.legend(handles=[red, orng, green], fontsize=8)

    # right: CDF at all three scales
    ax = axes[1]
    for arr, label, color, ls in [
        (native, "Native resolution", "#555555", "--"),
        (at640,  "After resize to 640",  "#d62728", "-"),
        (at1280, "After resize to 1280", "#ff7f0e", "-"),
    ]:
        sorted_arr = np.sort(arr)
        cdf = np.arange(1, len(sorted_arr) + 1) / len(sorted_arr)
        ax.plot(sorted_arr, cdf, label=label, color=color, linestyle=ls, lw=2)
    for thresh, label in [(4, "4 px²"), (16, "16 px²"), (64, "64 px²")]:
        frac = (at640 <= thresh).mean()
        ax.axvline(thresh, color="grey", linestyle=":", alpha=0.7)
        ax.text(thresh * 1.05, frac + 0.02, f"{frac*100:.1f}%\n@{label}",
                fontsize=7, color="grey")
    ax.set_xscale("log")
    ax.set_xlabel("Box area (px²) [log scale]")
    ax.set_ylabel("Cumulative fraction")
    ax.set_title("CDF of GT box area — resolution collapse")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


def plot_recall_by_area(gt_boxes: list[dict],
                        detections: dict[str, set[int]],
                        out_path: Path):
    """Bar chart of recall per area bin per method."""
    bin_recalls = {}
    for mkey, _, _ in METHOD_ORDER:
        if mkey not in detections:
            continue
        matched = detections[mkey]
        recalls = []
        for (lo, hi) in AREA_BINS:
            idx = [i for i, g in enumerate(gt_boxes)
                   if lo <= g["area_at_640"] < hi]
            recalls.append(sum(1 for i in idx if i in matched) / len(idx)
                           if idx else 0.0)
        bin_recalls[mkey] = recalls

    methods = [k for k, _, _ in METHOD_ORDER if k in bin_recalls]
    n_methods = len(methods)
    x = np.arange(len(AREA_BINS))
    width = 0.15
    offsets = np.linspace(-(n_methods - 1) / 2, (n_methods - 1) / 2, n_methods) * width

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, mkey in enumerate(methods):
        label = next(l for k, l, _ in METHOD_ORDER if k == mkey)
        color = next(c for k, _, c in METHOD_ORDER if k == mkey)
        ax.bar(x + offsets[i], bin_recalls[mkey], width,
               label=label, color=color, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{l} px²" for l in AREA_LABELS])
    ax.set_xlabel("GT box area after resize to 640px input")
    ax.set_ylabel("Recall")
    ax.set_title("Recall by defect size (resolution collapse sensitivity)")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


def plot_collapse_examples(gt_boxes: list[dict], processed_dir: Path,
                           out_path: Path, n_examples: int = 6):
    """
    Montage of defect crops showing native resolution vs what the model
    sees at 640px input scale.
    """
    # pick boxes smallest at 640 that have a real native image
    candidates = sorted(
        [g for g in gt_boxes if g["area_at_640"] < 100],
        key=lambda g: g["area_at_640"]
    )[:n_examples]

    if not candidates:
        print("  No collapse examples to show (no boxes < 100px² at 640).")
        return

    fig, axes = plt.subplots(2, len(candidates), figsize=(3 * len(candidates), 6))
    if len(candidates) == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for col, g in enumerate(candidates):
        img_path = processed_dir / "images" / "test" / g["file_name"]
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_h, img_w = img.shape[:2]

        # native crop with padding
        x1, y1, x2, y2 = int(g["x1"]), int(g["y1"]), int(g["x2"]), int(g["y2"])
        pad = max(20, int((x2 - x1) * 0.5), int((y2 - y1) * 0.5))
        cx1 = max(0, x1 - pad); cy1 = max(0, y1 - pad)
        cx2 = min(img_w, x2 + pad); cy2 = min(img_h, y2 + pad)
        crop_native = img_rgb[cy1:cy2, cx1:cx2]

        # resized-to-640 version
        scale = min(640 / img_w, 640 / img_h)
        img_small = cv2.resize(img_rgb,
                               (int(img_w * scale), int(img_h * scale)),
                               interpolation=cv2.INTER_AREA)
        sx1 = int(x1 * scale); sy1 = int(y1 * scale)
        sx2 = int(x2 * scale); sy2 = int(y2 * scale)
        pad_s = max(8, (sx2 - sx1) * 2, (sy2 - sy1) * 2)
        scx1 = max(0, sx1 - pad_s); scy1 = max(0, sy1 - pad_s)
        scx2 = min(img_small.shape[1], sx2 + pad_s)
        scy2 = min(img_small.shape[0], sy2 + pad_s)
        crop_small = img_small[scy1:scy2, scx1:scx2]

        # draw box on native crop
        bx1 = x1 - cx1; by1 = y1 - cy1
        bx2 = x2 - cx1; by2 = y2 - cy1
        import cv2 as _cv2
        disp = crop_native.copy()
        _cv2.rectangle(disp, (bx1, by1), (bx2, by2), (255, 50, 50), 2)

        axes[0][col].imshow(disp)
        axes[0][col].set_title(
            f"Native: {int(x2-x1)}×{int(y2-y1)} px\n"
            f"Area={int(g['area_native'])} px²", fontsize=7)
        axes[0][col].axis("off")

        # draw box on small crop
        disp_s = crop_small.copy()
        bsx1 = sx1 - scx1; bsy1 = sy1 - scy1
        bsx2 = sx2 - scx1; bsy2 = sy2 - scy1
        _cv2.rectangle(disp_s, (bsx1, bsy1), (bsx2, bsy2), (255, 50, 50), 1)
        axes[1][col].imshow(disp_s)
        axes[1][col].set_title(
            f"At 640: {sx2-sx1:.1f}×{sy2-sy1:.1f} px\n"
            f"Area={g['area_at_640']:.1f} px²", fontsize=7)
        axes[1][col].axis("off")

    axes[0][0].set_ylabel("Native resolution", fontsize=9)
    axes[1][0].set_ylabel("After resize to 640", fontsize=9)

    plt.suptitle("Resolution Collapse: same defect at native vs 640px scale",
                 fontsize=10, y=1.01)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg  = load_config(args.config)

    repo_root     = Path(args.config).parent.parent
    processed_dir = (repo_root / cfg["dataset"]["processed_dir"]).resolve()
    pred_dir      = (repo_root / cfg["outputs"]["dir"] / "predictions").resolve()
    fig_dir       = (repo_root / cfg["outputs"]["dir"] / "figures" / "resolution").resolve()
    fig_dir.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(processed_dir / "metadata.csv")

    print(f"Loading GT boxes for split='{args.split}'...")
    gt_boxes = load_gt(processed_dir, args.split, meta)
    print(f"  {len(gt_boxes)} GT boxes")

    # collapse stats
    at640 = np.array([g["area_at_640"] for g in gt_boxes])
    for lo, hi, label in [(0, 4, "<4"), (4, 16, "4–16"), (16, 64, "16–64"), (64, 1e9, ">64")]:
        n = ((at640 >= lo) & (at640 < hi)).sum()
        print(f"  Area at 640  {label:>7} px²: {n:3d} boxes ({100*n/len(at640):.1f}%)")

    # load detections per method
    pred_files = {
        "full_640":       pred_dir / "full_640.csv",
        "full_1280":      pred_dir / "full_1280.csv",
        "tile_640":       pred_dir / "tile_640_overlap0_nms.csv",
        "tile_640_ov128": pred_dir / "tile_640_overlap128_nms.csv",
        "tatm_640_ov128": pred_dir / "tatm_tile_640_overlap128.csv",
    }

    print("\nMatching predictions to GT (IoU ≥ 0.5)...")
    detections: dict[str, set[int]] = {}
    for mkey, fpath in pred_files.items():
        if not fpath.exists():
            print(f"  WARNING: {fpath.name} not found, skipping {mkey}")
            continue
        pred_df = load_pred(fpath)
        detections[mkey] = detected_gt_indices(gt_boxes, pred_df)
        print(f"  {mkey:20s}: {len(detections[mkey])}/{len(gt_boxes)} GT matched")

    print("\nGenerating figures...")
    plot_collapse_histogram(gt_boxes, fig_dir / "collapse_histogram.png")
    plot_recall_by_area(gt_boxes, detections, fig_dir / "recall_by_area.png")
    plot_collapse_examples(gt_boxes, processed_dir, fig_dir / "collapse_examples.png")

    # print summary table
    print("\nRecall by area bin at 640px input:")
    header = f"{'Method':22s}" + "".join(f"  {l:>9}" for l in AREA_LABELS)
    print(header)
    print("-" * len(header))
    for mkey, label, _ in METHOD_ORDER:
        if mkey not in detections:
            continue
        matched = detections[mkey]
        row = f"{label:22s}"
        for lo, hi in AREA_BINS:
            idx = [i for i, g in enumerate(gt_boxes)
                   if lo <= g["area_at_640"] < hi]
            r = sum(1 for i in idx if i in matched) / len(idx) if idx else float("nan")
            row += f"  {r:9.4f}" if not np.isnan(r) else f"  {'—':>9}"
        print(row)

    print(f"\nAll figures → {fig_dir}/")


if __name__ == "__main__":
    main()
