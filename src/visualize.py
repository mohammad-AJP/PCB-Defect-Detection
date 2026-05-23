"""
Step 9: Visualization.

Generates four types of comparison figures for the paper:

  1. comparison_grid/   — per-image 2×3 grid showing GT, Full-640 preds,
                          Full-1280 preds, Tile+NMS, Tile+Ov+NMS, TA-TM
  2. boundary_artifact/ — zoomed tile-boundary regions showing missed/duplicate
                          detections with and without TA-TM
  3. method_overlay/    — side-by-side TP/FP/FN coloured overlays per method
  4. summary_chart.png  — bar chart of all key metrics across methods

Usage:
    python src/visualize.py --config configs/default.yaml
    python src/visualize.py --config configs/default.yaml --n-images 5
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


# ── colour scheme ─────────────────────────────────────────────────────────────
# BGR for OpenCV drawing
CLR_GT   = (0,  200,   0)   # green  — ground truth
CLR_TP   = (255, 140,  0)   # orange — true positive
CLR_FP   = (0,   0,  220)   # red    — false positive
CLR_FN   = (0,  220, 220)   # yellow — false negative (missed GT)

CLASS_NAMES = ["missing_pad", "mouse_bite", "open_circuit",
               "short", "spur", "spurious_copper"]

METHODS = [
    ("full_640",        "Full-640",             "outputs/predictions/full_640.csv"),
    ("full_1280",       "Full-1280",            "outputs/predictions/full_1280.csv"),
    ("tile_640",        "Tile+NMS",             "outputs/predictions/tile_640_overlap0_nms.csv"),
    ("tile_640_ov128",  "Tile+Ov+NMS",          "outputs/predictions/tile_640_overlap128_nms.csv"),
    ("tatm_640_ov128",  "Tile+Ov+TA-TM",        "outputs/predictions/tatm_tile_640_overlap128.csv"),
]

IOU_THRESH = 0.5


# ── helpers ───────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Visualize detection results")
    p.add_argument("--config",    required=True)
    p.add_argument("--split",     default="test")
    p.add_argument("--n-images",  type=int, default=4,
                   help="Number of test images to visualize per figure type")
    p.add_argument("--max-side",  type=int, default=1200,
                   help="Max side length for saved images (px)")
    return p.parse_args()


def load_gt_for_image(lbl_path: Path, img_w: int, img_h: int) -> list[dict]:
    if not lbl_path.exists():
        return []
    boxes = []
    for line in lbl_path.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls_id = int(parts[0])
        cx, cy, bw, bh = map(float, parts[1:])
        x1, y1, x2, y2 = xywhn_to_xyxy(cx, cy, bw, bh, img_w, img_h)
        boxes.append({"cls": cls_id, "x1": x1, "y1": y1, "x2": x2, "y2": y2})
    return boxes


def load_pred_for_image(pred_df: pd.DataFrame, image_id: int) -> list[dict]:
    rows = pred_df[pred_df["image_id"] == image_id]
    return [{"cls": int(r.class_id),
             "x1": r.x1, "y1": r.y1, "x2": r.x2, "y2": r.y2,
             "conf": r.conf}
            for _, r in rows.iterrows()]


def match_tp_fp_fn(gt: list[dict], preds: list[dict]) -> tuple[list, list, list]:
    """Returns (tp_preds, fp_preds, fn_gts) using greedy IoU matching."""
    matched_gt = set()
    tp, fp = [], []
    for p in sorted(preds, key=lambda x: x["conf"], reverse=True):
        best_iou, best_gi = IOU_THRESH - 1e-9, -1
        for gi, g in enumerate(gt):
            if gi in matched_gt or g["cls"] != p["cls"]:
                continue
            iou = compute_iou([p["x1"], p["y1"], p["x2"], p["y2"]],
                              [g["x1"], g["y1"], g["x2"], g["y2"]])
            if iou > best_iou:
                best_iou, best_gi = iou, gi
        if best_gi >= 0:
            tp.append(p)
            matched_gt.add(best_gi)
        else:
            fp.append(p)
    fn = [gt[i] for i in range(len(gt)) if i not in matched_gt]
    return tp, fp, fn


def draw_box(img, x1, y1, x2, y2, color, label="", thickness=2):
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    if label:
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(img, (x1, max(0, y1 - th - 4)), (x1 + tw + 2, y1), color, -1)
        cv2.putText(img, label, (x1 + 1, max(th, y1 - 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


def resize_to_max(img, max_side: int):
    h, w = img.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        return cv2.resize(img, (int(w * scale), int(h * scale)))
    return img


def scale_boxes(boxes: list[dict], scale: float) -> list[dict]:
    return [{**b, "x1": b["x1"]*scale, "y1": b["y1"]*scale,
             "x2": b["x2"]*scale, "y2": b["y2"]*scale} for b in boxes]


# ── figure type 1: comparison grid ───────────────────────────────────────────

def draw_method_panel(img_bgr, gt, preds, title, max_side):
    """Draw TP/FP/FN boxes on a copy of img, return scaled panel."""
    vis = img_bgr.copy()
    tp, fp, fn = match_tp_fp_fn(gt, preds)
    for b in fn:
        draw_box(vis, b["x1"], b["y1"], b["x2"], b["y2"], CLR_FN, "FN")
    for b in tp:
        name = CLASS_NAMES[b["cls"]] if b["cls"] < len(CLASS_NAMES) else str(b["cls"])
        draw_box(vis, b["x1"], b["y1"], b["x2"], b["y2"], CLR_TP,
                 f"{name} {b['conf']:.2f}")
    for b in fp:
        name = CLASS_NAMES[b["cls"]] if b["cls"] < len(CLASS_NAMES) else str(b["cls"])
        draw_box(vis, b["x1"], b["y1"], b["x2"], b["y2"], CLR_FP,
                 f"FP:{name}")
    vis = resize_to_max(vis, max_side)
    # add title bar
    bar = np.zeros((28, vis.shape[1], 3), dtype=np.uint8)
    n_tp, n_fp, n_fn = len(tp), len(fp), len(fn)
    txt = f"{title}   TP={n_tp} FP={n_fp} FN={n_fn}"
    cv2.putText(bar, txt, (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (230, 230, 230), 1, cv2.LINE_AA)
    return np.vstack([bar, vis])


def save_comparison_grid(img_path: Path, img_id: int, gt: list[dict],
                         pred_dfs: dict, fig_dir: Path, max_side: int):
    img = cv2.imread(str(img_path))
    if img is None:
        return
    img_h, img_w = img.shape[:2]
    scale = min(1.0, max_side / max(img_h, img_w))
    gt_sc = scale_boxes(gt, scale)
    img_sc = cv2.resize(img, (int(img_w * scale), int(img_h * scale)))

    # GT panel
    gt_panel = img_sc.copy()
    for b in gt_sc:
        name = CLASS_NAMES[b["cls"]] if b["cls"] < len(CLASS_NAMES) else str(b["cls"])
        draw_box(gt_panel, b["x1"], b["y1"], b["x2"], b["y2"], CLR_GT, name)
    bar = np.zeros((28, gt_panel.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, f"Ground Truth  ({len(gt)} boxes)", (4, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
    gt_panel = np.vstack([bar, gt_panel])

    panels = [gt_panel]
    for mkey, label, _ in METHODS:
        if mkey not in pred_dfs:
            continue
        preds = load_pred_for_image(pred_dfs[mkey], img_id)
        preds_sc = scale_boxes(preds, scale)
        panel = draw_method_panel(img_sc, gt_sc, preds_sc, label, max_side * 2)
        panels.append(panel)

    # arrange in 2 rows of 3
    rows = []
    for i in range(0, len(panels), 3):
        row_panels = panels[i:i+3]
        # pad to same height
        max_h = max(p.shape[0] for p in row_panels)
        padded = []
        for p in row_panels:
            if p.shape[0] < max_h:
                pad = np.zeros((max_h - p.shape[0], p.shape[1], 3), dtype=np.uint8)
                p = np.vstack([p, pad])
            padded.append(p)
        rows.append(np.hstack(padded))
    # pad rows to same width
    max_w = max(r.shape[1] for r in rows)
    rows = [np.hstack([r, np.zeros((r.shape[0], max_w - r.shape[1], 3), np.uint8)])
            if r.shape[1] < max_w else r for r in rows]
    grid = np.vstack(rows)

    stem = img_path.stem
    out = fig_dir / f"{stem}_comparison.jpg"
    cv2.imwrite(str(out), grid, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"  Saved: {out.name}")


# ── figure type 2: boundary artifact zoom ─────────────────────────────────────

def save_boundary_zoom(img_path: Path, img_id: int, img_w: int, img_h: int,
                       gt: list[dict], pred_dfs: dict,
                       tile_size: int, fig_dir: Path):
    """
    Find the tile boundary line with the most GT boxes near it, then render
    a zoomed strip showing naive tile NMS vs TA-TM side by side.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return

    # find tile boundaries (vertical lines)
    from utils import generate_tiles
    tiles = generate_tiles(img_w, img_h, tile_size, overlap=0)
    x_bounds = sorted({t["tile_x"] + t["tile_w"]
                       for t in tiles if t["tile_x"] + t["tile_w"] < img_w})
    y_bounds = sorted({t["tile_y"] + t["tile_h"]
                       for t in tiles if t["tile_y"] + t["tile_h"] < img_h})

    # find the boundary with most nearby GT boxes (within 64px)
    best_score, best_axis, best_pos = 0, "x", 0
    for xb in x_bounds:
        n = sum(1 for g in gt if abs((g["x1"] + g["x2"]) / 2 - xb) < 64)
        if n > best_score:
            best_score, best_axis, best_pos = n, "x", xb
    for yb in y_bounds:
        n = sum(1 for g in gt if abs((g["y1"] + g["y2"]) / 2 - yb) < 64)
        if n > best_score:
            best_score, best_axis, best_pos = n, "y", yb

    if best_score == 0:
        return  # no GT near any boundary

    # define crop region
    margin = 200
    if best_axis == "x":
        x1c = max(0, best_pos - margin)
        x2c = min(img_w, best_pos + margin)
        y1c, y2c = 0, img_h
    else:
        y1c = max(0, best_pos - margin)
        y2c = min(img_h, best_pos + margin)
        x1c, x2c = 0, img_w

    crop = img[y1c:y2c, x1c:x2c]
    if crop.size == 0:
        return
    crop = resize_to_max(crop, 1200)
    ch, cw = crop.shape[:2]
    scale = cw / (x2c - x1c)

    def to_local(b):
        return {**b,
                "x1": (b["x1"] - x1c) * scale,
                "y1": (b["y1"] - y1c) * scale,
                "x2": (b["x2"] - x1c) * scale,
                "y2": (b["y2"] - y1c) * scale}

    gt_local = [to_local(b) for b in gt]

    panels = []
    for mkey, label, _ in [("tile_640_ov128", "Tile+Ov+NMS", ""),
                            ("tatm_640_ov128", "Tile+Ov+TA-TM", "")]:
        if mkey not in pred_dfs:
            continue
        preds = [to_local(p)
                 for p in load_pred_for_image(pred_dfs[mkey], img_id)]
        panel = draw_method_panel(crop, gt_local, preds, label, 2400)
        # draw boundary line
        if best_axis == "x":
            bx = int((best_pos - x1c) * scale)
            cv2.line(panel, (bx, 28), (bx, panel.shape[0]),
                     (180, 80, 255), 2, cv2.LINE_AA)
        else:
            by = int((best_pos - y1c) * scale) + 28
            cv2.line(panel, (0, by), (panel.shape[1], by),
                     (180, 80, 255), 2, cv2.LINE_AA)
        panels.append(panel)

    if len(panels) < 2:
        return

    max_h = max(p.shape[0] for p in panels)
    panels = [np.vstack([p, np.zeros((max_h - p.shape[0], p.shape[1], 3), np.uint8)])
              if p.shape[0] < max_h else p for p in panels]
    combined = np.hstack(panels)

    out = fig_dir / f"{img_path.stem}_boundary_zoom.jpg"
    cv2.imwrite(str(out), combined, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"  Saved: {out.name}")


# ── figure type 3: summary bar chart ─────────────────────────────────────────

def save_summary_chart(tables_dir: Path, fig_dir: Path):
    df = pd.read_csv(tables_dir / "main_results.csv")
    order = ["full_640", "full_1280", "tile_640", "tile_640_ov128", "tatm_640_ov128"]
    labels = {
        "full_640":       "Full-640",
        "full_1280":      "Full-1280",
        "tile_640":       "Tile+NMS",
        "tile_640_ov128": "Tile+Ov+NMS",
        "tatm_640_ov128": "Tile+Ov\n+TA-TM",
    }
    df = df[df.method.isin(order)].set_index("method").reindex(order)

    metrics = {
        "mAP@50":        ("mAP50",               "#4C72B0"),
        "Recall":        ("recall",               "#55A868"),
        "Precision":     ("precision",            "#C44E52"),
        "Small recall":  ("small_recall_lt64",    "#DD8452"),
        "Bdry recall\n(0–16px)": ("boundary_recall_lt16", "#8172B2"),
    }

    x = np.arange(len(order))
    width = 0.15
    n = len(metrics)
    offsets = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * width

    fig, ax = plt.subplots(figsize=(13, 5))
    for i, (metric_label, (col, color)) in enumerate(metrics.items()):
        vals = df[col].fillna(0).values
        ax.bar(x + offsets[i], vals, width, label=metric_label,
               color=color, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([labels[m] for m in order], fontsize=9)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.08)
    ax.set_title("PCB Defect Detection — Method Comparison")
    ax.legend(fontsize=8, ncol=5, loc="upper left")
    ax.grid(True, axis="y", alpha=0.3)

    out = fig_dir / "summary_chart.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(str(out), dpi=150)
    plt.close()
    print(f"  Saved: {out.name}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg  = load_config(args.config)

    repo_root     = Path(args.config).parent.parent
    processed_dir = (repo_root / cfg["dataset"]["processed_dir"]).resolve()
    pred_dir      = (repo_root / cfg["outputs"]["dir"] / "predictions").resolve()
    tables_dir    = (repo_root / cfg["outputs"]["dir"] / "tables").resolve()
    fig_dir       = (repo_root / cfg["outputs"]["dir"] / "figures").resolve()

    meta     = pd.read_csv(processed_dir / "metadata.csv")
    test_meta = meta[meta["split"] == args.split]
    tile_size = cfg["tiling"]["tile_size"]

    # load all prediction CSVs once
    pred_dfs = {}
    for mkey, _, rel_path in METHODS:
        fpath = repo_root / rel_path
        if not fpath.exists():
            print(f"WARNING: {fpath.name} not found, skipping {mkey}")
            continue
        df = pd.read_csv(fpath).dropna(subset=["x1"])
        if "score_adj" in df.columns:
            df["conf"] = df["score_adj"]
        else:
            df["conf"] = df["score"]
        df["class_id"] = df["class_id"].astype(int)
        pred_dfs[mkey] = df

    # pick representative images: mix of high/low tile counts
    rng = np.random.default_rng(42)
    sample_rows = test_meta.sample(
        min(args.n_images, len(test_meta)), random_state=42
    )

    # ── comparison grids ──
    grid_dir = fig_dir / "comparison_grid"
    grid_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nGenerating comparison grids → {grid_dir}/")
    for _, row in sample_rows.iterrows():
        img_path = processed_dir / "images" / args.split / row["file_name"]
        img_w, img_h = int(row["width"]), int(row["height"])
        lbl_path = processed_dir / "labels" / args.split / f"{Path(row['file_name']).stem}.txt"
        gt = load_gt_for_image(lbl_path, img_w, img_h)
        save_comparison_grid(img_path, int(row["image_id"]), gt,
                             pred_dfs, grid_dir, args.max_side)

    # ── boundary zoom ──
    bdry_dir = fig_dir / "boundary_artifact"
    bdry_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nGenerating boundary zoom figures → {bdry_dir}/")
    for _, row in sample_rows.iterrows():
        img_path = processed_dir / "images" / args.split / row["file_name"]
        img_w, img_h = int(row["width"]), int(row["height"])
        lbl_path = processed_dir / "labels" / args.split / f"{Path(row['file_name']).stem}.txt"
        gt = load_gt_for_image(lbl_path, img_w, img_h)
        save_boundary_zoom(img_path, int(row["image_id"]), img_w, img_h,
                           gt, pred_dfs, tile_size, bdry_dir)

    # ── summary chart ──
    print(f"\nGenerating summary chart → {fig_dir}/")
    save_summary_chart(tables_dir, fig_dir)

    print(f"\nAll figures saved under {fig_dir}/")


if __name__ == "__main__":
    main()
