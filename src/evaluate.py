"""
Step 7: Evaluation.

Matches predictions against ground-truth YOLO labels using IoU >= 0.5
per class, then computes:
  - Precision, Recall, AP@50, mAP@50
  - Small-defect recall  (GT boxes grouped by area after resizing to 640)
  - Boundary-zone recall (GT boxes grouped by distance to nearest tile edge)
  - Average inference time per image

Results are appended (or updated) in three accumulating CSVs so you can
run this script once per method and compare all methods in one place:

  outputs/tables/main_results.csv      — Table 1: method comparison
  outputs/tables/size_recall.csv       — Table 2: recall by resized-area bin
  outputs/tables/boundary_recall.csv   — Table 3: recall by tile-boundary dist

Usage:
    python src/evaluate.py --config configs/default.yaml \\
        --pred outputs/predictions/full_640.csv --method full_640

    python src/evaluate.py --config configs/default.yaml \\
        --pred outputs/predictions/full_1280.csv --method full_1280

    python src/evaluate.py --config configs/default.yaml \\
        --pred outputs/predictions/tile_640_overlap0_nms.csv --method tile_640

    python src/evaluate.py --config configs/default.yaml \\
        --pred outputs/predictions/tile_640_overlap128_nms.csv --method tile_640_ov128

    python src/evaluate.py --config configs/default.yaml \\
        --pred outputs/predictions/tatm_tile_640_overlap0.csv --method tatm_640
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from utils import (load_config, xywhn_to_xyxy, compute_iou,
                   gt_boundary_distance)


# ── constants ─────────────────────────────────────────────────────────────────

AREA_BINS   = [(0, 4), (4, 16), (16, 64), (64, float("inf"))]
AREA_LABELS = ["0-4 px²", "4-16 px²", "16-64 px²", ">64 px²"]

DIST_BINS   = [(0, 16), (16, 32), (32, float("inf"))]
DIST_LABELS = ["0-16 px", "16-32 px", ">32 px"]

IOU_THRESHOLD = 0.5


# ── ground truth loading ──────────────────────────────────────────────────────

def load_gt(processed_dir: Path, split: str, meta: pd.DataFrame,
            tile_size: int, overlap: int) -> list[dict]:
    """
    Load all GT boxes for the given split.
    Returns a list of dicts, one per box:
      image_id, file_name, img_w, img_h,
      x1, y1, x2, y2, class_id,
      area_orig (px²), area_at_640 (px²), area_at_1280 (px²),
      boundary_dist (px, relative to tile_size/overlap grid)
    """
    lbl_dir = processed_dir / "labels" / split

    gt_boxes = []
    for _, row in meta[meta["split"] == split].iterrows():
        stem  = Path(row["file_name"]).stem
        lbl_f = lbl_dir / f"{stem}.txt"
        if not lbl_f.exists():
            continue

        img_w, img_h = int(row["width"]), int(row["height"])
        scale_640  = min(640  / img_w, 640  / img_h)
        scale_1280 = min(1280 / img_w, 1280 / img_h)

        for line in lbl_f.read_text().strip().splitlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls_id = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:])
            x1, y1, x2, y2 = xywhn_to_xyxy(cx, cy, bw, bh, img_w, img_h)

            w_px = (x2 - x1)
            h_px = (y2 - y1)
            area_orig = w_px * h_px

            # centre for boundary distance
            gcx = (x1 + x2) / 2
            gcy = (y1 + y2) / 2
            bd  = gt_boundary_distance(gcx, gcy, img_w, img_h, tile_size, overlap)

            gt_boxes.append({
                "image_id":    int(row["image_id"]),
                "file_name":   row["file_name"],
                "img_w":       img_w,
                "img_h":       img_h,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "class_id":    cls_id,
                "area_orig":   area_orig,
                "area_at_640": (w_px * scale_640) * (h_px * scale_640),
                "area_at_1280":(w_px * scale_1280) * (h_px * scale_1280),
                "boundary_dist": bd,
            })

    return gt_boxes


# ── prediction loading ────────────────────────────────────────────────────────

def load_predictions(pred_path: Path) -> pd.DataFrame:
    """
    Load prediction CSV. Normalises the confidence column to 'conf':
      - TA-TM CSVs have 'score_adj' → use that
      - others have 'score'
    Drops empty placeholder rows (no detection).
    """
    df = pd.read_csv(pred_path)
    if "score_adj" in df.columns:
        df["conf"] = df["score_adj"]
    elif "score" in df.columns:
        df["conf"] = df["score"]
    else:
        raise ValueError(f"No confidence column found in {pred_path}")

    # drop empty rows (images with zero detections)
    df = df.dropna(subset=["conf", "x1", "y1", "x2", "y2"]).reset_index(drop=True)
    df["class_id"] = df["class_id"].astype(int)
    return df


# ── matching ──────────────────────────────────────────────────────────────────

def match_predictions_to_gt(
    gt_boxes: list[dict],
    pred_df: pd.DataFrame,
    iou_thresh: float = IOU_THRESHOLD,
) -> tuple[list[int], list[int], set[int]]:
    """
    Greedy IoU matching per image, per class.

    Predictions are sorted by confidence (descending) before matching.
    A GT box can be matched at most once.

    Returns:
      tp_flags   — list[int], 1 for each TP prediction, else 0
      fp_flags   — list[int], 1 for each FP prediction, else 0
      matched_gt — set of GT box indices that were successfully matched
    """
    # index GT by (image_id, class_id)
    from collections import defaultdict
    gt_by_img_cls: dict[tuple, list[tuple[int, dict]]] = defaultdict(list)
    for i, g in enumerate(gt_boxes):
        gt_by_img_cls[(g["image_id"], g["class_id"])].append((i, g))

    tp_flags = [0] * len(pred_df)
    fp_flags = [0] * len(pred_df)
    matched_gt: set[int] = set()

    # sort predictions by confidence descending
    order = pred_df["conf"].argsort()[::-1].values

    for pred_idx in order:
        row     = pred_df.iloc[pred_idx]
        img_id  = int(row["image_id"])
        cls_id  = int(row["class_id"])
        pb      = [row["x1"], row["y1"], row["x2"], row["y2"]]

        candidates = gt_by_img_cls.get((img_id, cls_id), [])
        best_iou, best_gt_idx = iou_thresh - 1e-9, -1

        for gt_idx, g in candidates:
            if gt_idx in matched_gt:
                continue
            iou = compute_iou(pb, [g["x1"], g["y1"], g["x2"], g["y2"]])
            if iou > best_iou:
                best_iou, best_gt_idx = iou, gt_idx

        if best_gt_idx >= 0:
            tp_flags[pred_idx] = 1
            matched_gt.add(best_gt_idx)
        else:
            fp_flags[pred_idx] = 1

    return tp_flags, fp_flags, matched_gt


# ── AP / mAP ──────────────────────────────────────────────────────────────────

def compute_ap(tp: np.ndarray, fp: np.ndarray, n_gt: int) -> float:
    """Compute AP@50 from sorted TP/FP arrays and total GT count."""
    if n_gt == 0 or tp.sum() == 0:
        return 0.0

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recalls    = cum_tp / n_gt
    precisions = cum_tp / (cum_tp + cum_fp + 1e-9)

    # add sentinel end-points
    r = np.concatenate([[0.0], recalls,    [recalls[-1]]])
    p = np.concatenate([[1.0], precisions, [0.0]])

    # make precision monotonically non-increasing from right
    for i in range(len(p) - 2, -1, -1):
        p[i] = max(p[i], p[i + 1])

    return float(np.sum((r[1:] - r[:-1]) * p[1:]))


def compute_map(gt_boxes: list[dict], pred_df: pd.DataFrame,
                tp_flags: list[int], fp_flags: list[int]) -> dict:
    """
    Compute per-class AP@50 and mAP@50.
    Returns dict: {class_id: ap, ..., 'mAP': float,
                   'precision': float, 'recall': float}
    """
    class_ids = sorted({g["class_id"] for g in gt_boxes})
    ap_per_cls: dict[int, float] = {}

    # sort predictions by confidence once
    order = pred_df["conf"].argsort()[::-1].values

    for cls in class_ids:
        cls_mask = pred_df["class_id"].values == cls
        n_gt_cls = sum(1 for g in gt_boxes if g["class_id"] == cls)

        # TP/FP for this class in confidence-sorted order
        tp_cls = np.array([tp_flags[i] for i in order if cls_mask[i]])
        fp_cls = np.array([fp_flags[i] for i in order if cls_mask[i]])

        ap_per_cls[cls] = compute_ap(tp_cls, fp_cls, n_gt_cls)

    mAP = float(np.mean(list(ap_per_cls.values()))) if ap_per_cls else 0.0

    total_tp = sum(tp_flags)
    total_fp = sum(fp_flags)
    total_gt = len(gt_boxes)

    precision = total_tp / (total_tp + total_fp + 1e-9)
    recall    = total_tp / (total_gt + 1e-9)

    return {**ap_per_cls, "mAP": mAP,
            "precision": precision, "recall": recall}


# ── size-based recall ─────────────────────────────────────────────────────────

def size_recall(gt_boxes: list[dict], matched_gt: set[int]) -> list[dict]:
    """Recall grouped by GT area after resizing to 640."""
    rows = []
    for (lo, hi), label in zip(AREA_BINS, AREA_LABELS):
        indices = [i for i, g in enumerate(gt_boxes)
                   if lo <= g["area_at_640"] < hi]
        n_gt  = len(indices)
        n_det = sum(1 for i in indices if i in matched_gt)
        rows.append({
            "area_bin":   label,
            "n_gt":       n_gt,
            "n_detected": n_det,
            "recall":     round(n_det / n_gt, 4) if n_gt else None,
        })
    return rows


# ── boundary-zone recall ──────────────────────────────────────────────────────

def boundary_recall(gt_boxes: list[dict], matched_gt: set[int]) -> list[dict]:
    """Recall grouped by GT centre distance to nearest tile boundary."""
    rows = []
    for (lo, hi), label in zip(DIST_BINS, DIST_LABELS):
        indices = [i for i, g in enumerate(gt_boxes)
                   if lo <= g["boundary_dist"] < hi]
        n_gt  = len(indices)
        n_det = sum(1 for i in indices if i in matched_gt)
        rows.append({
            "dist_bin":   label,
            "n_gt":       n_gt,
            "n_detected": n_det,
            "recall":     round(n_det / n_gt, 4) if n_gt else None,
        })
    return rows


# ── table I/O ─────────────────────────────────────────────────────────────────

def upsert_table(table_path: Path, new_rows: list[dict],
                 key_col: str, method: str) -> None:
    """Load existing table, remove rows for this method, append new rows, save."""
    if table_path.exists():
        existing = pd.read_csv(table_path)
        existing = existing[existing[key_col] != method]
    else:
        existing = pd.DataFrame()

    updated = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
    updated.to_csv(table_path, index=False)


# ── helpers ───────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate detection predictions against GT")
    p.add_argument("--config", required=True)
    p.add_argument("--pred",   required=True, help="Path to prediction CSV")
    p.add_argument("--method", required=True, help="Method label for results tables")
    p.add_argument("--split",  default="test")
    return p.parse_args()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    cfg    = load_config(args.config)

    repo_root     = Path(args.config).parent.parent
    processed_dir = (repo_root / cfg["dataset"]["processed_dir"]).resolve()
    tables_dir    = (repo_root / cfg["outputs"]["dir"] / "tables").resolve()
    tables_dir.mkdir(parents=True, exist_ok=True)

    tile_size = cfg["tiling"]["tile_size"]
    overlap   = cfg["tiling"]["overlap"]

    pred_path = Path(args.pred)
    if not pred_path.exists():
        print(f"ERROR: prediction CSV not found: {pred_path}")
        sys.exit(1)

    meta = pd.read_csv(processed_dir / "metadata.csv")

    print(f"Method   : {args.method}")
    print(f"Pred CSV : {pred_path}  ({pred_path.stat().st_size // 1024} KB)")
    print(f"Split    : {args.split}")
    print()

    # ── load GT ──
    gt_boxes = load_gt(processed_dir, args.split, meta, tile_size, overlap)
    print(f"GT boxes : {len(gt_boxes)}  across {meta[meta.split==args.split].shape[0]} images")

    # ── load predictions ──
    pred_df = load_predictions(pred_path)
    print(f"Pred     : {len(pred_df)} detections")

    # ── match ──
    tp_flags, fp_flags, matched_gt = match_predictions_to_gt(gt_boxes, pred_df)

    # ── mAP ──
    map_results = compute_map(gt_boxes, pred_df, tp_flags, fp_flags)
    mAP       = map_results["mAP"]
    precision = map_results["precision"]
    recall    = map_results["recall"]

    # ── inference time ──
    time_col = "infer_time_s"
    if time_col in pred_df.columns and pred_df[time_col].notna().any():
        # one infer_time_s per image → mean over unique images
        time_per_img_ms = (
            pred_df.groupby("image_id")[time_col].first().mean() * 1000
        )
    else:
        time_per_img_ms = float("nan")

    # ── small-defect recall (area_at_640 < 64 px²) ──
    small_idx = [i for i, g in enumerate(gt_boxes) if g["area_at_640"] < 64]
    n_small   = len(small_idx)
    n_small_det = sum(1 for i in small_idx if i in matched_gt)
    small_recall = n_small_det / n_small if n_small else float("nan")

    # ── boundary recall (dist < 16 px) ──
    bd_idx    = [i for i, g in enumerate(gt_boxes) if g["boundary_dist"] < 16]
    n_bd      = len(bd_idx)
    n_bd_det  = sum(1 for i in bd_idx if i in matched_gt)
    bd_recall = n_bd_det / n_bd if n_bd else float("nan")

    # ── print summary ──
    print()
    print("─" * 52)
    print(f"  mAP@50           : {mAP:.4f}")
    print(f"  Precision        : {precision:.4f}")
    print(f"  Recall           : {recall:.4f}")
    print(f"  Small recall     : {small_recall:.4f}  ({n_small_det}/{n_small} boxes < 64px²@640)")
    print(f"  Boundary recall  : {bd_recall:.4f}  ({n_bd_det}/{n_bd} boxes within 16px of tile edge)")
    print(f"  Time / image     : {time_per_img_ms:.1f} ms")
    print(f"  TP / FP          : {sum(tp_flags)} / {sum(fp_flags)}")
    print("  Per-class AP:")
    class_names = cfg.get("class_names",
        ["missing_pad","mouse_bite","open_circuit","short","spur","spurious_copper"])
    for cls_id in sorted(k for k in map_results if isinstance(k, int)):
        name = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
        print(f"    {cls_id} {name:20s}: {map_results[cls_id]:.4f}")
    print("─" * 52)

    # ── Table 1: main results ──
    main_row = [{
        "method":            args.method,
        "mAP50":             round(mAP, 4),
        "precision":         round(precision, 4),
        "recall":            round(recall, 4),
        "small_recall_lt64": round(small_recall, 4) if not np.isnan(small_recall) else None,
        "boundary_recall_lt16": round(bd_recall, 4) if not np.isnan(bd_recall) else None,
        "time_per_img_ms":   round(time_per_img_ms, 1) if not np.isnan(time_per_img_ms) else None,
        "n_detections":      len(pred_df),
        "tp":                sum(tp_flags),
        "fp":                sum(fp_flags),
        "n_gt":              len(gt_boxes),
    }]
    upsert_table(tables_dir / "main_results.csv", main_row, "method", args.method)

    # ── Table 2: size recall ──
    size_rows = size_recall(gt_boxes, matched_gt)
    for r in size_rows:
        r["method"] = args.method
    upsert_table(tables_dir / "size_recall.csv", size_rows, "method", args.method)

    # ── Table 3: boundary recall ──
    bd_rows = boundary_recall(gt_boxes, matched_gt)
    for r in bd_rows:
        r["method"] = args.method
    upsert_table(tables_dir / "boundary_recall.csv", bd_rows, "method", args.method)

    print(f"\nTables updated in {tables_dir}/")
    print(f"  main_results.csv  |  size_recall.csv  |  boundary_recall.csv")


if __name__ == "__main__":
    main()
