"""
Steps 4 & 5: Tile-based inference with global class-aware NMS merging.

Splits each test image into tiles (with optional overlap), runs YOLO on each
tile, remaps local box coordinates to global image coordinates, computes the
boundary distance for each detection, then applies class-aware NMS across all
tiles of an image to produce a clean merged result.

Two output CSVs are written per run:

  tile_{size}_overlap{ov}.csv       — raw per-tile detections (all tiles,
                                       before global NMS). Used by Step 6 TA-TM.
  tile_{size}_overlap{ov}_nms.csv   — global NMS merged result. Used by Step 7
                                       evaluate.py as the "Tile+NMS" baseline.

Raw CSV columns:
    image_id, image_path, tile_id, tile_x, tile_y, tile_w, tile_h,
    grid_row, grid_col, x1, y1, x2, y2, score, class_id,
    method, boundary_distance

NMS CSV columns:
    image_id, image_path, x1, y1, x2, y2, score, class_id,
    method, boundary_distance, infer_time_s

All (x1,y1,x2,y2) are in ORIGINAL image pixel coordinates.
boundary_distance is computed in tile-local pixel coordinates.

Usage:
    python src/infer_tiles.py --config configs/default.yaml --tile-size 640 --overlap 0
    python src/infer_tiles.py --config configs/default.yaml --tile-size 640 --overlap 128
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_config, class_aware_nms, boundary_distance, generate_tiles


# ── per-tile inference ────────────────────────────────────────────────────────

def infer_tile(model, crop: np.ndarray, tile_size: int,
               conf: float, iou: float, device) -> list[dict]:
    """
    Run YOLO on a single tile crop.
    Returns boxes in tile-LOCAL pixel coordinates.
    """
    results = model.predict(
        source  = crop,
        imgsz   = tile_size,
        conf    = conf,
        iou     = iou,
        device  = device,
        verbose = False,
    )

    dets = []
    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue
        boxes  = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
        cls    = r.boxes.cls.cpu().numpy().astype(int)
        for (x1, y1, x2, y2), score, cls_id in zip(boxes, scores, cls):
            dets.append({
                "x1_local": float(x1), "y1_local": float(y1),
                "x2_local": float(x2), "y2_local": float(y2),
                "score":    float(score),
                "class_id": int(cls_id),
            })
    return dets


# ── per-image pipeline ────────────────────────────────────────────────────────

def process_image(model, img_path: Path, tile_size: int, overlap: int,
                  conf: float, iou: float, device,
                  method: str, img_id: int) -> tuple[list[dict], float]:
    """
    Tile an image, run YOLO on each tile, remap to global coords.
    Returns (raw_rows, elapsed_seconds).
    raw_rows are pre-NMS and include all tile metadata.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        raise RuntimeError(f"Cannot read image: {img_path}")
    img_h, img_w = img.shape[:2]

    tiles = generate_tiles(img_w, img_h, tile_size, overlap)

    raw_rows = []
    t0 = time.perf_counter()

    for tile in tiles:
        tx, ty = tile["tile_x"], tile["tile_y"]
        tw, th = tile["tile_w"], tile["tile_h"]

        crop = img[ty:ty + th, tx:tx + tw]
        dets = infer_tile(model, crop, tile_size, conf, iou, device)

        for d in dets:
            xl, yl, xr, yr = d["x1_local"], d["y1_local"], d["x2_local"], d["y2_local"]

            # clamp local coords to tile dimensions
            xl = max(0.0, min(xl, tw))
            yl = max(0.0, min(yl, th))
            xr = max(0.0, min(xr, tw))
            yr = max(0.0, min(yr, th))

            bd = boundary_distance(xl, yl, xr, yr, tw, th)

            raw_rows.append({
                "image_id":        img_id,
                "image_path":      str(img_path),
                "tile_id":         tile["tile_id"],
                "tile_x":          tx,
                "tile_y":          ty,
                "tile_w":          tw,
                "tile_h":          th,
                "grid_row":        tile["grid_row"],
                "grid_col":        tile["grid_col"],
                # global image coordinates
                "x1": round(tx + xl, 2), "y1": round(ty + yl, 2),
                "x2": round(tx + xr, 2), "y2": round(ty + yr, 2),
                "score":           round(d["score"], 4),
                "class_id":        d["class_id"],
                "method":          method,
                "boundary_distance": round(bd, 2),
            })

    elapsed = time.perf_counter() - t0
    # stamp total image inference time on every raw row (needed by TA-TM evaluate)
    for r in raw_rows:
        r["infer_time_s"] = round(elapsed, 4)
    return raw_rows, elapsed, len(tiles)


def apply_nms_to_image(raw_rows: list[dict], iou_threshold: float,
                       method: str, infer_time_s: float) -> list[dict]:
    """Global class-aware NMS over all tile detections for one image."""
    if not raw_rows:
        return []

    boxes    = [[r["x1"], r["y1"], r["x2"], r["y2"]] for r in raw_rows]
    scores   = [r["score"] for r in raw_rows]
    cls_ids  = [r["class_id"] for r in raw_rows]

    keep = class_aware_nms(boxes, scores, cls_ids, iou_threshold)

    nms_rows = []
    for i in keep:
        r = raw_rows[i]
        nms_rows.append({
            "image_id":         r["image_id"],
            "image_path":       r["image_path"],
            "x1":               r["x1"],
            "y1":               r["y1"],
            "x2":               r["x2"],
            "y2":               r["y2"],
            "score":            r["score"],
            "class_id":         r["class_id"],
            "method":           method,
            "boundary_distance": r["boundary_distance"],
            "infer_time_s":     round(infer_time_s, 4),
        })
    return nms_rows


# ── helpers ───────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Tile-based YOLO inference with NMS merging")
    p.add_argument("--config",     required=True)
    p.add_argument("--tile-size",  type=int, default=None,
                   help="Tile size in pixels (default: from config tiling.tile_size)")
    p.add_argument("--overlap",    type=int, default=None,
                   help="Tile overlap in pixels (default: from config tiling.overlap)")
    p.add_argument("--weights",    default=None,
                   help="Override model weights path (default: read best_weights.txt)")
    p.add_argument("--split",      default="test")
    return p.parse_args()


def load_weights(runs_dir: Path, override: str | None) -> Path:
    if override:
        p = Path(override)
        if not p.exists():
            raise FileNotFoundError(f"Weights not found: {p}")
        return p
    pointer = runs_dir / "best_weights.txt"
    if not pointer.exists():
        raise FileNotFoundError(
            f"best_weights.txt not found at {pointer}\n"
            "Run train_detector.py first."
        )
    weights = Path(pointer.read_text().strip())
    if not weights.exists():
        raise FileNotFoundError(f"Weights listed in best_weights.txt not found: {weights}")
    return weights


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg  = load_config(args.config)

    repo_root     = Path(args.config).parent.parent
    processed_dir = (repo_root / cfg["dataset"]["processed_dir"]).resolve()
    runs_dir      = (repo_root / cfg["outputs"]["dir"] / "runs").resolve()
    pred_dir      = (repo_root / cfg["outputs"]["dir"] / "predictions").resolve()
    pred_dir.mkdir(parents=True, exist_ok=True)

    tiling_cfg = cfg["tiling"]
    tile_size  = args.tile_size if args.tile_size is not None else tiling_cfg["tile_size"]
    overlap    = args.overlap   if args.overlap   is not None else tiling_cfg["overlap"]
    conf       = tiling_cfg["conf_threshold"]
    iou        = tiling_cfg["iou_threshold"]
    tau        = tiling_cfg["boundary_tau"]
    device     = cfg["model"]["device"]

    method  = f"tile_{tile_size}_overlap{overlap}"
    out_raw = pred_dir / f"{method}.csv"
    out_nms = pred_dir / f"{method}_nms.csv"

    weights = load_weights(runs_dir, args.weights)

    img_dir = processed_dir / "images" / args.split
    if not img_dir.exists():
        print(f"ERROR: image directory not found: {img_dir}")
        sys.exit(1)
    images = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    if not images:
        print(f"ERROR: no images found in {img_dir}")
        sys.exit(1)

    meta_path = processed_dir / "metadata.csv"
    id_map = {}
    if meta_path.exists():
        meta   = pd.read_csv(meta_path)
        id_map = dict(zip(meta["file_name"], meta["image_id"]))

    print(f"Model     : {weights}")
    print(f"Split     : {args.split}  ({len(images)} images)")
    print(f"Tile size : {tile_size}   Overlap : {overlap}   Tau : {tau}")
    print(f"Conf      : {conf}   IoU NMS : {iou}")
    print(f"Raw CSV   : {out_raw}")
    print(f"NMS CSV   : {out_nms}")
    print()

    from ultralytics import YOLO
    model = YOLO(str(weights))

    all_raw  = []
    all_nms  = []
    total_tiles = 0

    for img_path in images:
        img_id = id_map.get(img_path.name, -1)

        raw_rows, elapsed, n_tiles = process_image(
            model, img_path, tile_size, overlap,
            conf, iou, device, method, img_id
        )
        total_tiles += n_tiles
        all_raw.extend(raw_rows)

        nms_rows = apply_nms_to_image(raw_rows, iou, method, elapsed)
        all_nms.extend(nms_rows)

        n_raw = len(raw_rows)
        n_nms = len(nms_rows)
        print(f"  {img_path.name:30s}  tiles={n_tiles:3d}  "
              f"raw={n_raw:4d}  nms={n_nms:3d}  {elapsed*1000:.0f}ms")

    # save raw (for TA-TM)
    pd.DataFrame(all_raw).to_csv(out_raw, index=False)

    # save NMS-merged (for evaluation)
    pd.DataFrame(all_nms).to_csv(out_nms, index=False)

    # summary
    n_imgs  = len(images)
    total_t = pd.DataFrame(all_nms)["infer_time_s"].sum() if all_nms else 0
    avg_t   = total_t / n_imgs
    avg_raw = len(all_raw) / n_imgs
    avg_nms = len(all_nms) / n_imgs
    avg_til = total_tiles / n_imgs

    print()
    print(f"Done.")
    print(f"  Images          : {n_imgs}")
    print(f"  Avg tiles/image : {avg_til:.1f}")
    print(f"  Avg raw dets    : {avg_raw:.1f}  → after NMS: {avg_nms:.1f}")
    print(f"  Avg time/image  : {avg_t*1000:.1f} ms")
    print(f"  Raw  → {out_raw}")
    print(f"  NMS  → {out_nms}")


if __name__ == "__main__":
    main()
