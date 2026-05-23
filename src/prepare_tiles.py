"""
Tile-aware training dataset preparation.

Cuts the processed full-image dataset into 640×640 tile crops (with overlap)
and re-labels each tile with the GT boxes that are sufficiently visible in it.
Only train and val splits are tiled — the test split stays as full images so
that all inference strategies are evaluated on identical input.

Output structure:
    data/tiles/
      images/train/   <orig_stem>_r<row>_c<col>.jpg
      images/val/
      labels/train/   <orig_stem>_r<row>_c<col>.txt
      labels/val/
      dataset.yaml
      tile_metadata.csv

Box visibility rule:
    A GT box is kept in a tile if the fraction of its area that falls inside
    the tile is >= min_visibility (default 0.4). This prevents the model from
    learning from nearly-invisible partial defects.

Usage:
    python src/prepare_tiles.py --config configs/default.yaml
    python src/prepare_tiles.py --config configs/default.yaml --tile-size 640 --overlap 128
"""

import argparse
import sys
from pathlib import Path

import cv2
import pandas as pd
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_config, xywhn_to_xyxy, generate_tiles


# ── box helpers ───────────────────────────────────────────────────────────────

def load_yolo_labels(lbl_path: Path, img_w: int, img_h: int) -> list[tuple]:
    """Return list of (cls_id, x1, y1, x2, y2) in absolute pixels."""
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
        boxes.append((cls_id, x1, y1, x2, y2))
    return boxes


def clip_box_to_tile(cls_id: int, x1: float, y1: float, x2: float, y2: float,
                     tx: int, ty: int, tw: int, th: int,
                     min_visibility: float) -> tuple | None:
    """
    Clip a GT box to a tile region.
    Returns (cls_id, cx_n, cy_n, w_n, h_n) in tile-normalised coords,
    or None if the visible fraction is below min_visibility.
    """
    orig_area = (x2 - x1) * (y2 - y1)
    if orig_area <= 0:
        return None

    # intersect with tile
    cx1 = max(x1, tx)
    cy1 = max(y1, ty)
    cx2 = min(x2, tx + tw)
    cy2 = min(y2, ty + th)

    if cx1 >= cx2 or cy1 >= cy2:
        return None

    vis = ((cx2 - cx1) * (cy2 - cy1)) / orig_area
    if vis < min_visibility:
        return None

    # tile-local normalised coords
    lx1, ly1 = cx1 - tx, cy1 - ty
    lx2, ly2 = cx2 - tx, cy2 - ty
    cx_n = (lx1 + lx2) / 2 / tw
    cy_n = (ly1 + ly2) / 2 / th
    w_n  = (lx2 - lx1) / tw
    h_n  = (ly2 - ly1) / th
    return cls_id, round(cx_n, 6), round(cy_n, 6), round(w_n, 6), round(h_n, 6)


# ── per-split processing ──────────────────────────────────────────────────────

def process_split(split: str, processed_dir: Path, tiles_dir: Path,
                  meta_df: pd.DataFrame, tile_size: int, overlap: int,
                  min_visibility: float, jpeg_quality: int) -> list[dict]:
    """
    Tile all images of one split. Returns list of metadata rows.
    """
    img_dir = processed_dir / "images" / split
    lbl_dir = processed_dir / "labels" / split
    out_img = tiles_dir / "images" / split
    out_lbl = tiles_dir / "labels" / split
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    rows = meta_df[meta_df["split"] == split]
    meta_rows = []

    n_pos = 0   # tiles with ≥1 GT box
    n_bg  = 0   # background tiles
    n_box = 0   # total GT boxes written

    for _, row in tqdm(rows.iterrows(), total=len(rows), desc=f"  {split}", ncols=72):
        img_path = img_dir / row["file_name"]
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"    WARNING: cannot read {img_path}, skipping")
            continue

        img_h, img_w = img.shape[:2]
        lbl_path = lbl_dir / f"{Path(row['file_name']).stem}.txt"
        gt_boxes = load_yolo_labels(lbl_path, img_w, img_h)

        tiles = generate_tiles(img_w, img_h, tile_size, overlap)

        for tile in tiles:
            tx, ty = tile["tile_x"], tile["tile_y"]
            tw, th = tile["tile_w"], tile["tile_h"]
            r, c   = tile["grid_row"], tile["grid_col"]

            # clip GT boxes to this tile
            clipped = []
            for cls_id, x1, y1, x2, y2 in gt_boxes:
                result = clip_box_to_tile(cls_id, x1, y1, x2, y2,
                                          tx, ty, tw, th, min_visibility)
                if result is not None:
                    clipped.append(result)

            tile_stem = f"{Path(row['file_name']).stem}_r{r:02d}_c{c:02d}"

            # save image crop
            crop = img[ty:ty + th, tx:tx + tw]
            cv2.imwrite(str(out_img / f"{tile_stem}.jpg"), crop,
                        [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])

            # save label file (empty = background tile)
            lbl_lines = [f"{cls} {cx} {cy} {bw} {bh}"
                         for cls, cx, cy, bw, bh in clipped]
            (out_lbl / f"{tile_stem}.txt").write_text(
                "\n".join(lbl_lines) + ("\n" if lbl_lines else "")
            )

            if clipped:
                n_pos += 1
                n_box += len(clipped)
            else:
                n_bg += 1

            meta_rows.append({
                "tile_file":    f"{tile_stem}.jpg",
                "orig_image":   row["file_name"],
                "orig_image_id": int(row["image_id"]),
                "tile_x":       tx, "tile_y": ty,
                "tile_w":       tw, "tile_h": th,
                "grid_row":     r,  "grid_col": c,
                "n_boxes":      len(clipped),
                "split":        split,
            })

    print(f"    → {n_pos} positive tiles  |  {n_bg} background  |  {n_box} GT boxes")
    return meta_rows


# ── helpers ───────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Prepare tile-based training dataset")
    p.add_argument("--config",        required=True)
    p.add_argument("--tile-size",     type=int, default=None,
                   help="Tile size in pixels (default: from config tile_training.tile_size)")
    p.add_argument("--overlap",       type=int, default=None,
                   help="Tile overlap in pixels (default: from config tile_training.overlap)")
    p.add_argument("--min-visibility",type=float, default=None,
                   help="Min GT box visibility to include in tile (default: from config)")
    p.add_argument("--jpeg-quality",  type=int, default=95,
                   help="JPEG quality for saved tiles (default: 95)")
    return p.parse_args()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg  = load_config(args.config)

    repo_root     = Path(args.config).parent.parent
    processed_dir = (repo_root / cfg["dataset"]["processed_dir"]).resolve()
    tile_cfg      = cfg["tile_training"]
    tiles_dir     = (repo_root / tile_cfg["tiles_dir"]).resolve()

    tile_size      = args.tile_size      or tile_cfg["tile_size"]
    overlap        = args.overlap        if args.overlap is not None else tile_cfg["overlap"]
    min_visibility = args.min_visibility or tile_cfg["min_visibility"]

    print(f"Processed dir : {processed_dir}")
    print(f"Tiles dir     : {tiles_dir}")
    print(f"Tile size     : {tile_size}   Overlap : {overlap}")
    print(f"Min visibility: {min_visibility}")
    print()

    # load metadata and class names
    meta_path = processed_dir / "metadata.csv"
    if not meta_path.exists():
        print("ERROR: metadata.csv not found. Run prepare_dataset.py first.")
        sys.exit(1)
    meta_df = pd.read_csv(meta_path)

    ds_yaml_path = processed_dir / "dataset.yaml"
    with open(ds_yaml_path) as f:
        ds_info = yaml.safe_load(f)
    class_names = ds_info["names"]

    tiles_dir.mkdir(parents=True, exist_ok=True)

    all_meta = []
    for split in ("train", "val"):
        print(f"Processing {split}...")
        rows = process_split(split, processed_dir, tiles_dir, meta_df,
                             tile_size, overlap, min_visibility, args.jpeg_quality)
        all_meta.extend(rows)

    # write tile dataset.yaml
    tile_yaml = {
        "path":  str(tiles_dir),
        "train": "images/train",
        "val":   "images/val",
        "nc":    ds_info["nc"],
        "names": class_names,
    }
    tile_yaml_path = tiles_dir / "dataset.yaml"
    with open(tile_yaml_path, "w") as f:
        yaml.dump(tile_yaml, f, default_flow_style=False, sort_keys=False)

    # write tile metadata
    tile_meta_path = tiles_dir / "tile_metadata.csv"
    pd.DataFrame(all_meta).to_csv(tile_meta_path, index=False)

    # summary
    tile_meta_df = pd.DataFrame(all_meta)
    for split in ("train", "val"):
        sub = tile_meta_df[tile_meta_df.split == split]
        n_total  = len(sub)
        n_pos    = (sub.n_boxes > 0).sum()
        n_bg     = (sub.n_boxes == 0).sum()
        print(f"\n  {split}: {n_total} tiles  "
              f"({n_pos} positive = {100*n_pos/n_total:.1f}%,  "
              f"{n_bg} background = {100*n_bg/n_total:.1f}%)")

    print(f"\nTile dataset.yaml → {tile_yaml_path}")
    print(f"Tile metadata     → {tile_meta_path}")
    print(f"\nNext: train on tile dataset with --data {tile_yaml_path}")


if __name__ == "__main__":
    main()
