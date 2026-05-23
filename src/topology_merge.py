"""
Step 6: Topology-Aware Tile Merging (TA-TM).

Reads the raw per-tile detection CSV produced by infer_tiles.py, then:

  1. Builds a tile-adjacency graph from (grid_row, grid_col) metadata.
  2. For each boundary-sensitive detection (boundary_distance < tau):
       - Identifies which tile edges it is near (left/right/top/bottom).
       - Searches the adjacent tile for same-class detections that are
         also near the shared boundary AND overlap in the perpendicular axis.
       - Sets agreement score A = max confidence of matching neighbor boxes.
  3. Adjusts confidence:
         s' = min(1.0,  s  +  λ · A  +  μ · C)
     where C is an optional edge-continuity proxy (disabled by default, μ=0).
  4. Applies global class-aware NMS using the ADJUSTED scores (so boundary
     detections with neighbour agreement win over isolated low-confidence ones).

Output CSV: outputs/predictions/tatm_<input_stem>.csv
  Columns: image_id, image_path, x1, y1, x2, y2, score_orig, score_adj,
           class_id, method, boundary_distance, infer_time_s

Usage:
    python src/topology_merge.py --config configs/default.yaml \\
        --pred outputs/predictions/tile_640_overlap0.csv

    python src/topology_merge.py --config configs/default.yaml \\
        --pred outputs/predictions/tile_640_overlap128.csv
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_config, class_aware_nms


# ── adjacency & agreement ─────────────────────────────────────────────────────

# Maps direction → (d_grid_row, d_grid_col, local_coord_check, perp_axes)
# local_coord_check: lambda(det, tau, tw, th) → bool  — is det near this edge?
# perp_check: lambda(det, neighbor) → bool  — do they overlap on the shared axis?
_DIRECTIONS = {
    "right":  ( 0, +1,
                lambda d, tau, tw, th: tw - d["x2_local"] < tau,
                lambda d, nb: d["y1"] < nb["y2"] and nb["y1"] < d["y2"],
                lambda nb, tau: nb["x1_local"] < tau),
    "left":   ( 0, -1,
                lambda d, tau, tw, th: d["x1_local"] < tau,
                lambda d, nb: d["y1"] < nb["y2"] and nb["y1"] < d["y2"],
                lambda nb, tau: nb["tile_w"] - nb["x2_local"] < tau),
    "bottom": (+1,  0,
                lambda d, tau, tw, th: th - d["y2_local"] < tau,
                lambda d, nb: d["x1"] < nb["x2"] and nb["x1"] < d["x2"],
                lambda nb, tau: nb["y1_local"] < tau),
    "top":    (-1,  0,
                lambda d, tau, tw, th: d["y1_local"] < tau,
                lambda d, nb: d["x1"] < nb["x2"] and nb["x1"] < d["x2"],
                lambda nb, tau: nb["tile_h"] - nb["y2_local"] < tau),
}


def _add_local_coords(rows: list[dict]) -> None:
    """Add x/y_local fields (tile-coordinate system) in-place."""
    for r in rows:
        r["x1_local"] = r["x1"] - r["tile_x"]
        r["y1_local"] = r["y1"] - r["tile_y"]
        r["x2_local"] = r["x2"] - r["tile_x"]
        r["y2_local"] = r["y2"] - r["tile_y"]


def _neighbour_agreement(det: dict, neighbours: list[dict],
                          direction: str, tau: int) -> float:
    """
    Return the maximum confidence among neighbour detections that:
      - are of the same class (guaranteed by caller),
      - are near the shared boundary from their side,
      - overlap with det along the perpendicular axis.
    Returns 0.0 if no qualifying neighbour is found.
    """
    _, _, _, perp_overlap, nb_near = _DIRECTIONS[direction]
    best = 0.0
    for nb in neighbours:
        if not nb_near(nb, tau):
            continue
        if not perp_overlap(det, nb):
            continue
        best = max(best, nb["score"])
    return best


def adjust_scores_for_image(rows: list[dict], tau: int,
                             lambda_adj: float, mu_edge: float,
                             use_edge_continuity: bool) -> list[dict]:
    """
    Compute adjusted scores for all detections of one image.
    Returns a new list of dicts with 'score_adj' added.
    """
    _add_local_coords(rows)

    # index detections by (grid_row, grid_col) and class
    tile_cls_dets: dict[tuple, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        tile_cls_dets[(r["grid_row"], r["grid_col"])][r["class_id"]].append(r)

    adjusted = []
    for r in rows:
        s      = r["score"]
        s_adj  = s

        if r["boundary_distance"] < tau:
            gr, gc = r["grid_row"], r["grid_col"]
            tw, th = r["tile_w"], r["tile_h"]
            cls    = r["class_id"]
            A      = 0.0

            for direction, (dr, dc, near_self, _, _) in _DIRECTIONS.items():
                if not near_self(r, tau, tw, th):
                    continue
                nb_key = (gr + dr, gc + dc)
                nb_dets = tile_cls_dets.get(nb_key, {}).get(cls, [])
                if nb_dets:
                    a = _neighbour_agreement(r, nb_dets, direction, tau)
                    A = max(A, a)

            # edge-continuity proxy C (optional, off by default)
            C = 0.0
            # (future: image-gradient continuity across shared edge)

            s_adj = min(1.0, s + lambda_adj * A + mu_edge * C)

        adjusted.append({**r, "score_adj": round(s_adj, 4)})

    return adjusted


# ── per-image TA-TM pipeline ──────────────────────────────────────────────────

def tatm_image(df_img: pd.DataFrame, tau: int, lambda_adj: float,
               mu_edge: float, use_edge_continuity: bool,
               iou_threshold: float, out_method: str) -> list[dict]:
    """Full TA-TM pipeline for the detections of one image."""
    rows = df_img.to_dict("records")
    if not rows:
        return []

    # score adjustment
    adj_rows = adjust_scores_for_image(rows, tau, lambda_adj, mu_edge, use_edge_continuity)

    # global class-aware NMS using ADJUSTED scores
    boxes   = [[r["x1"], r["y1"], r["x2"], r["y2"]] for r in adj_rows]
    scores  = [r["score_adj"] for r in adj_rows]
    cls_ids = [r["class_id"] for r in adj_rows]

    keep = class_aware_nms(boxes, scores, cls_ids, iou_threshold)

    # infer_time_s is the tile inference time (TA-TM is just post-processing)
    infer_time = df_img["infer_time_s"].iloc[0] if "infer_time_s" in df_img.columns else None

    result = []
    for i in keep:
        r = adj_rows[i]
        result.append({
            "image_id":         r["image_id"],
            "image_path":       r["image_path"],
            "x1":               r["x1"],
            "y1":               r["y1"],
            "x2":               r["x2"],
            "y2":               r["y2"],
            "score_orig":       round(r["score"], 4),
            "score_adj":        round(r["score_adj"], 4),
            "class_id":         r["class_id"],
            "method":           out_method,
            "boundary_distance": r["boundary_distance"],
            "infer_time_s":     infer_time,
        })
    return result


# ── helpers ───────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Topology-Aware Tile Merging (TA-TM)")
    p.add_argument("--config",  required=True)
    p.add_argument("--pred",    required=True,
                   help="Path to raw tile prediction CSV (tile_*_overlap*.csv)")
    p.add_argument("--tau",     type=int,   default=None,
                   help="Boundary distance threshold in px (overrides config tiling.boundary_tau)")
    p.add_argument("--lambda",  type=float, default=None, dest="lambda_adj",
                   help="Adjacent-agreement weight (overrides config topology.lambda_adj)")
    p.add_argument("--suffix",  default=None,
                   help="Extra suffix for output filename, e.g. '_tau32' → tatm_..._tau32.csv")
    return p.parse_args()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg  = load_config(args.config)

    pred_path = Path(args.pred)
    if not pred_path.exists():
        print(f"ERROR: prediction CSV not found: {pred_path}")
        sys.exit(1)

    repo_root = Path(args.config).parent.parent
    pred_dir  = (repo_root / cfg["outputs"]["dir"] / "predictions").resolve()

    topo_cfg = cfg["topology"]
    lambda_adj    = args.lambda_adj if args.lambda_adj is not None else topo_cfg["lambda_adj"]
    mu_edge       = topo_cfg["mu_edge"]
    use_edge_cont = topo_cfg["use_edge_continuity"]
    tau           = args.tau if args.tau is not None else cfg["tiling"]["boundary_tau"]
    iou_threshold = cfg["tiling"]["iou_threshold"]

    # output method name and file
    stem       = pred_path.stem
    suffix     = args.suffix or ""
    out_method = f"tatm_{stem}{suffix}"
    out_path   = pred_dir / f"{out_method}.csv"

    print(f"Input      : {pred_path}")
    print(f"Output     : {out_path}")
    print(f"τ (tau)    : {tau} px")
    print(f"λ (lambda) : {lambda_adj}")
    print(f"μ (mu)     : {mu_edge}  edge_continuity={use_edge_cont}")
    print(f"IoU NMS    : {iou_threshold}")
    print()

    # check required columns
    df = pd.read_csv(pred_path)
    required = {"image_id", "image_path", "tile_id", "tile_x", "tile_y",
                "tile_w", "tile_h", "grid_row", "grid_col",
                "x1", "y1", "x2", "y2", "score", "class_id", "boundary_distance"}
    missing = required - set(df.columns)
    if missing:
        print(f"ERROR: input CSV is missing columns: {missing}")
        print("Make sure you pass the RAW tile CSV (not the _nms.csv).")
        sys.exit(1)

    # add infer_time_s if missing (not present in raw tile CSV by default)
    if "infer_time_s" not in df.columns:
        df["infer_time_s"] = None

    all_results = []
    image_ids = df["image_id"].unique()

    for img_id in image_ids:
        df_img = df[df["image_id"] == img_id]
        results = tatm_image(
            df_img, tau, lambda_adj, mu_edge, use_edge_cont,
            iou_threshold, out_method
        )
        all_results.extend(results)

        n_raw = len(df_img)
        n_out = len(results)
        n_adj = sum(1 for r in results if abs(r["score_adj"] - r["score_orig"]) > 1e-6)
        img_name = Path(df_img["image_path"].iloc[0]).name
        print(f"  {img_name:30s}  raw={n_raw:4d}  tatm={n_out:3d}  score-adjusted={n_adj}")

    out_df = pd.DataFrame(all_results)
    out_df.to_csv(out_path, index=False)

    # summary stats
    n_imgs     = len(image_ids)
    n_total    = len(out_df)
    n_adjusted = (out_df["score_adj"] > out_df["score_orig"]).sum() if len(out_df) else 0
    avg_dets   = n_total / n_imgs if n_imgs else 0

    print()
    print(f"Done.")
    print(f"  Images           : {n_imgs}")
    print(f"  Total detections : {n_total}  ({avg_dets:.1f}/image)")
    print(f"  Score-boosted    : {n_adjusted}  (boundary detections with neighbour agreement)")
    print(f"  Saved → {out_path}")


if __name__ == "__main__":
    main()
