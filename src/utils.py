"""
Shared utility functions used across all pipeline scripts.

Includes:
- Config loading
- YOLO/COCO format helpers
- IoU computation
- NMS
- Box coordinate helpers
"""

import yaml
import numpy as np


def load_config(config_path: str) -> dict:
    """Load YAML config file and return as dict."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def xywhn_to_xyxy(x_center, y_center, w, h, img_w, img_h):
    """Convert normalized YOLO box to absolute xyxy."""
    x1 = (x_center - w / 2) * img_w
    y1 = (y_center - h / 2) * img_h
    x2 = (x_center + w / 2) * img_w
    y2 = (y_center + h / 2) * img_h
    return x1, y1, x2, y2


def xyxy_to_xywhn(x1, y1, x2, y2, img_w, img_h):
    """Convert absolute xyxy box to normalized YOLO format."""
    x_center = (x1 + x2) / 2 / img_w
    y_center = (y1 + y2) / 2 / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return x_center, y_center, w, h


def compute_iou(box_a, box_b):
    """Compute IoU between two boxes in xyxy format."""
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b

    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = (xa2 - xa1) * (ya2 - ya1)
    area_b = (xb2 - xb1) * (yb2 - yb1)
    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def nms(boxes, scores, iou_threshold=0.45):
    """
    Class-agnostic NMS.
    boxes: (N, 4) array of xyxy boxes
    scores: (N,) array of confidence scores
    Returns indices of kept boxes.
    """
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes, dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)

    order = scores.argsort()[::-1]
    keep = []

    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        rest = order[1:]
        ious = np.array([compute_iou(boxes[i], boxes[j]) for j in rest])
        order = rest[ious < iou_threshold]

    return keep


def class_aware_nms(boxes, scores, class_ids, iou_threshold=0.45):
    """
    NMS applied per class.
    Returns kept indices.
    """
    boxes = np.array(boxes, dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)
    class_ids = np.array(class_ids)

    keep = []
    for cls in np.unique(class_ids):
        mask = class_ids == cls
        idx = np.where(mask)[0]
        kept_local = nms(boxes[idx], scores[idx], iou_threshold)
        keep.extend(idx[kept_local].tolist())

    return sorted(keep)


def boundary_distance(x1, y1, x2, y2, tile_w, tile_h):
    """Distance from box to nearest tile edge (in pixels)."""
    return min(x1, y1, tile_w - x2, tile_h - y2)


def generate_tiles(img_w: int, img_h: int, tile_size: int, overlap: int) -> list[dict]:
    """
    Generate tile descriptors covering an image of size (img_w, img_h).

    Each tile dict has: tile_id, tile_x, tile_y, tile_w, tile_h,
                        grid_row, grid_col.

    Edge strategy: the last tile on each axis is anchored at
    (dim - tile_size) so it always reaches the image boundary.
    """
    stride = tile_size - overlap

    def axis_starts(total: int) -> list[int]:
        if total <= tile_size:
            return [0]
        starts = list(range(0, total - tile_size, stride))
        if not starts or starts[-1] + tile_size < total:
            starts.append(total - tile_size)
        return starts

    xs = axis_starts(img_w)
    ys = axis_starts(img_h)

    tiles = []
    for row, ty in enumerate(ys):
        for col, tx in enumerate(xs):
            tiles.append({
                "tile_id":  len(tiles),
                "tile_x":   tx,
                "tile_y":   ty,
                "tile_w":   min(tile_size, img_w - tx),
                "tile_h":   min(tile_size, img_h - ty),
                "grid_row": row,
                "grid_col": col,
            })
    return tiles


def gt_boundary_distance(cx: float, cy: float,
                          img_w: int, img_h: int,
                          tile_size: int, overlap: int) -> float:
    """
    Minimum distance from a GT box centre (cx, cy) to the nearest
    tile boundary line, given the tile grid for this image.
    Returns a large value (min image dimension) if there are no interior
    boundaries (i.e. the image fits in a single tile).
    """
    tiles = generate_tiles(img_w, img_h, tile_size, overlap)
    if len(tiles) == 1:
        return float(min(img_w, img_h))

    # collect unique interior boundary positions on each axis
    x_bounds = set()
    y_bounds = set()
    for t in tiles:
        right = t["tile_x"] + t["tile_w"]
        bottom = t["tile_y"] + t["tile_h"]
        if 0 < right < img_w:
            x_bounds.add(right)
        if 0 < bottom < img_h:
            y_bounds.add(bottom)

    min_dist = float(min(img_w, img_h))
    for xb in x_bounds:
        min_dist = min(min_dist, abs(cx - xb))
    for yb in y_bounds:
        min_dist = min(min_dist, abs(cy - yb))
    return min_dist
