"""
Step 3: Full-image inference.

Resizes each test image to --imgsz (e.g. 640 or 1280), runs the trained
YOLO detector, then maps predictions back to original-image pixel coordinates.

Reads the trained model path from outputs/runs/best_weights.txt.

Output CSV columns:
    image_id, image_path, x1, y1, x2, y2, score, class_id, method

All coordinates are in ORIGINAL image pixels (before any resizing).

Usage:
    python src/infer_full.py --config configs/default.yaml --imgsz 640
    python src/infer_full.py --config configs/default.yaml --imgsz 1280
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_config


# ── helpers ───────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Full-image YOLO inference")
    p.add_argument("--config",   required=True)
    p.add_argument("--imgsz",    type=int, default=640,
                   help="Inference image size (640 or 1280)")
    p.add_argument("--weights",  default=None,
                   help="Override model weights path (default: read best_weights.txt)")
    p.add_argument("--split",    default="test",
                   help="Dataset split to run inference on (default: test)")
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
        raise FileNotFoundError(f"Weights file listed in best_weights.txt not found: {weights}")
    return weights


def collect_test_images(processed_dir: Path, split: str) -> list[Path]:
    img_dir = processed_dir / "images" / split
    if not img_dir.exists():
        raise FileNotFoundError(
            f"Image directory not found: {img_dir}\n"
            "Run prepare_dataset.py first."
        )
    images = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    if not images:
        raise RuntimeError(f"No images found in {img_dir}")
    return images


def build_image_id_map(processed_dir: Path) -> dict[str, int]:
    """Return {filename → image_id} from metadata.csv."""
    meta_path = processed_dir / "metadata.csv"
    if not meta_path.exists():
        return {}
    meta = pd.read_csv(meta_path)
    return dict(zip(meta["file_name"], meta["image_id"]))


# ── inference ─────────────────────────────────────────────────────────────────

def run_inference(model, img_path: Path, imgsz: int, conf: float, iou: float,
                  device) -> list[dict]:
    """
    Run full-image inference on one image.
    Returns list of dicts with x1,y1,x2,y2 in original pixel coordinates.
    Ultralytics predict() already returns boxes in original-image coordinates
    when imgsz is passed — the internal rescaling is handled by the engine.
    """
    results = model.predict(
        source  = str(img_path),
        imgsz   = imgsz,
        conf    = conf,
        iou     = iou,
        device  = device,
        verbose = False,
    )

    detections = []
    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue
        boxes_xyxy = r.boxes.xyxy.cpu().numpy()   # (N, 4) — original pixel coords
        scores     = r.boxes.conf.cpu().numpy()   # (N,)
        class_ids  = r.boxes.cls.cpu().numpy().astype(int)  # (N,)

        for (x1, y1, x2, y2), score, cls_id in zip(boxes_xyxy, scores, class_ids):
            detections.append({
                "x1": float(x1), "y1": float(y1),
                "x2": float(x2), "y2": float(y2),
                "score": float(score),
                "class_id": int(cls_id),
            })
    return detections


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
    conf       = tiling_cfg["conf_threshold"]
    iou        = tiling_cfg["iou_threshold"]
    device     = cfg["model"]["device"]
    imgsz      = args.imgsz
    method     = f"full_{imgsz}"
    out_csv    = pred_dir / f"{method}.csv"

    weights = load_weights(runs_dir, args.weights)
    images  = collect_test_images(processed_dir, args.split)
    id_map  = build_image_id_map(processed_dir)

    print(f"Model   : {weights}")
    print(f"Split   : {args.split}  ({len(images)} images)")
    print(f"Imgsz   : {imgsz}")
    print(f"Conf    : {conf}   IoU : {iou}")
    print(f"Output  : {out_csv}")
    print()

    from ultralytics import YOLO
    model = YOLO(str(weights))

    rows = []
    total_time = 0.0

    for img_path in images:
        t0 = time.perf_counter()
        dets = run_inference(model, img_path, imgsz, conf, iou, device)
        elapsed = time.perf_counter() - t0
        total_time += elapsed

        img_id = id_map.get(img_path.name, -1)

        if not dets:
            # write one empty row so the image is represented in the CSV
            rows.append({
                "image_id": img_id, "image_path": str(img_path),
                "x1": None, "y1": None, "x2": None, "y2": None,
                "score": None, "class_id": None,
                "method": method, "infer_time_s": round(elapsed, 4),
            })
            continue

        for d in dets:
            rows.append({
                "image_id":     img_id,
                "image_path":   str(img_path),
                "x1":           round(d["x1"], 2),
                "y1":           round(d["y1"], 2),
                "x2":           round(d["x2"], 2),
                "y2":           round(d["y2"], 2),
                "score":        round(d["score"], 4),
                "class_id":     d["class_id"],
                "method":       method,
                "infer_time_s": round(elapsed, 4),
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    n_imgs  = len(images)
    n_dets  = int(df["score"].notna().sum())
    avg_t   = total_time / n_imgs
    avg_det = n_dets / n_imgs

    print(f"Done.")
    print(f"  Images processed : {n_imgs}")
    print(f"  Total detections : {n_dets}  ({avg_det:.1f} per image)")
    print(f"  Avg time/image   : {avg_t*1000:.1f} ms")
    print(f"  Saved → {out_csv}")


if __name__ == "__main__":
    main()
