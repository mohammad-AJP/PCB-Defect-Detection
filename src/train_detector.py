"""
Step 2: Train a YOLO detector on the preprocessed PCB dataset.

Supported model families (Ultralytics 8.4+):

  Family   | Nano        | Small       | Medium      | Notes
  ---------|-------------|-------------|-------------|---------------------------
  YOLO11   | yolo11n     | yolo11s     | yolo11m     | Latest stable (default)
  YOLO12   | yolo12n     | yolo12s     | yolo12m     | Newest, attention-based
  YOLOv9   | yolov9t     | yolov9s     | yolov9c     | Programmable gradient info
  YOLOv8   | yolov8n     | yolov8s     | yolov8m     | Well-established baseline

Pass the bare model name (without .pt) to --model, e.g.:
  --model yolo11n      (default)
  --model yolo11s
  --model yolo12n
  --model yolov8n

Saves all run artefacts under outputs/runs/<name>/.
Writes absolute path of best.pt to outputs/runs/best_weights.txt.

Usage:
    python src/train_detector.py --config configs/default.yaml
    python src/train_detector.py --config configs/default.yaml --model yolo12n
    python src/train_detector.py --config configs/default.yaml --model yolov8s --epochs 100
    python src/train_detector.py --config configs/default.yaml --resume
    python src/train_detector.py --config configs/default.yaml --list-models
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_config

# ── model catalogue ───────────────────────────────────────────────────────────
# Maps short name → (pt filename, description)
MODEL_CATALOGUE = {
    # YOLO11 — latest stable, C2PSA attention block
    "yolo11n": ("yolo11n.pt", "YOLO11 Nano   — fastest, smallest  [DEFAULT]"),
    "yolo11s": ("yolo11s.pt", "YOLO11 Small  — good speed/accuracy balance"),
    "yolo11m": ("yolo11m.pt", "YOLO11 Medium — higher accuracy, more VRAM"),
    "yolo11l": ("yolo11l.pt", "YOLO11 Large"),
    "yolo11x": ("yolo11x.pt", "YOLO11 XLarge — highest accuracy"),
    # YOLO12 — newest, flash-attention based
    "yolo12n": ("yolo12n.pt", "YOLO12 Nano   — newest architecture, fast"),
    "yolo12s": ("yolo12s.pt", "YOLO12 Small"),
    "yolo12m": ("yolo12m.pt", "YOLO12 Medium"),
    "yolo12l": ("yolo12l.pt", "YOLO12 Large"),
    "yolo12x": ("yolo12x.pt", "YOLO12 XLarge"),
    # YOLOv9 — programmable gradient information
    "yolov9t": ("yolov9t.pt", "YOLOv9 Tiny   — very fast, lightweight"),
    "yolov9s": ("yolov9s.pt", "YOLOv9 Small"),
    "yolov9c": ("yolov9c.pt", "YOLOv9 Compact"),
    "yolov9e": ("yolov9e.pt", "YOLOv9 Extended — best in v9 family"),
    # YOLOv8 — well-established baseline
    "yolov8n": ("yolov8n.pt", "YOLOv8 Nano   — proven baseline"),
    "yolov8s": ("yolov8s.pt", "YOLOv8 Small"),
    "yolov8m": ("yolov8m.pt", "YOLOv8 Medium"),
    "yolov8l": ("yolov8l.pt", "YOLOv8 Large"),
    "yolov8x": ("yolov8x.pt", "YOLOv8 XLarge"),
    # YOLOv10
    "yolov10n": ("yolov10n.pt", "YOLOv10 Nano  — NMS-free detection"),
    "yolov10s": ("yolov10s.pt", "YOLOv10 Small"),
    "yolov10m": ("yolov10m.pt", "YOLOv10 Medium"),
}

DEFAULT_MODEL = "yolo11n"


def list_models():
    print("\nAvailable models (pass the key to --model):\n")
    families = [("YOLO11 (latest stable)", "yolo11"),
                ("YOLO12 (newest)",        "yolo12"),
                ("YOLOv9",                 "yolov9"),
                ("YOLOv8 (baseline)",      "yolov8"),
                ("YOLOv10 (NMS-free)",     "yolov10")]
    for family_name, prefix in families:
        print(f"  {family_name}")
        for key, (_, desc) in MODEL_CATALOGUE.items():
            if key.startswith(prefix):
                print(f"    {key:<12}  {desc}")
        print()


# ── helpers ───────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Train a YOLO detector on the PCB defect dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config",       required=True)
    p.add_argument("--model",        default=None,
                   help=f"Model key (default from config, usually '{DEFAULT_MODEL}'). "
                        "Run --list-models to see all options.")
    p.add_argument("--data",         default=None,
                   help="Override dataset.yaml path (e.g. data/tiles/dataset.yaml "
                        "for tile-aware training). Default: data/processed/dataset.yaml")
    p.add_argument("--epochs",       type=int,   default=None)
    p.add_argument("--batch",        type=int,   default=None)
    p.add_argument("--imgsz",        type=int,   default=None)
    p.add_argument("--name",         default="train",
                   help="Run subfolder name inside outputs/runs/")
    p.add_argument("--resume",       action="store_true",
                   help="Resume from last.pt if it exists")
    p.add_argument("--list-models",  action="store_true",
                   help="Print all available model options and exit")
    return p.parse_args()


def resolve_weights(model_arg: str | None, config_name: str) -> tuple[str, str]:
    """
    Returns (pt_filename, short_key).
    Priority: --model CLI > config model.name > default.
    Accepts both short key ('yolo11n') and full filename ('yolo11n.pt').
    """
    raw = model_arg or config_name or DEFAULT_MODEL
    key = raw.removesuffix(".pt")   # normalise away .pt if user typed it

    if key in MODEL_CATALOGUE:
        pt, _ = MODEL_CATALOGUE[key]
        return pt, key

    # unknown name — pass through as-is (lets users supply custom .pt paths)
    print(f"  NOTE: '{raw}' not in catalogue, passing directly to Ultralytics.")
    return raw if raw.endswith(".pt") else raw + ".pt", raw


def find_last_checkpoint(run_dir: Path) -> Path | None:
    last = run_dir / "weights" / "last.pt"
    return last if last.exists() else None


def write_best_weights(run_dir: Path, pointer_file: Path):
    best = run_dir / "weights" / "best.pt"
    if not best.exists():
        print(f"WARNING: best.pt not found at {best}")
        return
    pointer_file.write_text(str(best.resolve()) + "\n")
    print(f"Best weights : {best}")
    print(f"Pointer file : {pointer_file}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if args.list_models:
        list_models()
        sys.exit(0)

    cfg       = load_config(args.config)
    repo_root = Path(args.config).parent.parent
    out_dir   = (repo_root / cfg["outputs"]["dir"] / "runs").resolve()
    run_dir   = out_dir / args.name

    # --data overrides the default processed dataset
    if args.data:
        data_yaml = (repo_root / args.data).resolve()
    else:
        data_yaml = (repo_root / cfg["dataset"]["processed_dir"] / "dataset.yaml").resolve()

    if not data_yaml.exists():
        print(f"ERROR: dataset.yaml not found: {data_yaml}")
        if not args.data:
            print("Run prepare_dataset.py first.")
        else:
            print("Run prepare_tiles.py first.")
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    model_cfg = cfg["model"]
    weights, model_key = resolve_weights(args.model, model_cfg.get("name", DEFAULT_MODEL))
    epochs  = args.epochs or model_cfg["epochs"]
    batch   = args.batch  or model_cfg["batch"]
    imgsz   = args.imgsz  or model_cfg["imgsz"]
    device  = model_cfg["device"]

    print("=" * 60)
    print("Training configuration")
    print(f"  model    : {model_key}  →  {weights}")
    print(f"  data     : {data_yaml}")
    print(f"  epochs   : {epochs}")
    print(f"  batch    : {batch}")
    print(f"  imgsz    : {imgsz}")
    print(f"  device   : {device}")
    print(f"  run dir  : {run_dir}")
    print("=" * 60)

    from ultralytics import YOLO

    if args.resume:
        last_ckpt = find_last_checkpoint(run_dir)
        if last_ckpt:
            print(f"Resuming from {last_ckpt}")
            model = YOLO(str(last_ckpt))
            model.train(resume=True)
            pointer_file = out_dir / "best_weights.txt"
            write_best_weights(run_dir, pointer_file)
            print("\nTraining complete.")
            return
        else:
            print("No last.pt found — starting fresh.")

    model = YOLO(weights)
    model.train(
        data     = str(data_yaml),
        epochs   = epochs,
        imgsz    = imgsz,
        batch    = batch,
        device   = device,
        project  = str(out_dir),
        name     = args.name,
        exist_ok = True,
        patience = 20,
        save     = True,
        plots    = True,
        val      = True,
    )

    pointer_file = out_dir / "best_weights.txt"
    write_best_weights(run_dir, pointer_file)
    print("\nTraining complete.")


if __name__ == "__main__":
    main()
