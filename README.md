# From Full Boards to Tiny Defects
### Resolution Collapse and Topology-Aware Tiling for High-Resolution PCB Defect Detection


---

## Research Idea

Modern PCB defect detectors resize full board images to fixed inputs (e.g. 640×640),
causing **resolution collapse** — defects shrink to a few pixels at the wrong scale
and are missed.  Tile-based inference runs the detector on native-resolution crops,
recovering local detail, but introduces **boundary artefacts**: defects split across
tile borders cause missed detections, duplicates, or false positives.

This codebase provides a complete, reproducible comparison of five strategies:

| Strategy | Key idea |
|---|---|
| **Full-640** | Resize full board to 640 px — fast, loses fine detail |
| **Full-1280** | Resize full board to 1280 px — slower, recovers some detail |
| **Tile-640 + NMS** | 640 px crops → per-tile inference → global NMS |
| **Tile-640 + Overlap + NMS** | Same with 128 px overlap — fixes boundary recall |
| **Tile-640 + Overlap + TA-TM** | Topology-Aware Tile Merging on top of overlap tiles |

The contribution is the **inspection strategy and analysis**, not a new detector
architecture.  The same YOLO12n model is used for every comparison; only the
inference strategy changes.

**Key experimental results (test set, 35 images, 267 GT boxes):**

| Method | mAP@50 | Recall | Small recall | Bdry recall (0–16 px) | Time/img |
|---|---:|---:|---:|---:|---:|
| Full-640 | 0.072 | 0.094 | 0.000 | 0.130 | 94 ms |
| Full-1280 | 0.644 | 0.648 | 0.000 | 0.652 | 98 ms |
| Tile-640 + NMS | 0.707 | 0.757 | 0.462 | 0.261 | 330 ms |
| Tile-640 + Ov + NMS | 0.720 | 0.783 | 0.462 | **0.696** | 446 ms |
| Tile-640 + Ov + TA-TM | **0.722** | **0.783** | **0.462** | **0.696** | 459 ms |

> The model is trained on 640 px tile crops, so full-image inference suffers from
> scale mismatch (defects appear too small). Tile inference matches the training
> scale and recovers small defects. Overlap is the critical fix for boundary recall.
> TA-TM provides an additional consistent improvement.



---

## Dataset

This codebase evaluates two independent high-resolution PCB datasets:

### PCB-Defect (Primary Dataset)
230 high-resolution PCB images with COCO JSON annotations.

| Property | Value |
|---|---|
| Total images | 230 |
| Annotations | 1 704 |
| Classes | 6: missing_pad, mouse_bite, open_circuit, short, spur, spurious_copper |
| Resolution | 1 500 – 6 000 px (variable, all high-res) |
| Split (seed 42) | 161 train / 34 val / 35 test |
| Defect size (native) | 340 – 489 k px²;  median ≈ 7 k px² |

**Download:** https://github.com/Ixiaohuihuihui/Tiny-Defect-Detection-for-PCB

### HRIPCB (Validation Dataset)
Independent high-resolution PCB dataset for cross-dataset validation.

**Download:** https://www.kaggle.com/datasets/youssefhassan12/hripcb-dataset?resource=download

---

## Environment Setup

```bash
conda create -n pcb_tiling python=3.11 -y
conda activate pcb_tiling
pip install -r requirements.txt

# If PyTorch installs a CUDA version newer than your driver supports,
# downgrade (example for driver ≤ 550.x / CUDA 12.4):
pip install "torch==2.6.0+cu124" "torchvision==0.21.0+cu124" \
    --index-url https://download.pytorch.org/whl/cu124
```

All commands below assume the `pcb_tiling` environment is active.

---

## Step-by-Step Guide

### Step 0 — Repository Skeleton *(already done)*

All folders, `configs/default.yaml`, `requirements.txt`, and script stubs are
created.  Edit `configs/default.yaml` to change any hyperparameter — do not
hard-code values in scripts.

---

### Step 1 — Dataset Preprocessing

Reads the COCO JSON, remaps category IDs, splits 70/15/15, converts boxes to
YOLO format, writes `dataset.yaml` and `metadata.csv`.

**YOLO class mapping** (COCO ID 0 is supercategory — skipped):

| COCO id | Class | YOLO id |
|---|---|---|
| 1 | missing_pad | 0 |
| 2 | mouse_bite | 1 |
| 3 | open_circuit | 2 |
| 4 | short | 3 |
| 5 | spur | 4 |
| 6 | spurious_copper | 5 |

```bash
python src/prepare_dataset.py --config configs/default.yaml

# Verify (saves annotated sample images + stats chart)
python src/verify_dataset.py --config configs/default.yaml --samples 4
```

Expected terminal output:
```
Split: 161 train / 34 val / 35 test
train: 161 images, 161 label files  — All labels valid.
val  :  34 images,  34 label files  — All labels valid.
test :  35 images,  35 label files  — All labels valid.
```

Verification figures → `outputs/figures/verify/`

---

### Step 1c — Tile-Aware Training Data

**Why this step is necessary:** A model trained on full images resized to 640
sees defects at ≈20 px.  At tile inference time the same defects appear at their
native ≈80 px — a 4× scale mismatch that collapses detection performance
(mAP 0.70 → 0.01).  Training on 640×640 tile crops fixes this.

Cuts every train/val image into 640×640 crops (overlap=128 px for good coverage
of boundary defects), re-labels each crop, and writes `data/tiles/dataset.yaml`.

```bash
python src/prepare_tiles.py --config configs/default.yaml
```

Expected output:
```
train: 5293 tiles  (1781 positive = 33.6%,  3512 background = 66.4%)
val  : 1263 tiles  ( 366 positive = 29.0%,   897 background = 71.0%)
```

Key config knobs (`configs/default.yaml → tile_training`):

| Key | Default | Effect |
|---|---|---|
| `tile_size` | 640 | Crop size in pixels |
| `overlap` | 128 | Overlap between adjacent crops |
| `min_visibility` | 0.4 | Min GT box area fraction to include in crop |

---

### Step 2 — Train Detector

Trains **YOLO12n** on the tile dataset via Ultralytics.  Writes
`outputs/runs/best_weights.txt` — every downstream script reads this pointer.

```bash
# List all supported model families
python src/train_detector.py --config configs/default.yaml --list-models

# Train on tile dataset (recommended — what the paper uses)
python src/train_detector.py --config configs/default.yaml \
    --model yolo12n \
    --data data/tiles/dataset.yaml \
    --name train_tiles \
    --batch 16

# Resume an interrupted run
python src/train_detector.py --config configs/default.yaml \
    --data data/tiles/dataset.yaml --name train_tiles --resume
```

Supported model families (pass bare name, e.g. `yolo12n`):

| Family | Nano | Small | Notes |
|---|---|---|---|
| YOLO12 | `yolo12n` | `yolo12s` | Newest — used in this paper |
| YOLO11 | `yolo11n` | `yolo11s` | Latest stable |
| YOLOv9 | `yolov9t` | `yolov9s` | Programmable gradient info |
| YOLOv8 | `yolov8n` | `yolov8s` | Well-established baseline |

After training completes, `outputs/runs/best_weights.txt` points to `best.pt`
and all subsequent steps use it automatically.

---

### Step 3 — Full-Image Inference

Runs the detector on test images resized to a fixed input size.
Boxes are returned in **original image pixel coordinates**.

```bash
python src/infer_full.py --config configs/default.yaml --imgsz 640
python src/infer_full.py --config configs/default.yaml --imgsz 1280
```

Outputs: `outputs/predictions/full_640.csv`, `full_1280.csv`

CSV columns: `image_id, image_path, x1, y1, x2, y2, score, class_id, method, infer_time_s`

---

### Step 4+5 — Tile Inference + NMS

Splits each test image into tiles, runs the detector on each, remaps boxes to
global coordinates, and applies class-aware NMS.  Writes **two** CSVs per run:

- `tile_*_overlap*.csv` — raw per-tile detections with tile metadata and
  `boundary_distance` (used as TA-TM input in Step 6)
- `tile_*_overlap*_nms.csv` — NMS-merged result (used by Step 7 evaluate)

```bash
# No overlap
python src/infer_tiles.py --config configs/default.yaml --tile-size 640 --overlap 0

# 128 px overlap (recommended for boundary recall)
python src/infer_tiles.py --config configs/default.yaml --tile-size 640 --overlap 128
```

Raw CSV columns:
```
image_id, image_path, tile_id, tile_x, tile_y, tile_w, tile_h,
grid_row, grid_col, x1, y1, x2, y2, score, class_id, method,
boundary_distance, infer_time_s
```

`boundary_distance = min(x1_local, y1_local, tile_w−x2_local, tile_h−y2_local)`

---

### Step 6 — Topology-Aware Tile Merging (TA-TM)

Reads the **raw** tile CSV, adjusts confidence scores for boundary-sensitive
detections using neighbour-tile agreement, then applies global NMS.

**Algorithm:**
1. A detection is *boundary-sensitive* if `boundary_distance < τ` (default τ=16 px).
2. For each boundary-sensitive box, find same-class detections in the adjacent tile
   that are also near the shared edge and overlap in the perpendicular axis.
3. Adjusted score: `s' = min(1.0, s + λ·A + μ·C)` where A is the max neighbour
   confidence, C is an optional edge-continuity proxy (off by default), λ=0.2, μ=0.
4. Apply global class-aware NMS using adjusted scores.

```bash
# Default (τ=16, λ=0.2)
python src/topology_merge.py --config configs/default.yaml \
    --pred outputs/predictions/tile_640_overlap0.csv

python src/topology_merge.py --config configs/default.yaml \
    --pred outputs/predictions/tile_640_overlap128.csv

# Tune parameters without editing config
python src/topology_merge.py --config configs/default.yaml \
    --pred outputs/predictions/tile_640_overlap128.csv \
    --tau 32 --lambda 0.4 --suffix _tau32
```

Output: `outputs/predictions/tatm_<input_stem>[suffix].csv`

Output CSV columns:
```
image_id, image_path, x1, y1, x2, y2, score_orig, score_adj,
class_id, method, boundary_distance, infer_time_s
```

---

### Step 7 — Evaluation

Matches predictions to GT at IoU ≥ 0.5 per class and computes all metrics.
Results are **appended** to the table CSVs — run once per method, all methods
accumulate in the same files.

```bash
python src/evaluate.py --config configs/default.yaml \
    --pred outputs/predictions/full_640.csv --method full_640

python src/evaluate.py --config configs/default.yaml \
    --pred outputs/predictions/full_1280.csv --method full_1280

python src/evaluate.py --config configs/default.yaml \
    --pred outputs/predictions/tile_640_overlap0_nms.csv --method tile_640

python src/evaluate.py --config configs/default.yaml \
    --pred outputs/predictions/tile_640_overlap128_nms.csv --method tile_640_ov128

python src/evaluate.py --config configs/default.yaml \
    --pred outputs/predictions/tatm_tile_640_overlap128.csv --method tatm_640_ov128
```

Output tables:

| File | Paper table | Contents |
|---|---|---|
| `outputs/tables/main_results.csv` | Table 1 | mAP@50, precision, recall, small recall, boundary recall, time |
| `outputs/tables/size_recall.csv` | Table 2 | Recall grouped by GT area at 640 px input |
| `outputs/tables/boundary_recall.csv` | Table 3 | Recall grouped by GT distance to nearest tile edge |

---

### Step 8 — Resolution-Collapse Analysis

Quantifies how much each defect shrinks when the full image is resized to 640
or 1280, links each GT box to its detection outcome per method, and generates
three publication-ready figures.

```bash
python src/analyze_resolution.py --config configs/default.yaml
```

Outputs in `outputs/figures/resolution/`:

| Figure | Description |
|---|---|
| `collapse_histogram.png` | GT area distribution at native / 640 / 1280 scales |
| `recall_by_area.png` | Recall per defect-size bin across all methods |
| `collapse_examples.png` | Side-by-side crops: native vs 640-scale appearance |

Terminal output includes a recall table broken down by area bin at 640 px input:

```
Method                        0–4     4–16    16–64      >64
-------------------------------------------------------------------
Full-640                       —        —     0.0000   0.0984
Full-1280                      —        —     0.0000   0.6811
Tile-640 + NMS                 —        —     0.4615   0.7717
Tile-640 + Ov + NMS            —        —     0.4615   0.7992
Tile-640 + Ov + TA-TM          —        —     0.4615   0.7992
```

---

### Step 9 — Visualization

Generates all paper figures: per-image TP/FP/FN overlays for every method,
zoomed tile-boundary comparisons, and a summary bar chart.

```bash
python src/visualize.py --config configs/default.yaml --n-images 4
```

| Flag | Default | Description |
|---|---|---|
| `--n-images` | 4 | Number of test images to visualize |
| `--max-side` | 1200 | Max pixel dimension of saved images |

Outputs:

| Location | Description |
|---|---|
| `outputs/figures/comparison_grid/*.jpg` | 2×3 grid: GT + 5 methods with TP/FP/FN colours |
| `outputs/figures/boundary_artifact/*.jpg` | Zoomed tile-boundary region, NMS vs TA-TM |
| `outputs/figures/summary_chart.png` | Bar chart: all key metrics across all methods |

Colour coding in comparison grids:

| Colour | Meaning |
|---|---|
| Green | Ground truth box |
| Orange | True positive (TP) |
| Red | False positive (FP) |
| Yellow | False negative / missed GT (FN) |

---

## Full Reproduction Commands

Run these in order from the repo root with the `pcb_tiling` environment active:

```bash
# ── dataset ──────────────────────────────────────────────────────────
python src/prepare_dataset.py  --config configs/default.yaml
python src/verify_dataset.py   --config configs/default.yaml --samples 4
python src/prepare_tiles.py    --config configs/default.yaml

# ── training ─────────────────────────────────────────────────────────
python src/train_detector.py   --config configs/default.yaml \
    --model yolo12n --data data/tiles/dataset.yaml \
    --name train_tiles --batch 16

# ── inference ─────────────────────────────────────────────────────────
python src/infer_full.py       --config configs/default.yaml --imgsz 640
python src/infer_full.py       --config configs/default.yaml --imgsz 1280
python src/infer_tiles.py      --config configs/default.yaml --tile-size 640 --overlap 0
python src/infer_tiles.py      --config configs/default.yaml --tile-size 640 --overlap 128
python src/topology_merge.py   --config configs/default.yaml \
    --pred outputs/predictions/tile_640_overlap0.csv
python src/topology_merge.py   --config configs/default.yaml \
    --pred outputs/predictions/tile_640_overlap128.csv

# ── evaluation ────────────────────────────────────────────────────────
python src/evaluate.py --config configs/default.yaml \
    --pred outputs/predictions/full_640.csv          --method full_640
python src/evaluate.py --config configs/default.yaml \
    --pred outputs/predictions/full_1280.csv         --method full_1280
python src/evaluate.py --config configs/default.yaml \
    --pred outputs/predictions/tile_640_overlap0_nms.csv    --method tile_640
python src/evaluate.py --config configs/default.yaml \
    --pred outputs/predictions/tile_640_overlap128_nms.csv  --method tile_640_ov128
python src/evaluate.py --config configs/default.yaml \
    --pred outputs/predictions/tatm_tile_640_overlap128.csv --method tatm_640_ov128

# ── analysis & figures ────────────────────────────────────────────────
python src/analyze_resolution.py --config configs/default.yaml
python src/visualize.py          --config configs/default.yaml --n-images 4
```

---

## Output Files Reference

### Prediction CSVs

**Full-image** (`full_640.csv`, `full_1280.csv`):
```
image_id, image_path, x1, y1, x2, y2, score, class_id, method, infer_time_s
```

**Raw tile** (`tile_640_overlap*.csv`) — input to TA-TM:
```
image_id, image_path, tile_id, tile_x, tile_y, tile_w, tile_h,
grid_row, grid_col, x1, y1, x2, y2, score, class_id, method,
boundary_distance, infer_time_s
```

**NMS-merged tile** (`tile_640_overlap*_nms.csv`) — input to evaluate:
```
image_id, image_path, x1, y1, x2, y2, score, class_id, method,
boundary_distance, infer_time_s
```

**TA-TM output** (`tatm_tile_640_overlap*.csv`):
```
image_id, image_path, x1, y1, x2, y2, score_orig, score_adj,
class_id, method, boundary_distance, infer_time_s
```

All `(x1, y1, x2, y2)` coordinates are in **original image pixels**.

---

## Method Details

### TA-TM Score Adjustment

For a detection `b = [x1, y1, x2, y2]` in tile of size `W_t × H_t`:

```
boundary_distance(b) = min(x1_local, y1_local, W_t − x2_local, H_t − y2_local)
```

A detection is *boundary-sensitive* if `boundary_distance < τ` (τ = 16 px default).

For each boundary-sensitive detection, the score is adjusted by:

```
s' = min(1.0,  s  +  λ·A  +  μ·C)
```

where:
- `A` = max confidence of same-class detections in the adjacent tile that are
  also near the shared boundary and overlap in the perpendicular axis
- `C` = optional edge-continuity proxy (disabled by default, μ = 0)
- `λ = 0.2`, `μ = 0.0`  (configurable via `--lambda` / `--tau` CLI flags)

Global class-aware NMS is applied **after** score adjustment using adjusted scores.
No additional training. No learned parameters.

### Key Config Sections

```yaml
tile_training:
  tile_size: 640        # crop size for training tiles
  overlap: 128          # overlap between training crops
  min_visibility: 0.4   # min GT box visibility fraction to include

tiling:
  tile_size: 640        # tile size at inference
  overlap: 0            # default inference overlap (overridden by --overlap)
  boundary_tau: 16      # boundary-sensitivity threshold (px)
  conf_threshold: 0.25  # YOLO confidence threshold
  iou_threshold: 0.45   # NMS IoU threshold

topology:
  lambda_adj: 0.2       # adjacent-agreement weight
  mu_edge: 0.0          # edge-continuity weight (disabled)
  use_edge_continuity: false
```
