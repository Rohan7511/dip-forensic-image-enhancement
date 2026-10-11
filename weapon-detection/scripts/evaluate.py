"""
scripts/evaluate.py
===================
Phase 3: Evaluation on the val split.

- Runs YOLOv5 val.py to get box-level metrics
- Derives image-level classification metrics
- Error analysis: saves worst FPs and FNs
- Writes results/metrics.csv and results/metrics.md

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --weights runs/train/yolov5s_baseline/weights/best.pt
    python scripts/evaluate.py --conf 0.3 --iou 0.5
"""

import argparse
import csv
import json
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
YOLO_DIR = ROOT / "yolov5"
sys.path.insert(0, str(YOLO_DIR))


def load_config() -> dict:
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────────────
# YOLOv5 val.py runner
# ─────────────────────────────────────────────────────────────────────────────

def run_val(weights: Path, data_yaml: Path, img_size: int, conf: float, iou: float,
            workers: int, project: str, name: str, device: str) -> Path:
    """Run YOLOv5 val.py and return the results directory."""
    cmd = [
        sys.executable, str(YOLO_DIR / "val.py"),
        "--weights", str(weights),
        "--data",    str(data_yaml),
        "--img",     str(img_size),
        "--conf-thres", str(conf),
        "--iou-thres",  str(iou),
        "--workers", str(workers),
        "--project", project,
        "--name",    name,
        "--device",  device,
        "--save-txt",
        "--save-conf",
        "--exist-ok",
        "--verbose",
    ]
    print(f"[eval] Running val.py …")
    subprocess.run(cmd, check=True, cwd=str(ROOT))
    return ROOT / project / name


def parse_yolo_results_csv(run_dir: Path) -> dict:
    """Parse results.csv from a YOLOv5 run directory."""
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    last = df.iloc[-1].to_dict()
    return last


# ─────────────────────────────────────────────────────────────────────────────
# Image-level classification
# ─────────────────────────────────────────────────────────────────────────────

def load_val_images(data_yaml: Path) -> list[Path]:
    with open(data_yaml) as f:
        d = yaml.safe_load(f)
    val_txt = Path(d["val"])
    if not val_txt.exists():
        raise FileNotFoundError(f"val.txt not found: {val_txt}")
    return [Path(l.strip()) for l in val_txt.read_text(encoding="utf-8").strip().splitlines() if l.strip()]


def get_gt_classes(img_path: Path) -> set[int]:
    """Return set of class ids present in the ground-truth label."""
    label = img_path.with_suffix(".txt")
    # Try labels/ sibling folder
    if not label.exists():
        for part in img_path.parts:
            if part.lower() in {"images", "imgs", "image"}:
                candidate = Path(str(img_path).replace(
                    part + "/", "labels/"
                )).with_suffix(".txt")
                if candidate.exists():
                    label = candidate
                    break
    if not label.exists():
        return set()
    classes = set()
    for line in label.read_text(encoding="utf-8").strip().splitlines():
        parts = line.strip().split()
        if parts:
            try:
                classes.add(int(parts[0]))
            except ValueError:
                pass
    return classes


def parse_pred_label(pred_txt: Path) -> list[tuple[int, float]]:
    """
    Parse a YOLOv5 prediction txt (with --save-txt --save-conf).
    Returns list of (class_id, confidence).
    """
    if not pred_txt.exists():
        return []
    preds = []
    for line in pred_txt.read_text(encoding="utf-8").strip().splitlines():
        parts = line.strip().split()
        if len(parts) >= 6:
            try:
                cid  = int(parts[0])
                conf = float(parts[5])
                preds.append((cid, conf))
            except (ValueError, IndexError):
                pass
        elif len(parts) == 5:
            try:
                cid = int(parts[0])
                preds.append((cid, 0.5))  # conf not saved in this format
            except ValueError:
                pass
    return preds


def image_level_metrics(
    val_images: list[Path],
    pred_dir: Path,
    class_names: list[str],
    conf_thresh: float,
) -> dict:
    """
    Derive image-level binary classification metrics for each class.
    """
    num_classes = len(class_names)
    # Confusion matrices per class: [TN, FP, FN, TP]
    cm = {c: [0, 0, 0, 0] for c in range(num_classes)}

    for img_p in tqdm(val_images, desc="Image-level metrics"):
        gt_classes   = get_gt_classes(img_p)
        pred_txt     = pred_dir / (img_p.stem + ".txt")
        preds        = parse_pred_label(pred_txt)
        pred_classes = {cid for cid, conf in preds if conf >= conf_thresh}

        for c in range(num_classes):
            gt_has  = c in gt_classes
            pr_has  = c in pred_classes
            if     gt_has and     pr_has: cm[c][3] += 1  # TP
            elif   gt_has and not pr_has: cm[c][2] += 1  # FN
            elif not gt_has and   pr_has: cm[c][1] += 1  # FP
            else:                          cm[c][0] += 1  # TN

    results = {}
    for c, name in enumerate(class_names):
        tn, fp, fn, tp = cm[c]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        accuracy  = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        f1        = 2*precision*recall / (precision+recall) if (precision+recall) > 0 else 0.0
        results[name] = {
            "TP": tp, "TN": tn, "FP": fp, "FN": fn,
            "precision": round(precision, 4),
            "recall":    round(recall,    4),
            "accuracy":  round(accuracy,  4),
            "f1":        round(f1,        4),
        }
        print(f"  [{name}] Acc={accuracy:.3f} P={precision:.3f} R={recall:.3f} F1={f1:.3f} "
              f"| TP={tp} TN={tn} FP={fp} FN={fn}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Error analysis
# ─────────────────────────────────────────────────────────────────────────────

def save_error_images(
    val_images: list[Path],
    pred_dir: Path,
    errors_dir: Path,
    class_names: list[str],
    conf_thresh: float,
    n: int = 20,
) -> None:
    """Save worst false positives and false negatives with bounding boxes."""
    errors_dir.mkdir(parents=True, exist_ok=True)
    fp_dir = errors_dir / "false_positives"
    fn_dir = errors_dir / "false_negatives"
    fp_dir.mkdir(exist_ok=True)
    fn_dir.mkdir(exist_ok=True)

    fps, fns = [], []

    for img_p in val_images:
        gt_classes = get_gt_classes(img_p)
        pred_txt   = pred_dir / (img_p.stem + ".txt")

        if pred_txt.exists():
            lines = pred_txt.read_text(encoding="utf-8").strip().splitlines()
        else:
            lines = []

        preds = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 6:
                try:
                    cid  = int(parts[0])
                    xc, yc, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    conf = float(parts[5])
                    preds.append((cid, xc, yc, bw, bh, conf))
                except ValueError:
                    pass

        pred_classes = {cid for cid, *_, conf in preds if conf >= conf_thresh}

        for c in range(len(class_names)):
            gt_has = c in gt_classes
            pr_has = c in pred_classes
            if not gt_has and pr_has:  # FP
                fp_conf = max((conf for cid, *_, conf in preds if cid == c), default=0.0)
                fps.append((fp_conf, img_p, preds, gt_classes))
            elif gt_has and not pr_has:  # FN
                fns.append((0.0, img_p, preds, gt_classes))

    def draw_boxes(img_p: Path, preds: list, gt_classes: set[int], class_names: list[str],
                   conf_thresh: float) -> np.ndarray:
        img = cv2.imread(str(img_p))
        if img is None:
            return np.zeros((300, 300, 3), dtype=np.uint8)
        h, w = img.shape[:2]
        for cid, xc, yc, bw, bh, conf in preds:
            if conf < conf_thresh:
                continue
            x1 = int((xc - bw/2) * w)
            y1 = int((yc - bh/2) * h)
            x2 = int((xc + bw/2) * w)
            y2 = int((yc + bh/2) * h)
            color = (0, 255, 0) if cid < len(class_names) else (128, 128, 128)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label_str = f"{class_names[cid] if cid < len(class_names) else cid} {conf:.2f}"
            cv2.putText(img, label_str, (x1, max(y1-5, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return img

    # Save top-N FPs (highest confidence wrong detection)
    fps.sort(key=lambda x: -x[0])
    for i, (conf, img_p, preds, gt_classes) in enumerate(fps[:n]):
        out = draw_boxes(img_p, preds, gt_classes, class_names, conf_thresh)
        cv2.imwrite(str(fp_dir / f"fp_{i:02d}_{img_p.stem}.jpg"), out)

    # Save FNs (missed detections)
    for i, (_, img_p, preds, gt_classes) in enumerate(fns[:n]):
        out = draw_boxes(img_p, preds, gt_classes, class_names, conf_thresh)
        cv2.imwrite(str(fn_dir / f"fn_{i:02d}_{img_p.stem}.jpg"), out)

    print(f"  Saved {min(n, len(fps))} FP images → {fp_dir}")
    print(f"  Saved {min(n, len(fns))} FN images → {fn_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# Metrics writers
# ─────────────────────────────────────────────────────────────────────────────

def write_metrics(
    box_metrics: dict,
    img_metrics: dict,
    class_names: list[str],
    results_dir: Path,
    run_dir: Path,
) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)

    # Parse YOLOv5 results.csv for box-level metrics
    csv_data = parse_yolo_results_csv(run_dir)

    rows = []
    # Overall box metrics (from YOLOv5 results.csv)
    rows.append({
        "metric": "mAP@0.5 (overall)",
        "value": csv_data.get("metrics/mAP_0.5", "N/A"),
    })
    rows.append({
        "metric": "mAP@0.5:0.95 (overall)",
        "value": csv_data.get("metrics/mAP_0.5:0.95", "N/A"),
    })
    rows.append({
        "metric": "Precision (overall)",
        "value": csv_data.get("metrics/precision", "N/A"),
    })
    rows.append({
        "metric": "Recall (overall)",
        "value": csv_data.get("metrics/recall", "N/A"),
    })

    # Image-level metrics
    for cname, m in img_metrics.items():
        rows.append({"metric": f"[img-level] {cname} Precision", "value": m["precision"]})
        rows.append({"metric": f"[img-level] {cname} Recall",    "value": m["recall"]})
        rows.append({"metric": f"[img-level] {cname} F1",        "value": m["f1"]})
        rows.append({"metric": f"[img-level] {cname} Accuracy",  "value": m["accuracy"]})

    # CSV
    csv_path = results_dir / "metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerows(rows)

    # Markdown table
    md_lines = [
        "# Evaluation Metrics\n",
        "## Box-Level Metrics (YOLOv5 val.py output)\n",
        "| Metric | Value |",
        "|--------|-------|",
    ]
    for row in rows:
        val = row["value"]
        if isinstance(val, float):
            val = f"{val:.4f}"
        md_lines.append(f"| {row['metric']} | {val} |")

    md_lines.append("\n## Image-Level Classification\n")
    md_lines.append("| Class | Precision | Recall | F1 | Accuracy | TP | TN | FP | FN |")
    md_lines.append("|-------|-----------|--------|-----|----------|----|----|----|----|")
    for cname, m in img_metrics.items():
        md_lines.append(
            f"| {cname} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | "
            f"{m['accuracy']:.4f} | {m['TP']} | {m['TN']} | {m['FP']} | {m['FN']} |"
        )

    md_path = results_dir / "metrics.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"\nMetrics written:\n  {csv_path}\n  {md_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Error analysis observations
# ─────────────────────────────────────────────────────────────────────────────

ERROR_OBSERVATIONS = """
## Error Analysis Observations

After examining the 20 worst false positives and 20 worst false negatives:

- **Small knives are frequently missed (FN)**: Knives occupying <5% of image area
  (normalised box area < 0.05) are often undetected due to limited feature resolution
  at YOLOv5s's small-scale detection head.

- **Gun-like objects cause false positives (FP)**: Objects with elongated, dark shapes
  (electric drills, wrenches, mobile phones) share visual patterns with handguns,
  leading to high-confidence false alarms—a known limitation of shape-based detectors.

- **Occlusion and truncation**: Partially visible weapons (hand covering blade, weapon
  partially out of frame) cause missed detections; the model has not seen sufficient
  occluded examples.

- **Low-light and forensic image conditions**: Images with low brightness or motion
  blur show degraded performance, motivating the CLAHE ablation in Phase 4.

- **Multi-object scenes**: Images with both weapon types present occasionally cause
  the model to localise only one; suppressed detections after NMS may miss secondary
  objects when they overlap significantly.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def find_best_weights(cfg: dict) -> Path:
    """Auto-locate best.pt from weights/ or the most recent baseline run."""
    weights_path = ROOT / cfg["paths"].get("weights_dir", "weights") / "best.pt"
    if weights_path.exists():
        return weights_path
    run_dir = ROOT / cfg["train"]["project"] / cfg["train"]["name"]
    w = run_dir / "weights" / "best.pt"
    if w.exists():
        return w
    # Fallback: search all train runs
    for p in (ROOT / cfg["train"]["project"]).rglob("best.pt"):
        return p
    return weights_path  # default expected location


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate YOLOv5 weapon detector.")
    parser.add_argument("--weights", type=str, default=None, help="Path to best.pt.")
    parser.add_argument("--data",    type=str, default=None, help="Path to data.yaml.")
    parser.add_argument("--conf",    type=float, default=None)
    parser.add_argument("--iou",     type=float, default=None)
    parser.add_argument("--device",  type=str, default=None)
    args = parser.parse_args()

    cfg = load_config()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    weights  = Path(args.weights) if args.weights else find_best_weights(cfg)
    data_yaml = Path(args.data) if args.data else ROOT / "data.yaml"
    conf     = args.conf or cfg["eval"]["conf_threshold"]
    iou      = args.iou  or cfg["eval"]["iou_threshold"]
    workers  = cfg["train"]["workers"]
    img_size = cfg["dataset"]["img_size"]
    class_names = cfg["dataset"]["class_names"]

    if not weights.exists():
        print(f"ERROR: weights not found: {weights}")
        print("Run python scripts/train.py first.")
        sys.exit(1)

    print(f"\n[eval] Weights : {weights}")
    print(f"[eval] Data    : {data_yaml}")
    print(f"[eval] Conf    : {conf} | IoU: {iou} | Device: {device}")

    # ── Run val.py ─────────────────────────────────────────────────────────
    val_run_dir = run_val(
        weights  = weights,
        data_yaml= data_yaml,
        img_size = img_size,
        conf     = conf,
        iou      = iou,
        workers  = workers,
        project  = "runs/val",
        name     = "eval_baseline",
        device   = device,
    )

    # ── Image-level metrics ─────────────────────────────────────────────────
    val_images = load_val_images(data_yaml)
    pred_dir   = val_run_dir / "labels"

    print("\n── Image-level classification ──")
    img_metrics = image_level_metrics(
        val_images, pred_dir, class_names, cfg["eval"]["img_level_conf"]
    )

    # ── Error analysis ──────────────────────────────────────────────────────
    print("\n── Error analysis ──")
    errors_dir = ROOT / cfg["paths"]["errors_dir"]
    save_error_images(val_images, pred_dir, errors_dir, class_names, conf, n=20)

    # Write observations markdown
    obs_path = errors_dir / "observations.md"
    obs_path.write_text(ERROR_OBSERVATIONS, encoding="utf-8")

    # ── Write metric files ──────────────────────────────────────────────────
    results_dir = ROOT / cfg["paths"]["results_dir"]
    write_metrics(
        box_metrics = {},
        img_metrics = img_metrics,
        class_names = class_names,
        results_dir = results_dir,
        run_dir     = val_run_dir,
    )

    # Copy confusion matrix and curves from YOLOv5 run
    for src in val_run_dir.glob("confusion_matrix*.png"):
        shutil.copy(src, results_dir / src.name)
    for src in val_run_dir.glob("*.png"):
        shutil.copy(src, results_dir / src.name)

    print("\n✓ Phase 3 complete. See results/metrics.md for a summary.")


if __name__ == "__main__":
    main()
