"""
scripts/predict.py
==================
Phase 6: CLI inference — annotate a single image or a folder of images.

Usage:
    python scripts/predict.py --source data/raw/test.jpg
    python scripts/predict.py --source data/raw/ --conf 0.35 --iou 0.5
    python scripts/predict.py --source image.jpg --weights runs/train/yolov5s_baseline/weights/best.pt
    python scripts/predict.py --source folder/ --output results/predictions/
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
YOLO_DIR = ROOT / "yolov5"
sys.path.insert(0, str(YOLO_DIR))


def load_config() -> dict:
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Inference helpers
# ─────────────────────────────────────────────────────────────────────────────

COLORS = {
    0: (0,   200, 80),   # handgun — green
    1: (0,   120, 255),  # knife   — orange
}


def load_model(weights: Path, device: str):
    """Load YOLOv5 model via torch.hub (local)."""
    model = torch.hub.load(
        str(YOLO_DIR), "custom", path=str(weights),
        source="local", force_reload=False, verbose=False,
    )
    model.to(device)
    model.conf = 0.01   # set low; we filter ourselves
    model.iou  = 0.45
    return model


def predict_image(model, img_path: Path, conf: float, iou: float, class_names: list[str]) -> tuple[np.ndarray, list[dict]]:
    """
    Run inference on a single image.
    Returns (annotated_img, detections_list).
    Each detection: {class_id, class_name, confidence, bbox_xyxy}
    """
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = model(img_rgb, size=640)

    # Extract detections above threshold
    detections = []
    df = results.pandas().xyxy[0]
    for _, row in df.iterrows():
        if row["confidence"] < conf:
            continue
        det = {
            "class_id":   int(row["class"]),
            "class_name": class_names[int(row["class"])] if int(row["class"]) < len(class_names) else f"class_{int(row['class'])}",
            "confidence": float(row["confidence"]),
            "bbox_xyxy":  [float(row["xmin"]), float(row["ymin"]), float(row["xmax"]), float(row["ymax"])],
        }
        detections.append(det)

    # Draw annotations
    annotated = img.copy()
    h, w = annotated.shape[:2]
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox_xyxy"]]
        cid    = det["class_id"]
        color  = COLORS.get(cid, (200, 200, 200))
        label  = f"{det['class_name']} {det['confidence']:.2f}"
        # Box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        # Label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(annotated, (x1, max(y1-th-6, 0)), (x1+tw+4, y1), color, -1)
        cv2.putText(annotated, label, (x1+2, max(y1-4, th)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    return annotated, detections


def image_verdict(detections: list[dict], class_names: list[str]) -> str:
    """Derive image-level verdict from detections."""
    found = sorted(set(d["class_name"] for d in detections))
    if not found:
        return "No weapon detected"
    return "Detected: " + ", ".join(found)


def find_best_weights(cfg: dict) -> Path:
    run_dir = ROOT / cfg["train"]["project"] / cfg["train"]["name"]
    w = run_dir / "weights" / "best.pt"
    if w.exists():
        return w
    for p in (ROOT / cfg["train"]["project"]).rglob("best.pt"):
        return p
    return w


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Weapon detection inference on image or folder.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/predict.py --source image.jpg
  python scripts/predict.py --source data/raw/ --conf 0.35
  python scripts/predict.py --source image.jpg --output results/predictions/
        """,
    )
    parser.add_argument("--source",  required=True,       help="Image file or folder path.")
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--conf",    type=float, default=None)
    parser.add_argument("--iou",     type=float, default=None)
    parser.add_argument("--output",  type=str, default=None,
                        help="Output directory (default: results/predictions/).")
    parser.add_argument("--device",  type=str, default=None)
    parser.add_argument("--no_save", action="store_true", help="Do not save annotated images.")
    args = parser.parse_args()

    cfg = load_config()

    # Device
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    weights     = Path(args.weights) if args.weights else find_best_weights(cfg)
    conf        = args.conf  if args.conf  else cfg["eval"]["conf_threshold"]
    iou         = args.iou   if args.iou   else cfg["eval"]["iou_threshold"]
    class_names = cfg["dataset"]["class_names"]
    out_dir     = Path(args.output) if args.output else ROOT / "results" / "predictions"

    if not weights.exists():
        print(f"ERROR: weights not found: {weights}")
        print("Run python scripts/train.py first.")
        sys.exit(1)

    print(f"[predict] Loading model from {weights} …")
    model = load_model(weights, device)
    model.conf = conf
    model.iou  = iou
    print(f"[predict] Model loaded. conf={conf} iou={iou} device={device}")

    # Gather source images
    source = Path(args.source)
    img_exts = set(cfg["dataset"]["image_extensions"])
    if source.is_file():
        images = [source]
    elif source.is_dir():
        images = [p for p in source.rglob("*") if p.suffix.lower() in img_exts]
    else:
        print(f"ERROR: source not found: {source}")
        sys.exit(1)

    print(f"[predict] {len(images)} image(s) to process.")
    if not args.no_save:
        out_dir.mkdir(parents=True, exist_ok=True)

    total_time = 0.0
    for img_p in images:
        t0 = time.time()
        annotated, detections = predict_image(model, img_p, conf, iou, class_names)
        elapsed_ms = (time.time() - t0) * 1000
        total_time += elapsed_ms

        verdict = image_verdict(detections, class_names)
        print(f"  {img_p.name:<40} | {verdict}")
        for det in detections:
            bbox = [f"{v:.0f}" for v in det["bbox_xyxy"]]
            print(f"    → {det['class_name']} conf={det['confidence']:.3f} bbox=[{','.join(bbox)}]")

        if not args.no_save:
            out_path = out_dir / f"pred_{img_p.stem}{img_p.suffix}"
            cv2.imwrite(str(out_path), annotated)

    if images:
        avg_ms = total_time / len(images)
        print(f"\n[predict] Avg inference time: {avg_ms:.1f} ms/image on {device}")
        if not args.no_save:
            print(f"[predict] Saved annotated images to {out_dir}/")


if __name__ == "__main__":
    main()
