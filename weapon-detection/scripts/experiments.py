"""
scripts/experiments.py
======================
Phase 4: Experiments

A. Model-size comparison: yolov5n vs yolov5s vs yolov5m
B. CLAHE preprocessing ablation + degradation robustness test

Usage:
    python scripts/experiments.py --part A          # model size comparison only
    python scripts/experiments.py --part B          # CLAHE ablation only
    python scripts/experiments.py                   # run both parts
    python scripts/experiments.py --epochs 30       # override epoch count
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parent.parent
YOLO_DIR = ROOT / "yolov5"
sys.path.insert(0, str(YOLO_DIR))


def load_config() -> dict:
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_device() -> str:
    import torch
    if torch.cuda.is_available():        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return "mps"
    return "cpu"


def run_yolo_train(weights: str, data_yaml: Path, name: str, epochs: int,
                   img_size: int, batch: int, workers: int, seed: int,
                   project: str, device: str) -> Path:
    cmd = [
        sys.executable, str(YOLO_DIR / "train.py"),
        "--data",    str(data_yaml.resolve()),
        "--weights", weights,
        "--img",     str(img_size),
        "--epochs",  str(epochs),
        "--batch",   str(batch),
        "--workers", str(workers),
        "--project", project,
        "--name",    name,
        "--patience", str(epochs),      # disable early-stop for ablation
        "--seed",    str(seed),
        "--device",  device,
        "--exist-ok",
    ]
    print(f"\n[experiments] Training {name} …")
    subprocess.run(cmd, check=True, cwd=str(ROOT))
    return ROOT / project / name


def run_yolo_val(weights: Path, data_yaml: Path, name: str,
                 img_size: int, conf: float, iou: float, workers: int,
                 project: str, device: str) -> Path:
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
        "--exist-ok",
        "--verbose",
    ]
    subprocess.run(cmd, check=True, cwd=str(ROOT))
    return ROOT / project / name


def parse_last_map(run_dir: Path) -> dict:
    """Parse the last row of results.csv from a YOLOv5 run."""
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    last = df.iloc[-1].to_dict()
    return {
        "mAP50":    round(float(last.get("metrics/mAP_0.5",      0.0)), 4),
        "mAP5095":  round(float(last.get("metrics/mAP_0.5:0.95", 0.0)), 4),
        "precision":round(float(last.get("metrics/precision",     0.0)), 4),
        "recall":   round(float(last.get("metrics/recall",        0.0)), 4),
    }


def model_params_flops(model_name: str, img_size: int = 640) -> tuple[str, str]:
    """Return (params_M, GFLOPs) strings by running a quick model summary."""
    try:
        out = subprocess.check_output(
            [sys.executable, "-c",
             f"import sys; sys.path.insert(0,'{YOLO_DIR}');"
             f"from models.common import DetectMultiBackend;"
             f"import torch; m=DetectMultiBackend('{model_name}.pt',device=torch.device('cpu'));"
             f"from thop import profile; inp=torch.zeros(1,3,{img_size},{img_size});"
             f"macs,params=profile(m.model,inputs=(inp,),verbose=False);"
             f"print(f'{{params/1e6:.2f}} {{macs/1e9*2:.2f}}')"],
            stderr=subprocess.DEVNULL, cwd=str(ROOT),
        ).decode().strip()
        parts = out.split()
        return parts[0], parts[1]
    except Exception:
        return "N/A", "N/A"


# ─────────────────────────────────────────────────────────────────────────────
# Part A — model size comparison
# ─────────────────────────────────────────────────────────────────────────────

def part_a(cfg: dict, epochs: int, device: str) -> None:
    print("\n" + "="*60)
    print("  PART A: Model-size comparison")
    print(f"  epochs={epochs} | device={device}")
    print("="*60)

    model_sizes = cfg["experiments"]["model_sizes"]
    img_size    = cfg["dataset"]["img_size"]
    batch       = cfg["train"]["batch_size"]
    workers     = cfg["train"]["workers"]
    seed        = cfg["train"]["seed"]
    conf        = cfg["eval"]["conf_threshold"]
    iou         = cfg["eval"]["iou_threshold"]
    data_yaml   = ROOT / "data.yaml"

    results_dir = ROOT / cfg["paths"]["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for model in model_sizes:
        weights = f"{model}.pt"
        name    = f"exp_{model}"
        try:
            run_dir = run_yolo_train(
                weights=weights, data_yaml=data_yaml, name=name,
                epochs=epochs, img_size=img_size, batch=batch,
                workers=workers, seed=seed, project="runs/experiments",
                device=device,
            )
            metrics = parse_last_map(run_dir)
            params, gflops = model_params_flops(model, img_size)
        except Exception as e:
            print(f"  ⚠ {model} failed: {e}")
            metrics = {"mAP50": "ERR", "mAP5095": "ERR", "precision": "ERR", "recall": "ERR"}
            params, gflops = "ERR", "ERR"

        row = {"model": model, "params_M": params, "GFLOPs": gflops, **metrics}
        rows.append(row)
        print(f"  {model}: mAP@0.5={metrics.get('mAP50','?')}  mAP@0.5:0.95={metrics.get('mAP5095','?')}")

    # Save table
    df = pd.DataFrame(rows)
    out_csv = results_dir / "model_comparison.csv"
    df.to_csv(out_csv, index=False)

    md_lines = [
        "# Model Size Comparison\n",
        f"> Note: Trained for {epochs} epochs (reduced from 100 for compute efficiency).\n",
        "| Model | Params (M) | GFLOPs | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall |",
        "|-------|-----------|--------|---------|--------------|-----------|--------|",
    ]
    for row in rows:
        md_lines.append(
            f"| {row['model']} | {row['params_M']} | {row['GFLOPs']} | "
            f"{row['mAP50']} | {row['mAP5095']} | {row['precision']} | {row['recall']} |"
        )
    md_path = results_dir / "model_comparison.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"\n  Part A results: {out_csv}")
    print(f"  Part A report : {md_path}")

    # Bar plot
    plot_model_comparison(df, results_dir / "model_comparison.png")


def plot_model_comparison(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = ["#4C9BE8", "#E8844C", "#6ABF69"]

    for ax, metric, title in zip(axes,
                                  ["mAP50", "mAP5095"],
                                  ["mAP@0.5", "mAP@0.5:0.95"]):
        try:
            vals = [float(v) for v in df[metric]]
        except (ValueError, TypeError):
            continue
        bars = ax.bar(df["model"], vals, color=colors[:len(df)])
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_ylabel(title)
        ax.set_ylim(0, 1.0)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=10)

    fig.suptitle("YOLOv5 Model Size Comparison", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Part B — CLAHE preprocessing ablation
# ─────────────────────────────────────────────────────────────────────────────

def apply_clahe_to_dir(src_dir: Path, dst_dir: Path, clip_limit: float, tile_grid: list) -> None:
    """Apply CLAHE (on L channel of LAB) to all images in src_dir → dst_dir."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tuple(tile_grid),
    )
    imgs = list(src_dir.rglob("*"))
    imgs = [p for p in imgs if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    for img_p in tqdm(imgs, desc=f"CLAHE → {dst_dir.name}"):
        img = cv2.imread(str(img_p))
        if img is None:
            continue
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        rel = img_p.relative_to(src_dir)
        out_p = dst_dir / rel
        out_p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_p), out)


def apply_degradation(img: np.ndarray, mode: str, cfg: dict) -> np.ndarray:
    """Apply forensic degradation to a single image."""
    d = cfg["experiments"]["degradation"]
    if mode == "noise":
        std = d["gaussian_noise_std"]
        noise = np.random.normal(0, std, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    elif mode == "blur":
        k = d["blur_ksize"]
        img = cv2.GaussianBlur(img, (k, k), 0)
    elif mode == "dark":
        factor = d["brightness_factor"]
        img = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    return img


def degrade_val_set(data_yaml: Path, output_dir: Path, mode: str, cfg: dict) -> Path:
    """Create a degraded copy of the val set and return a new data.yaml."""
    with open(data_yaml) as f:
        d = yaml.safe_load(f)
    val_txt = Path(d["val"])
    val_images = [Path(l.strip()) for l in val_txt.read_text(encoding="utf-8").strip().splitlines() if l.strip()]

    degrad_dir = output_dir / mode
    degrad_dir.mkdir(parents=True, exist_ok=True)
    new_paths = []

    for img_p in tqdm(val_images, desc=f"Degrading ({mode})"):
        img = cv2.imread(str(img_p))
        if img is None:
            continue
        img_d = apply_degradation(img, mode, cfg)
        out_p = degrad_dir / img_p.name
        cv2.imwrite(str(out_p), img_d)
        # Copy label
        lbl = img_p.with_suffix(".txt")
        if lbl.exists():
            import shutil
            shutil.copy(lbl, degrad_dir / lbl.name)
        new_paths.append(out_p)

    # Write new val.txt and data.yaml
    new_val_txt = degrad_dir / "val.txt"
    new_val_txt.write_text("\n".join(str(p) for p in new_paths), encoding="utf-8")
    new_yaml = degrad_dir / "data.yaml"
    new_data = {
        "train": d["train"],
        "val":   str(new_val_txt.resolve()),
        "nc":    d["nc"],
        "names": d["names"],
    }
    with open(new_yaml, "w") as f:
        yaml.dump(new_data, f, default_flow_style=False, sort_keys=False)
    return new_yaml


def part_b(cfg: dict, epochs: int, device: str) -> None:
    print("\n" + "="*60)
    print("  PART B: CLAHE ablation + degradation robustness")
    print(f"  epochs={epochs} | device={device}")
    print("="*60)

    results_dir = ROOT / cfg["paths"]["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)
    conf = cfg["eval"]["conf_threshold"]
    iou  = cfg["eval"]["iou_threshold"]
    img_size = cfg["dataset"]["img_size"]
    workers  = cfg["train"]["workers"]
    seed     = cfg["train"]["seed"]
    batch    = cfg["train"]["batch_size"]

    # ── B1: Train YOLOv5s with CLAHE-preprocessed data ─────────────────────
    # First build CLAHE-preprocessed dataset copy
    data_yaml = ROOT / "data.yaml"
    with open(data_yaml) as f:
        d = yaml.safe_load(f)

    # For simplicity: apply CLAHE to validation set only (robustness eval)
    # and retrain on original train set + CLAHE (creates processed/ copies)
    clahe_cfg = cfg["experiments"]["clahe"]
    proc_dir  = ROOT / cfg["paths"]["data_processed"]

    print("\n[B] Applying CLAHE to train images …")
    # We apply CLAHE to a copy of the train images
    train_txt_path = Path(d["train"])
    train_images   = [Path(l.strip()) for l in train_txt_path.read_text().strip().splitlines() if l.strip()]

    clahe_train_dir = proc_dir / "clahe" / "train"
    clahe_train_dir.mkdir(parents=True, exist_ok=True)
    clahe_val_dir   = proc_dir / "clahe" / "val"
    clahe_val_dir.mkdir(parents=True, exist_ok=True)

    import shutil, cv2

    def apply_clahe_img(img_p: Path, dst_dir: Path) -> Path:
        clahe = cv2.createCLAHE(
            clipLimit=clahe_cfg["clip_limit"],
            tileGridSize=tuple(clahe_cfg["tile_grid"]),
        )
        img = cv2.imread(str(img_p))
        if img is None:
            return img_p
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        out_p = dst_dir / img_p.name
        cv2.imwrite(str(out_p), out)
        lbl = img_p.with_suffix(".txt")
        if lbl.exists():
            shutil.copy(lbl, dst_dir / lbl.name)
        return out_p

    clahe_train_paths = [apply_clahe_img(p, clahe_train_dir) for p in tqdm(train_images, desc="CLAHE train")]
    val_txt_path = Path(d["val"])
    val_images   = [Path(l.strip()) for l in val_txt_path.read_text().strip().splitlines() if l.strip()]
    clahe_val_paths = [apply_clahe_img(p, clahe_val_dir) for p in tqdm(val_images, desc="CLAHE val")]

    # Write new split files
    clahe_split_dir = proc_dir / "clahe"
    clahe_train_txt = clahe_split_dir / "train.txt"
    clahe_val_txt   = clahe_split_dir / "val.txt"
    clahe_train_txt.write_text("\n".join(str(p) for p in clahe_train_paths), encoding="utf-8")
    clahe_val_txt.write_text("\n".join(str(p) for p in clahe_val_paths), encoding="utf-8")

    clahe_yaml = clahe_split_dir / "data.yaml"
    clahe_data = {
        "train": str(clahe_train_txt.resolve()),
        "val":   str(clahe_val_txt.resolve()),
        "nc":    d["nc"],
        "names": d["names"],
    }
    with open(clahe_yaml, "w") as f:
        yaml.dump(clahe_data, f, default_flow_style=False, sort_keys=False)

    # Train CLAHE model
    clahe_run_dir = run_yolo_train(
        weights="yolov5s.pt", data_yaml=clahe_yaml,
        name="yolov5s_clahe", epochs=epochs,
        img_size=img_size, batch=batch, workers=workers,
        seed=seed, project="runs/experiments", device=device,
    )
    clahe_metrics = parse_last_map(clahe_run_dir)

    # Baseline metrics (from already-trained baseline)
    baseline_run = ROOT / cfg["train"]["project"] / cfg["train"]["name"]
    baseline_metrics = parse_last_map(baseline_run)

    # ── B2: Degradation robustness ───────────────────────────────────────────
    baseline_weights = baseline_run / "weights" / "best.pt"
    clahe_weights    = clahe_run_dir / "weights" / "best.pt"
    degradation_modes = ["noise", "blur", "dark"]
    deg_dir = proc_dir / "degraded"

    robust_rows = []
    for mode in degradation_modes:
        for label, weights, w_name in [
            ("Baseline", baseline_weights, "baseline"),
            ("CLAHE",    clahe_weights,    "clahe"),
        ]:
            try:
                degrad_yaml = degrade_val_set(data_yaml, deg_dir, mode, cfg)
                val_run = run_yolo_val(
                    weights=weights, data_yaml=degrad_yaml,
                    name=f"robustness_{mode}_{w_name}",
                    img_size=img_size, conf=conf, iou=iou,
                    workers=workers, project="runs/experiments", device=device,
                )
                m = parse_last_map(val_run)
            except Exception as e:
                print(f"  ⚠ Robustness {mode}/{w_name} failed: {e}")
                m = {"mAP50": "ERR", "mAP5095": "ERR"}
            robust_rows.append({"degradation": mode, "model": label, **m})

    # ── Write ablation table ─────────────────────────────────────────────────
    ablation_rows = [
        {"model": "Baseline (no CLAHE)", **baseline_metrics},
        {"model": "CLAHE preprocessing", **clahe_metrics},
    ]

    md_lines = [
        "# CLAHE Preprocessing Ablation\n",
        f"> Trained for {epochs} epochs (reduced from 100 for compute efficiency).\n",
        "## Training-set ablation\n",
        "| Model | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall |",
        "|-------|---------|--------------|-----------|--------|",
    ]
    for row in ablation_rows:
        md_lines.append(
            f"| {row['model']} | {row.get('mAP50','?')} | {row.get('mAP5095','?')} | "
            f"{row.get('precision','?')} | {row.get('recall','?')} |"
        )
    md_lines.append("\n## Robustness on degraded val images\n")
    md_lines.append("| Degradation | Model | mAP@0.5 | mAP@0.5:0.95 |")
    md_lines.append("|-------------|-------|---------|--------------|")
    for row in robust_rows:
        md_lines.append(
            f"| {row['degradation']} | {row['model']} | {row.get('mAP50','?')} | {row.get('mAP5095','?')} |"
        )

    md_path = results_dir / "clahe_ablation.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"\n  Part B results: {md_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4 experiments.")
    parser.add_argument("--part",   choices=["A", "B", "both"], default="both")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override epoch count (default from config).")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    cfg    = load_config()
    device = args.device or get_device()
    epochs = args.epochs or cfg["experiments"]["ablation_epochs"]

    if device == "cpu":
        print("⚠  No GPU detected. Experiments will run on CPU — this is very slow.")
        print("   Recommend running in Google Colab (see notebooks/train_colab.ipynb).")

    if args.part in ("A", "both"):
        part_a(cfg, epochs, device)
    if args.part in ("B", "both"):
        part_b(cfg, epochs, device)

    print("\n✓ Phase 4 complete.")


if __name__ == "__main__":
    main()
