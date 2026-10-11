"""
scripts/audit_split.py
======================
Phase 1: Dataset audit, deduplication, train/val split, and EDA plots.

Usage:
    python scripts/audit_split.py --inspect          # just print dataset structure
    python scripts/audit_split.py                    # full audit + split
    python scripts/audit_split.py --data_dir data/raw/my_dataset
"""

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import imagehash
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from PIL import Image
from tqdm import tqdm

matplotlib.use("Agg")  # headless – no display required

# ── Project root is one level above this script ──────────────────────────────
ROOT = Path(__file__).resolve().parent.parent


def load_config() -> dict:
    cfg_path = ROOT / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


CFG = load_config()
IMG_EXTS = set(CFG["dataset"]["image_extensions"])


# ─────────────────────────────────────────────────────────────────────────────
# Dataset discovery helpers
# ─────────────────────────────────────────────────────────────────────────────

def find_images(root: Path) -> list[Path]:
    """Recursively find all image files under *root*."""
    imgs = []
    for p in root.rglob("*"):
        if p.suffix.lower() in IMG_EXTS and p.is_file():
            imgs.append(p)
    return sorted(imgs)


def find_label_for_image(img_path: Path) -> Path | None:
    """Return the corresponding .txt label path for an image (YOLOv5 layout)."""
    # Common layouts: images/ → labels/  OR  same directory
    label_path = img_path.with_suffix(".txt")
    if label_path.exists():
        return label_path
    # Try sibling 'labels' folder
    for part in img_path.parts:
        if part.lower() in {"images", "imgs", "image"}:
            candidate = Path(str(img_path).replace(
                os.sep + part + os.sep, os.sep + "labels" + os.sep, 1
            )).with_suffix(".txt")
            if candidate.exists():
                return candidate
    return None


def inspect_dataset(data_dir: Path) -> None:
    """Print dataset structure, image/label counts, class ids, and image sizes."""
    print(f"\n{'='*60}")
    print(f"  DATASET INSPECTION: {data_dir}")
    print(f"{'='*60}")

    # Directory tree (2 levels)
    print("\n── Directory structure (2 levels) ──")
    for item in sorted(data_dir.rglob("*")):
        depth = len(item.relative_to(data_dir).parts)
        if depth <= 2:
            prefix = "  " * (depth - 1) + ("├── " if depth > 0 else "")
            print(f"{prefix}{item.name}{'/' if item.is_dir() else ''}")

    images = find_images(data_dir)
    print(f"\nTotal images found : {len(images)}")

    # Label stats
    class_counts: Counter = Counter()
    labelled = 0
    unlabelled = []
    malformed = []
    coords_oob = []

    widths, heights = [], []
    boxes_per_img = []

    for img_p in tqdm(images, desc="Scanning images"):
        try:
            with Image.open(img_p) as im:
                w, h = im.size
                widths.append(w)
                heights.append(h)
        except Exception:
            w, h = 0, 0

        lbl = find_label_for_image(img_p)
        if lbl is None:
            unlabelled.append(img_p)
            continue

        labelled += 1
        lines = lbl.read_text(encoding="utf-8").strip().splitlines()
        if not lines:
            boxes_per_img.append(0)
            continue

        n_boxes = 0
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                malformed.append((img_p, line))
                continue
            try:
                cid, xc, yc, bw, bh = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            except ValueError:
                malformed.append((img_p, line))
                continue
            class_counts[cid] += 1
            if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 < bw <= 1.0 and 0.0 < bh <= 1.0):
                coords_oob.append((img_p, line))
            n_boxes += 1
        boxes_per_img.append(n_boxes)

    print(f"\n── Label summary ──")
    print(f"  Labelled images    : {labelled}")
    print(f"  Unlabelled images  : {len(unlabelled)}")
    print(f"  Malformed lines    : {len(malformed)}")
    print(f"  OOB coord lines    : {len(coords_oob)}")
    print(f"\n── Class id → box count ──")
    for cid, cnt in sorted(class_counts.items()):
        print(f"  class {cid}: {cnt} boxes")
    print(f"\n── Image size stats ──")
    if widths:
        print(f"  Width  – min:{min(widths)} max:{max(widths)} mean:{np.mean(widths):.0f}")
        print(f"  Height – min:{min(heights)} max:{max(heights)} mean:{np.mean(heights):.0f}")
    print(f"\n── Boxes per image ──")
    if boxes_per_img:
        arr = np.array(boxes_per_img)
        print(f"  min:{arr.min()} max:{arr.max()} mean:{arr.mean():.2f} median:{np.median(arr):.1f}")

    if unlabelled:
        print(f"\n⚠  Images without label files (first 10):")
        for p in unlabelled[:10]:
            print(f"    {p}")

    if malformed:
        print(f"\n⚠  Malformed label lines (first 5):")
        for p, line in malformed[:5]:
            print(f"    {p}: '{line}'")


# ─────────────────────────────────────────────────────────────────────────────
# Deduplication
# ─────────────────────────────────────────────────────────────────────────────

def compute_phash(img_path: Path, hash_size: int = 8) -> str | None:
    """Compute perceptual hash for an image. Returns None on failure."""
    try:
        with Image.open(img_path) as im:
            return str(imagehash.phash(im, hash_size=hash_size))
    except Exception:
        return None


def remove_duplicates(images: list[Path], hamming_thresh: int = 4, hash_size: int = 8) -> tuple[list[Path], int]:
    """
    Detect and remove near-duplicate images using perceptual hashing.
    Returns (deduplicated_list, n_removed).
    """
    print("\n── Deduplication (phash) ──")
    hashes: dict[str, Path] = {}
    to_remove: set[Path] = set()

    for img_p in tqdm(images, desc="Hashing images"):
        h = compute_phash(img_p, hash_size=hash_size)
        if h is None:
            continue

        duplicate = False
        for existing_h, existing_p in hashes.items():
            dist = bin(int(h, 16) ^ int(existing_h, 16)).count("1")
            if dist <= hamming_thresh:
                to_remove.add(img_p)
                duplicate = True
                break
        if not duplicate:
            hashes[h] = img_p

    kept = [p for p in images if p not in to_remove]
    print(f"  Original   : {len(images)}")
    print(f"  Removed    : {len(to_remove)} (duplicates)")
    print(f"  Kept       : {len(kept)}")
    return kept, len(to_remove)


# ─────────────────────────────────────────────────────────────────────────────
# Label parsing and class mapping
# ─────────────────────────────────────────────────────────────────────────────

def get_image_classes(img_path: Path, class_names: list[str]) -> list[int]:
    """Return list of unique class ids found in the label file for an image."""
    lbl = find_label_for_image(img_path)
    if lbl is None:
        return []
    classes = set()
    for line in lbl.read_text(encoding="utf-8").strip().splitlines():
        parts = line.strip().split()
        if len(parts) == 5:
            try:
                classes.add(int(parts[0]))
            except ValueError:
                pass
    return list(classes)


# ─────────────────────────────────────────────────────────────────────────────
# Train / val split (stratified by primary class)
# ─────────────────────────────────────────────────────────────────────────────

def stratified_split(
    images: list[Path],
    val_frac: float,
    seed: int,
    class_names: list[str],
) -> tuple[list[Path], list[Path]]:
    """
    80/20 stratified split by primary class (first class_id in label).
    Falls back to random split if no labels available.
    """
    rng = random.Random(seed)

    # group by primary class
    buckets: dict[int, list[Path]] = defaultdict(list)
    no_label = []
    for img_p in images:
        classes = get_image_classes(img_p, class_names)
        if classes:
            buckets[classes[0]].append(img_p)
        else:
            no_label.append(img_p)

    train_imgs, val_imgs = [], []
    for cid, imgs in buckets.items():
        rng.shuffle(imgs)
        n_val = max(1, round(len(imgs) * val_frac))
        val_imgs.extend(imgs[:n_val])
        train_imgs.extend(imgs[n_val:])

    # Distribute no-label images randomly
    rng.shuffle(no_label)
    n_val_nl = round(len(no_label) * val_frac)
    val_imgs.extend(no_label[:n_val_nl])
    train_imgs.extend(no_label[n_val_nl:])

    rng.shuffle(train_imgs)
    rng.shuffle(val_imgs)

    return train_imgs, val_imgs


# ─────────────────────────────────────────────────────────────────────────────
# EDA plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_eda(images: list[Path], class_names: list[str], eda_dir: Path) -> None:
    """Generate EDA plots: class balance, boxes/image, box-size distribution."""
    eda_dir.mkdir(parents=True, exist_ok=True)

    class_box_counts: Counter = Counter()
    boxes_per_img = []
    box_widths, box_heights, box_areas = [], [], []

    for img_p in tqdm(images, desc="EDA scan"):
        lbl = find_label_for_image(img_p)
        if lbl is None:
            boxes_per_img.append(0)
            continue
        lines = [l.strip() for l in lbl.read_text().strip().splitlines() if l.strip()]
        boxes_per_img.append(len(lines))
        for line in lines:
            parts = line.split()
            if len(parts) != 5:
                continue
            try:
                cid, xc, yc, bw, bh = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                name = class_names[cid] if cid < len(class_names) else f"class_{cid}"
                class_box_counts[name] += 1
                box_widths.append(bw)
                box_heights.append(bh)
                box_areas.append(bw * bh)
            except (ValueError, IndexError):
                pass

    sns.set_theme(style="darkgrid")

    # 1. Class balance bar chart
    fig, ax = plt.subplots(figsize=(7, 4))
    classes = list(class_box_counts.keys())
    counts = [class_box_counts[c] for c in classes]
    bars = ax.bar(classes, counts, color=["#4C9BE8", "#E8844C"])
    ax.set_title("Box Count per Class", fontsize=14, fontweight="bold")
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of bounding boxes")
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(counts) * 0.01,
                str(cnt), ha="center", va="bottom", fontsize=11)
    fig.tight_layout()
    fig.savefig(eda_dir / "class_balance.png", dpi=150)
    plt.close(fig)

    # 2. Boxes per image histogram
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(boxes_per_img, bins=range(0, max(boxes_per_img) + 2), color="#4C9BE8", edgecolor="white")
    ax.set_title("Bounding Boxes per Image", fontsize=14, fontweight="bold")
    ax.set_xlabel("Number of boxes")
    ax.set_ylabel("Image count")
    ax.axvline(np.mean(boxes_per_img), color="#E8844C", lw=2, linestyle="--", label=f"mean={np.mean(boxes_per_img):.1f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(eda_dir / "boxes_per_image.png", dpi=150)
    plt.close(fig)

    # 3. Box size distribution (width, height, area)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, vals, title, color in zip(
        axes,
        [box_widths, box_heights, box_areas],
        ["Box Width (normalised)", "Box Height (normalised)", "Box Area (normalised)"],
        ["#4C9BE8", "#E8844C", "#6ABF69"],
    ):
        ax.hist(vals, bins=40, color=color, edgecolor="white", alpha=0.85)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")
        ax.axvline(np.mean(vals), color="black", lw=1.5, linestyle="--", label=f"mean={np.mean(vals):.3f}")
        ax.legend(fontsize=8)
    fig.suptitle("Bounding Box Size Distribution", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(eda_dir / "box_size_distribution.png", dpi=150)
    plt.close(fig)

    print(f"\nEDA plots saved to {eda_dir}/")


# ─────────────────────────────────────────────────────────────────────────────
# data.yaml writer
# ─────────────────────────────────────────────────────────────────────────────

def write_data_yaml(
    data_dir: Path,
    train_txt: Path,
    val_txt: Path,
    class_names: list[str],
    output: Path,
) -> None:
    """Write a data.yaml compatible with YOLOv5's train.py."""
    data = {
        "train": str(train_txt.resolve()),
        "val":   str(val_txt.resolve()),
        "nc":    len(class_names),
        "names": class_names,
    }
    with open(output, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    print(f"data.yaml written → {output}")


# ─────────────────────────────────────────────────────────────────────────────
# Split file writer
# ─────────────────────────────────────────────────────────────────────────────

def write_split_files(
    train_imgs: list[Path],
    val_imgs: list[Path],
    splits_dir: Path,
) -> tuple[Path, Path]:
    """
    Write train.txt and val.txt (one absolute image path per line).
    These files are consumed by YOLOv5's train.py / val.py.
    """
    splits_dir.mkdir(parents=True, exist_ok=True)
    train_txt = splits_dir / "train.txt"
    val_txt   = splits_dir / "val.txt"
    train_txt.write_text("\n".join(str(p.resolve()) for p in train_imgs) + "\n", encoding="utf-8")
    val_txt.write_text("\n".join(str(p.resolve()) for p in val_imgs) + "\n", encoding="utf-8")
    print(f"Split files written:\n  {train_txt} ({len(train_imgs)} images)\n  {val_txt} ({len(val_imgs)} images)")
    return train_txt, val_txt


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Dataset audit, dedup and split for weapon-detection.")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Override data directory (default: data/raw first subdirectory found).")
    parser.add_argument("--inspect", action="store_true",
                        help="Print dataset structure only; do not split.")
    parser.add_argument("--no_dedup", action="store_true",
                        help="Skip deduplication (faster for testing).")
    args = parser.parse_args()

    cfg = load_config()

    # Locate dataset
    raw_dir = ROOT / cfg["paths"]["data_raw"]
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        # Auto-discover: look for a subdirectory containing images
        subdirs = [d for d in raw_dir.iterdir() if d.is_dir()] if raw_dir.exists() else []
        data_dir = subdirs[0] if subdirs else raw_dir

    if not data_dir.exists():
        print(f"ERROR: data directory not found: {data_dir}")
        print("Please download the dataset into data/raw/ first.")
        print("See README.md → Dataset Download section.")
        sys.exit(1)

    # ── Always inspect ──────────────────────────────────────────────────────
    inspect_dataset(data_dir)
    if args.inspect:
        return

    # ── Gather images ───────────────────────────────────────────────────────
    images = find_images(data_dir)
    print(f"\n[audit_split] {len(images)} images found in {data_dir}")

    # ── Validate labels ─────────────────────────────────────────────────────
    class_names = cfg["dataset"]["class_names"]
    num_classes  = cfg["dataset"]["num_classes"]
    bad_cid, bad_coord, no_label = [], [], []

    for img_p in tqdm(images, desc="Validating labels"):
        lbl = find_label_for_image(img_p)
        if lbl is None:
            no_label.append(img_p)
            continue
        for line in lbl.read_text(encoding="utf-8").strip().splitlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            try:
                cid, xc, yc, bw, bh = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                if cid < 0 or cid >= num_classes:
                    bad_cid.append((img_p, cid))
                if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 < bw <= 1.0 and 0.0 < bh <= 1.0):
                    bad_coord.append((img_p, line))
            except ValueError:
                pass

    print(f"\n── Validation summary ──")
    print(f"  Images missing label : {len(no_label)}")
    print(f"  Out-of-range class ids: {len(bad_cid)}")
    print(f"  OOB coordinates      : {len(bad_coord)}")
    if bad_cid:
        print("  ⚠ Example bad class ids:", [(str(p.name), c) for p, c in bad_cid[:5]])

    # ── Remove images without labels ─────────────────────────────────────────
    valid_images = [p for p in images if find_label_for_image(p) is not None]
    print(f"\n  Valid images (have label): {len(valid_images)}")

    # ── Deduplication ────────────────────────────────────────────────────────
    if not args.no_dedup:
        valid_images, n_removed = remove_duplicates(
            valid_images,
            hamming_thresh=cfg["dedup"]["hamming_threshold"],
            hash_size=cfg["dedup"]["hash_size"],
        )
    else:
        n_removed = 0
        print("\n[skip] deduplication disabled (--no_dedup)")

    # ── EDA plots ─────────────────────────────────────────────────────────────
    eda_dir = ROOT / cfg["paths"]["eda_dir"]
    plot_eda(valid_images, class_names, eda_dir)

    # ── Train/val split ───────────────────────────────────────────────────────
    train_imgs, val_imgs = stratified_split(
        valid_images,
        val_frac=cfg["split"]["val_fraction"],
        seed=cfg["split"]["seed"],
        class_names=class_names,
    )
    print(f"\n── Split summary ──")
    print(f"  Train : {len(train_imgs)}")
    print(f"  Val   : {len(val_imgs)}")

    splits_dir = ROOT / cfg["paths"]["splits_dir"]
    train_txt, val_txt = write_split_files(train_imgs, val_imgs, splits_dir)

    # ── data.yaml ─────────────────────────────────────────────────────────────
    data_yaml = ROOT / "data.yaml"
    write_data_yaml(data_dir, train_txt, val_txt, class_names, data_yaml)

    # ── Summary report ────────────────────────────────────────────────────────
    report = {
        "total_images": len(images),
        "valid_images": len(valid_images),
        "duplicates_removed": n_removed,
        "no_label": len(no_label),
        "bad_class_id": len(bad_cid),
        "bad_coords": len(bad_coord),
        "train": len(train_imgs),
        "val": len(val_imgs),
        "class_names": class_names,
    }
    report_path = ROOT / "results" / "audit_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nAudit report saved → {report_path}")
    print("\n✓ Phase 1 complete.")


if __name__ == "__main__":
    main()
