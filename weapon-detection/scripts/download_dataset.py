"""
scripts/download_dataset.py
===========================
Download the Weapon Detection Dataset from Kaggle.

Usage:
    python scripts/download_dataset.py          # try kagglehub API
    python scripts/download_dataset.py --manual  # show manual instructions
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_config() -> dict:
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


DATASET_SLUG = "raghavnanjappan/weapon-dataset-for-yolov5"
KAGGLE_JSON  = Path.home() / ".kaggle" / "kaggle.json"


def check_kaggle_creds() -> bool:
    """Return True if kaggle.json exists and is readable."""
    return KAGGLE_JSON.exists() and KAGGLE_JSON.stat().st_size > 10


def try_kagglehub(dest: Path) -> bool:
    """Attempt download via kagglehub. Returns True on success."""
    try:
        import kagglehub
        print(f"[download] Downloading via kagglehub: {DATASET_SLUG} …")
        path = kagglehub.dataset_download(DATASET_SLUG)
        src  = Path(path)
        print(f"[download] Downloaded to cache: {src}")

        # Copy to our data/raw/
        dst = dest / "dataset"
        if dst.exists():
            print(f"[download] Destination already exists: {dst}. Skipping copy.")
        else:
            print(f"[download] Copying to {dst} …")
            shutil.copytree(src, dst)
        print(f"[download] ✓ Dataset at: {dst}")
        return True
    except Exception as e:
        print(f"[download] kagglehub failed: {e}")
        return False


def show_manual_instructions(dest: Path) -> None:
    print("""
╔══════════════════════════════════════════════════════════════════╗
║         MANUAL DATASET DOWNLOAD INSTRUCTIONS                     ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  1. Open this URL in your browser:                               ║
║     https://www.kaggle.com/datasets/raghavnanjappan/             ║
║     weapon-dataset-for-yolov5                                    ║
║                                                                  ║
║  2. Click the "Download" button (you need a Kaggle account).     ║
║                                                                  ║
║  3. Unzip the downloaded file.                                   ║
║                                                                  ║
║  4. Place the unzipped folder inside:                            ║
║""" + f"\n║     {dest}    " + """
║                                                                  ║
║  5. Your structure should look like:                             ║
║     data/raw/                                                    ║
║       └── <dataset_folder>/                                      ║
║             ├── images/  (or train/val/test/)                    ║
║             └── labels/                                          ║
║                                                                  ║
║  6. Re-run: python scripts/download_dataset.py                   ║
║     or:     python scripts/audit_split.py                        ║
║                                                                  ║
╠══════════════════════════════════════════════════════════════════╣
║  API OPTION (faster):                                            ║
║  1. Go to https://www.kaggle.com/account → API → "New Token"    ║
║  2. Save kaggle.json to:                                         ║
║     C:\\Users\\<USERNAME>\\.kaggle\\kaggle.json                  ║
║  3. Re-run this script.                                          ║
╚══════════════════════════════════════════════════════════════════╝
""")


def validate_download(dest: Path) -> bool:
    """Check that images and labels are present after download."""
    img_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    images = list(p for p in dest.rglob("*") if p.suffix.lower() in img_exts)
    labels = list(dest.rglob("*.txt"))
    print(f"\n[validate] Images found : {len(images)}")
    print(f"[validate] Labels found : {len(labels)}")
    if len(images) < 100:
        print("[validate] ⚠ Very few images — download may be incomplete.")
        return False
    print("[validate] ✓ Dataset looks complete.")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Download weapon detection dataset from Kaggle.")
    parser.add_argument("--manual", action="store_true", help="Show manual download instructions.")
    args = parser.parse_args()

    cfg  = load_config()
    dest = ROOT / cfg["paths"]["data_raw"]
    dest.mkdir(parents=True, exist_ok=True)

    if args.manual:
        show_manual_instructions(dest)
        return

    # Check if data already exists
    img_exts = set(cfg["dataset"]["image_extensions"])
    existing = list(p for p in dest.rglob("*") if p.suffix.lower() in img_exts)
    if len(existing) >= 100:
        print(f"[download] Dataset already present ({len(existing)} images in {dest})")
        validate_download(dest)
        return

    # Try kaggle credentials
    if not check_kaggle_creds():
        print(f"\n[download] WARNING: kaggle.json not found at {KAGGLE_JSON}")
        print("[download] Attempting unauthenticated kagglehub download ...")

    ok = try_kagglehub(dest)

    if not ok:
        show_manual_instructions(dest)
        print("\n[download] Please download manually and re-run.")
        sys.exit(1)

    validate_download(dest)


if __name__ == "__main__":
    main()
