"""
scripts/import_results.py
=========================
Unpack a results zip produced by the Kaggle/Colab notebook into the local
results/ directory, and copy best.pt to weights/best.pt.

Usage:
    python scripts/import_results.py --zip /path/to/results.zip
    python scripts/import_results.py --zip results.zip --dry_run

The zip is expected to contain at least:
  - metrics.csv / metrics.md
  - eda/*.png
  - errors/
  - feature_maps/
  - architecture_summary.json   (optional)

It also looks for best.pt / last.pt at the root of the zip or inside a
runs/train/.../weights/ sub-path and copies it to weights/best.pt.
"""

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_config() -> dict:
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def find_weights_in_zip(zf: zipfile.ZipFile) -> str | None:
    """Return the first path inside the zip that ends with best.pt or last.pt."""
    for name in zf.namelist():
        p = Path(name)
        if p.name in {"best.pt", "last.pt"}:
            return name
    return None


def import_results(zip_path: Path, dry_run: bool = False) -> None:
    if not zip_path.exists():
        print(f"ERROR: zip not found: {zip_path}")
        sys.exit(1)

    cfg = load_config()
    results_dir = ROOT / cfg["paths"]["results_dir"]
    weights_dir = ROOT / cfg["paths"]["weights_dir"]

    print(f"\n[import] Opening {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        print(f"[import] {len(names)} entries in zip")

        # ── Extract all into results/ ─────────────────────────────────────────
        if not dry_run:
            results_dir.mkdir(parents=True, exist_ok=True)
            for name in names:
                if name.endswith("/"):
                    continue  # skip directories
                # Strip leading component if the zip has a top-level folder
                parts = Path(name).parts
                if len(parts) > 1 and parts[0] in {"results", "results/"}:
                    rel = Path(*parts[1:])
                else:
                    rel = Path(name)
                dest = results_dir / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                data = zf.read(name)
                dest.write_bytes(data)
                print(f"  Extracted: {dest.relative_to(ROOT)}")

        # ── Find and copy weights ─────────────────────────────────────────────
        wt_name = find_weights_in_zip(zf)
        if wt_name:
            print(f"\n[import] Found weights: {wt_name}")
            if not dry_run:
                weights_dir.mkdir(parents=True, exist_ok=True)
                dest_pt = weights_dir / "best.pt"
                dest_pt.write_bytes(zf.read(wt_name))
                print(f"  Saved to: {dest_pt.relative_to(ROOT)}")
        else:
            print("\n[import] WARNING: No best.pt / last.pt found in zip.")
            print("  Copy it manually to weights/best.pt")

    if dry_run:
        print("\n[import] Dry run — no files written.")
    else:
        print(f"\n[import] Done. Results at: {results_dir}")
        print("[import] Next step: python scripts/inspect_model.py")


def _self_test() -> None:
    """Create a small fake zip, import it, check results, delete temp files."""
    import os
    import tempfile

    cfg = load_config()
    results_dir = ROOT / cfg["paths"]["results_dir"]

    # Build fake zip
    tmp = Path(tempfile.mkdtemp())
    fake_zip = tmp / "fake_results.zip"
    fake_csv = b"metric,value\nmAP50,0.91\n"
    fake_pt  = b"fake model weights"  # not a real .pt — just for path test

    with zipfile.ZipFile(fake_zip, "w") as zf:
        zf.writestr("results/metrics.csv", fake_csv)
        zf.writestr("results/eda/class_balance.png", b"\x89PNG fake")
        zf.writestr("runs/train/yolov5s_baseline/weights/best.pt", fake_pt)

    print("\n[self_test] Importing fake zip …")
    import_results(fake_zip, dry_run=False)

    # Verify
    csv_path = results_dir / "metrics.csv"
    pt_path  = ROOT / "weights" / "best.pt"
    assert csv_path.exists(), f"metrics.csv not found at {csv_path}"
    assert pt_path.exists(),  f"best.pt not found at {pt_path}"
    assert csv_path.read_bytes() == fake_csv, "metrics.csv content mismatch"
    assert pt_path.read_bytes() == fake_pt,   "best.pt content mismatch"
    print("[self_test] Assertions PASSED.")

    # Clean up
    csv_path.unlink()
    (results_dir / "eda" / "class_balance.png").unlink(missing_ok=True)
    pt_path.unlink()
    shutil.rmtree(tmp)
    print("[self_test] Temp files deleted.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import results zip from Kaggle/Colab training run."
    )
    parser.add_argument("--zip",      type=str,        help="Path to results.zip")
    parser.add_argument("--dry_run",  action="store_true", help="List contents without extracting.")
    parser.add_argument("--self_test", action="store_true", help="Run self-test with fake zip.")
    args = parser.parse_args()

    if args.self_test:
        _self_test()
        return

    if not args.zip:
        parser.error("--zip is required (or use --self_test)")

    import_results(Path(args.zip), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
