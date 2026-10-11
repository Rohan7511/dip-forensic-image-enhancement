"""
scripts/train.py
================
Phase 2: Training YOLOv5s with smoke test + full training.

Usage:
    python scripts/train.py --smoke          # 2-epoch smoke test only
    python scripts/train.py                  # smoke test then full training
    python scripts/train.py --no_smoke       # skip smoke test (not recommended)
    python scripts/train.py --weights yolov5n.pt  # override model size
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
YOLO_DIR = ROOT / "yolov5"


def load_config() -> dict:
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def get_device() -> str:
    """Return 'cuda', 'mps', or 'cpu'."""
    if torch.cuda.is_available():
        dev = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        dev = "mps"
    else:
        dev = "cpu"
    print(f"[train] Device detected: {dev}")
    return dev


def clone_yolov5(commit: str = "v7.0") -> None:
    """Clone or download ultralytics/yolov5 at the given tag/commit."""
    if YOLO_DIR.exists():
        print(f"[train] YOLOv5 already present at {YOLO_DIR}")
        return
    print(f"[train] Obtaining ultralytics/yolov5 @ {commit} …")
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", commit,
             "https://github.com/ultralytics/yolov5.git", str(YOLO_DIR)],
            check=True,
            capture_output=True,
        )
        print(f"[train] Cloned yolov5 @ {commit}")
    except Exception as e:
        print(f"[train] git clone failed ({e}), falling back to zip download …")
        import urllib.request, zipfile, io, shutil
        url = f"https://github.com/ultralytics/yolov5/archive/refs/tags/{commit}.zip"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp:
            z = zipfile.ZipFile(io.BytesIO(resp.read()))
            temp_dir = ROOT / "yolov5_temp"
            z.extractall(temp_dir)
            extracted = list(temp_dir.iterdir())[0]
            shutil.move(str(extracted), str(YOLO_DIR))
            shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"[train] Downloaded and extracted yolov5 @ {commit}")


def install_yolov5_deps() -> None:
    """Install YOLOv5's own requirements (skip torch since we manage it)."""
    req = YOLO_DIR / "requirements.txt"
    if not req.exists():
        return
    print("[train] Installing YOLOv5 dependencies …")
    # Install but skip torch/torchvision to avoid overwriting our pinned version
    lines = req.read_text().splitlines()
    filtered = [l for l in lines if not any(
        l.lower().startswith(skip) for skip in ["torch", "torchvision", "torchaudio"]
    )]
    tmp_req = ROOT / "yolov5_deps.txt"
    tmp_req.write_text("\n".join(filtered))
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(tmp_req), "-q"], check=True)
    tmp_req.unlink(missing_ok=True)


def patch_yolov5_torch_load() -> None:
    """Ensure PyTorch 2.6+, NumPy 2.0+, and Pillow 10+ compatibility in YOLOv5."""
    patch_snippet = '''# Compatibility patches for PyTorch 2.6+, NumPy 2.0+, Pillow 10+
try:
    import torch
    _orig_torch_load = torch.load
    def _compat_torch_load(*args, **kwargs):
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return _orig_torch_load(*args, **kwargs)
    torch.load = _compat_torch_load
except Exception:
    pass
'''
    for fname in ["train.py", "val.py", "detect.py"]:
        fpath = YOLO_DIR / fname
        if fpath.exists():
            content = fpath.read_text(encoding="utf-8")
            if "_compat_torch_load" not in content:
                fpath.write_text(patch_snippet + "\n" + content, encoding="utf-8")

    # Pillow font.getsize patch in utils/plots.py
    plots_py = YOLO_DIR / "utils" / "plots.py"
    if plots_py.exists():
        content = plots_py.read_text(encoding="utf-8")
        if "self.font.getsize(label)" in content and "getbbox" not in content:
            content = content.replace(
                "w, h = self.font.getsize(label)",
                "try:\n                    w, h = self.font.getsize(label)\n                except AttributeError:\n                    _b = self.font.getbbox(label)\n                    w, h = _b[2] - _b[0], _b[3] - _b[1]"
            )
            plots_py.write_text(content, encoding="utf-8")

    # NumPy 2.0 trapz patch in utils/metrics.py
    metrics_py = YOLO_DIR / "utils" / "metrics.py"
    if metrics_py.exists():
        content = metrics_py.read_text(encoding="utf-8")
        if "np.trapz(" in content:
            content = content.replace(
                "ap = np.trapz(np.interp(x, mrec, mpre), x)",
                "_trapz = getattr(np, 'trapezoid', getattr(np, 'trapz', None))\n        ap = _trapz(np.interp(x, mrec, mpre), x)"
            )
            metrics_py.write_text(content, encoding="utf-8")


def make_smoke_subset(data_yaml: Path, n: int = 100, val_n: int = 25, seed: int = 42) -> Path:
    """
    Create a smoke-test data.yaml pointing to a small subset of train and val images.
    Returns the path to the new smoke data.yaml.
    """
    import random
    random.seed(seed)

    with open(data_yaml) as f:
        d = yaml.safe_load(f)

    train_txt = Path(d["train"])
    if not train_txt.exists():
        raise FileNotFoundError(f"train.txt not found: {train_txt}")

    lines = train_txt.read_text(encoding="utf-8").strip().splitlines()
    subset = random.sample(lines, min(n, len(lines)))

    smoke_dir = ROOT / "data" / "smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    smoke_train = smoke_dir / "train.txt"
    smoke_train.write_text("\n".join(subset), encoding="utf-8")

    # val subset for fast smoke test
    val_txt = Path(d["val"])
    val_lines = val_txt.read_text(encoding="utf-8").strip().splitlines() if val_txt.exists() else []
    val_subset = random.sample(val_lines, min(val_n, len(val_lines))) if val_lines else subset[:val_n]
    smoke_val = smoke_dir / "val.txt"
    smoke_val.write_text("\n".join(val_subset), encoding="utf-8")

    smoke_yaml = smoke_dir / "smoke.yaml"
    smoke_data = {
        "train": str(smoke_train.resolve()),
        "val":   str(smoke_val.resolve()),
        "nc":    d["nc"],
        "names": d["names"],
    }
    with open(smoke_yaml, "w") as f:
        yaml.dump(smoke_data, f, default_flow_style=False, sort_keys=False)
    return smoke_yaml


def run_train(
    data_yaml: Path,
    weights: str,
    img_size: int,
    epochs: int,
    batch_size: int,
    workers: int,
    project: str,
    name: str,
    patience: int,
    seed: int,
    device: str,
    extra_args: list[str] | None = None,
) -> Path:
    """
    Call YOLOv5's train.py and return the run directory.
    """
    cmd = [
        sys.executable, str(YOLO_DIR / "train.py"),
        "--data",    str(data_yaml.resolve()),
        "--weights", weights,
        "--img",     str(img_size),
        "--epochs",  str(epochs),
        "--batch",   str(batch_size),
        "--workers", str(workers),
        "--project", project,
        "--name",    name,
        "--patience",str(patience),
        "--seed",    str(seed),
        "--device",  device,
        "--exist-ok",
    ]
    if extra_args:
        cmd.extend(extra_args)

    print(f"\n[train] Running: {' '.join(cmd)}\n")
    t0 = time.time()
    subprocess.run(cmd, check=True, cwd=str(ROOT))
    elapsed = time.time() - t0
    print(f"\n[train] Training completed in {elapsed/60:.1f} min")

    run_dir = ROOT / project / name
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLOv5 for weapon detection.")
    parser.add_argument("--smoke", action="store_true", help="Run smoke test only (2 epochs).")
    parser.add_argument("--no_smoke", action="store_true", help="Skip smoke test.")
    parser.add_argument("--weights", type=str, default=None, help="Model weights (e.g. yolov5n.pt).")
    parser.add_argument("--epochs",  type=int, default=None, help="Override epoch count.")
    parser.add_argument("--name",    type=str, default=None, help="Override run name.")
    parser.add_argument("--data",    type=str, default=None, help="Override data.yaml path.")
    args = parser.parse_args()

    cfg = load_config()

    # Clone & install YOLOv5
    clone_yolov5(commit="v7.0")
    install_yolov5_deps()
    patch_yolov5_torch_load()

    device = get_device()
    if device == "cpu":
        print("\n⚠  NO GPU DETECTED. Full training will be VERY slow on CPU.")
        print("   Recommend: use the Colab notebook at notebooks/train_colab.ipynb")
        print("   Running smoke test to verify pipeline …\n")

    data_yaml = Path(args.data) if args.data else ROOT / "data.yaml"
    if not data_yaml.exists():
        print(f"ERROR: data.yaml not found at {data_yaml}")
        print("Run scripts/audit_split.py first to generate data.yaml and split files.")
        sys.exit(1)

    weights  = args.weights or cfg["train"]["weights"]
    img_size = cfg["train"]["img_size"]
    batch    = cfg["train"]["batch_size"]
    workers  = cfg["train"]["workers"]
    seed     = cfg["train"]["seed"]
    patience = cfg["train"]["patience"]

    # ── Smoke test ────────────────────────────────────────────────────────────
    if not args.no_smoke:
        print("\n" + "="*60)
        smoke_cfg = cfg.get("smoke_test", {})
        smoke_subset = smoke_cfg.get("subset", 100)
        smoke_val_subset = smoke_cfg.get("val_subset", 25)
        smoke_img_size = smoke_cfg.get("img_size", 320)
        smoke_epochs = smoke_cfg.get("epochs", 2)
        smoke_name = smoke_cfg.get("name", "smoke_test")

        print(f"  SMOKE TEST ({smoke_epochs} epochs, {smoke_subset}-image subset, imgsz {smoke_img_size})")
        print("="*60)
        smoke_yaml = make_smoke_subset(data_yaml, n=smoke_subset, val_n=smoke_val_subset, seed=seed)
        run_train(
            data_yaml  = smoke_yaml,
            weights    = weights,
            img_size   = smoke_img_size,
            epochs     = smoke_epochs,
            batch_size = batch,
            workers    = workers,
            project    = cfg["train"]["project"],
            name       = smoke_name,
            patience   = smoke_epochs,
            seed       = seed,
            device     = device,
        )
        print("\n✓ Smoke test PASSED. Pipeline verified.")

    if args.smoke:
        return  # smoke-only mode

    # ── Full training ─────────────────────────────────────────────────────────
    if device == "cpu":
        print("\n⚠  Skipping full training on CPU.")
        print("   Open notebooks/train_colab.ipynb in Google Colab for full training.")
        return

    print("\n" + "="*60)
    name   = args.name   or cfg["train"]["name"]
    epochs = args.epochs or cfg["train"]["epochs"]
    print(f"  FULL TRAINING: {weights} | epochs={epochs} | device={device}")
    print("="*60)

    run_dir = run_train(
        data_yaml  = data_yaml,
        weights    = weights,
        img_size   = img_size,
        epochs     = epochs,
        batch_size = batch,
        workers    = workers,
        project    = cfg["train"]["project"],
        name       = name,
        patience   = patience,
        seed       = seed,
        device     = device,
    )
    best_weights = run_dir / "weights" / "best.pt"
    print(f"\n✓ Training complete. Best weights: {best_weights}")
    print("  Next step: python scripts/evaluate.py")


if __name__ == "__main__":
    main()
