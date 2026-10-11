"""
scripts/inspect_model.py
========================
Phase 5: Architecture verification.

- Prints layer-by-layer structure with output channels and feature-map sizes
- Counts Conv2d layers in backbone / neck / head
- Verifies channel widths after width_multiple=0.5
- Saves feature-map visualisations for early/mid/late layers on 2 sample images

Usage:
    python scripts/inspect_model.py
    python scripts/inspect_model.py --weights runs/train/yolov5s_baseline/weights/best.pt
    python scripts/inspect_model.py --sample_dir data/raw/  # pick samples from here
"""

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parent.parent
YOLO_DIR = ROOT / "yolov5"
sys.path.insert(0, str(YOLO_DIR))


def load_config() -> dict:
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(weights: Path, device: str = "cpu"):
    """Load a YOLOv5 model from .pt weights using torch.hub or direct load."""
    # Try direct torch.hub load first
    try:
        model = torch.hub.load(
            str(YOLO_DIR), "custom", path=str(weights),
            source="local", force_reload=False, verbose=False,
        )
        model = model.model  # unwrap DetectMultiBackend → actual nn.Module
    except Exception:
        # Fallback: direct torch.load
        ckpt = torch.load(str(weights), map_location=device)
        model = ckpt.get("model", ckpt)
        if hasattr(model, "float"):
            model = model.float()

    model = model.to(device).eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Layer analysis
# ─────────────────────────────────────────────────────────────────────────────

# YOLOv5s layer index boundaries (approximate, based on yaml architecture)
# Backbone: layers 0-9, Neck: 10-23, Head: 24+
# Exact boundaries depend on the specific yaml; we detect by module name.
BACKBONE_MODULE_TYPES = {"Conv", "C3", "SPPF", "Focus", "BottleneckCSP"}
NECK_MODULE_TYPES     = {"C3", "Concat", "Upsample"}
HEAD_MODULE_TYPES     = {"Detect"}

def classify_layer(module_type: str, layer_idx: int, total_layers: int) -> str:
    """
    Standard YOLOv5 architecture boundaries:
    Backbone: layers 0-9 (10 layers)
    Neck (PANet): layers 10-23 (14 layers)
    Head (Detect): layer 24 (or >=24)
    """
    if layer_idx < 10:
        return "backbone"
    elif layer_idx < 24:
        return "neck"
    else:
        return "head"


def count_conv_layers(model: torch.nn.Module) -> dict:
    """
    Count Conv2d layers in backbone / neck / head.
    YOLOv5's model.model is a Sequential of modules; each has an index.
    """
    import torch.nn as nn

    if not hasattr(model, "model"):
        print("  ⚠ model.model attribute not found. Counting ALL Conv2d layers.")
        total = sum(1 for m in model.modules() if isinstance(m, nn.Conv2d))
        return {"backbone": "?", "neck": "?", "head": "?", "total": total}

    layers = list(model.model)
    n = len(layers)

    counts = {"backbone": 0, "neck": 0, "head": 0, "total": 0}
    for idx, layer in enumerate(layers):
        section = classify_layer("", idx, n)
        for m in layer.modules():
            if isinstance(m, nn.Conv2d):
                counts[section] += 1
                counts["total"]  += 1

    return counts


def print_layer_structure(model: torch.nn.Module, img_size: int = 640) -> None:
    """
    Print layer-by-layer structure with output channel sizes,
    using forward hooks to capture feature-map shapes.
    """
    import torch.nn as nn

    shapes = {}
    hooks  = []

    def make_hook(name):
        def hook_fn(module, inp, out):
            if isinstance(out, torch.Tensor):
                shapes[name] = tuple(out.shape)
        return hook_fn

    if hasattr(model, "model"):
        layers = list(model.model)
    else:
        layers = list(model.children())

    for i, layer in enumerate(layers):
        h = layer.register_forward_hook(make_hook(f"layer_{i}"))
        hooks.append(h)

    dummy = torch.zeros(1, 3, img_size, img_size)
    with torch.no_grad():
        try:
            model(dummy)
        except Exception:
            pass  # Detection head may raise—shapes are captured before that

    for h in hooks:
        h.remove()

    print(f"\n{'='*70}")
    print(f"  YOLOv5s Layer Structure (input: 1×3×{img_size}×{img_size})")
    print(f"{'='*70}")
    print(f"  {'Idx':<5} {'Type':<30} {'Output Shape':<25}")
    print(f"  {'-'*60}")
    for i, layer in enumerate(layers):
        ltype = type(layer).__name__
        shape = shapes.get(f"layer_{i}", "—")
        shape_str = "×".join(map(str, shape)) if isinstance(shape, tuple) else "—"
        print(f"  {i:<5} {ltype:<30} {shape_str:<25}")

    print(f"{'='*70}\n")


def verify_channel_widths(model: torch.nn.Module, expected: list[int]) -> None:
    """Check that the unique output channel counts match expected widths."""
    import torch.nn as nn
    channels = sorted(set(
        m.out_channels for m in model.modules()
        if isinstance(m, nn.Conv2d)
    ))
    print(f"\n── Channel widths in Conv2d layers ──")
    print(f"  Found    : {channels}")
    print(f"  Expected : {expected}")
    # Intersect ignoring detection-head channels (num_classes+5)
    core = [c for c in channels if c in expected]
    missing = [c for c in expected if c not in channels]
    if missing:
        print(f"  ⚠ Expected widths NOT found: {missing}")
    else:
        print(f"  ✓ All expected widths present: {expected}")


# ─────────────────────────────────────────────────────────────────────────────
# Feature-map visualisation
# ─────────────────────────────────────────────────────────────────────────────

def save_feature_maps(
    model: torch.nn.Module,
    sample_images: list[Path],
    layer_indices: dict[str, list[int]],
    out_dir: Path,
    img_size: int = 640,
) -> None:
    """
    Save feature-map grid images for early, mid, and late layers.
    """
    import torch.nn as nn

    out_dir.mkdir(parents=True, exist_ok=True)

    if hasattr(model, "model"):
        layers = list(model.model)
    else:
        layers = list(model.children())

    for img_idx, img_p in enumerate(sample_images[:2]):
        img_raw = cv2.imread(str(img_p))
        if img_raw is None:
            print(f"  ⚠ Cannot read {img_p}")
            continue

        # Resize and preprocess
        img_resized = cv2.resize(img_raw, (img_size, img_size))
        inp = torch.from_numpy(img_resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0

        for group, indices in layer_indices.items():
            activations = {}
            hooks = []

            def make_hook(name):
                def fn(m, i, o):
                    if isinstance(o, torch.Tensor):
                        activations[name] = o.detach().cpu()
                return fn

            for li in indices:
                if li < len(layers):
                    h = layers[li].register_forward_hook(make_hook(f"layer_{li}"))
                    hooks.append(h)

            with torch.no_grad():
                try:
                    model(inp)
                except Exception:
                    pass

            for h in hooks:
                h.remove()

            # Plot
            valid = [(k, v) for k, v in activations.items() if v.ndim == 4 and v.shape[1] > 0]
            if not valid:
                continue

            n_layers = len(valid)
            n_cols   = 8  # channels to show per layer
            n_rows   = n_layers

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
            if n_rows == 1:
                axes = [axes]

            for row, (layer_name, feat) in enumerate(valid):
                n_ch = feat.shape[1]
                for col in range(n_cols):
                    ax = axes[row][col] if n_rows > 1 else axes[0][col]
                    if col < n_ch:
                        ch = feat[0, col].numpy()
                        ch = (ch - ch.min()) / (ch.max() - ch.min() + 1e-8)
                        ax.imshow(ch, cmap="viridis")
                        if col == 0:
                            ax.set_ylabel(layer_name, fontsize=7, rotation=0, labelpad=35)
                    else:
                        ax.axis("off")
                    ax.set_xticks([])
                    ax.set_yticks([])

            title = f"Feature Maps — {group} layers | Sample {img_idx+1}: {img_p.name}"
            fig.suptitle(title, fontsize=10, fontweight="bold")
            fig.tight_layout()
            out_path = out_dir / f"sample{img_idx+1}_{group}_layers.png"
            fig.savefig(out_path, dpi=120)
            plt.close(fig)
            print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def find_best_weights(cfg: dict) -> tuple[Path, bool]:
    """Auto-locate best.pt, or fallback to pretrained yolov5s.pt with warning."""
    # 1. Check weights/best.pt
    w_dir = ROOT / cfg["paths"].get("weights_dir", "weights")
    if (w_dir / "best.pt").exists():
        return w_dir / "best.pt", False

    # 2. Check baseline train run
    run_dir = ROOT / cfg["train"]["project"] / cfg["train"]["name"]
    w = run_dir / "weights" / "best.pt"
    if w.exists():
        return w, False

    # 3. Search all train runs
    for p in (ROOT / cfg["train"]["project"]).rglob("best.pt"):
        return p, False

    # 4. Fallback to pretrained yolov5s.pt
    pretrained = ROOT / "yolov5s.pt"
    if not pretrained.exists():
        yolo_pre = ROOT / "yolov5" / "yolov5s.pt"
        if yolo_pre.exists():
            return yolo_pre, True
        # Download pretrained weights
        try:
            import urllib.request
            print("[inspect] Downloading pretrained yolov5s.pt …")
            url = "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt"
            urllib.request.urlretrieve(url, str(pretrained))
            return pretrained, True
        except Exception as e:
            print(f"[inspect] Could not download yolov5s.pt: {e}")
    return pretrained, True


def find_sample_images(cfg: dict, n: int = 2) -> list[Path]:
    """Find sample images from the val set."""
    try:
        data_yaml = ROOT / "data.yaml"
        with open(data_yaml) as f:
            d = yaml.safe_load(f)
        val_txt = Path(d["val"])
        lines = val_txt.read_text(encoding="utf-8").strip().splitlines()
        images = [Path(l.strip()) for l in lines[:n] if l.strip()]
        valid  = [p for p in images if p.exists()]
        if valid:
            return valid[:n]
    except Exception:
        pass
    # Fallback: search data/raw/
    ext = set(cfg["dataset"]["image_extensions"])
    for p in (ROOT / cfg["paths"]["data_raw"]).rglob("*"):
        if p.suffix.lower() in ext:
            yield p
            n -= 1
            if n == 0:
                return


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect YOLOv5 model architecture.")
    parser.add_argument("--weights",    type=str, default=None)
    parser.add_argument("--sample_dir", type=str, default=None,
                        help="Directory to pick sample images from.")
    args = parser.parse_args()

    cfg = load_config()
    is_fallback = False
    if args.weights:
        weights = Path(args.weights)
    else:
        weights, is_fallback = find_best_weights(cfg)

    if not weights.exists():
        print(f"ERROR: weights not found: {weights}")
        print("Run python scripts/train.py first.")
        sys.exit(1)

    if is_fallback:
        print("\n" + "!" * 70)
        print("  WARNING: Trained weights (weights/best.pt) NOT found.")
        print(f"  Falling back to PRETRAINED weights: {weights.name} for architecture inspection.")
        print("!" * 70 + "\n")

    print(f"\n[inspect] Loading model from {weights} …")
    model = load_model(weights)
    print("[inspect] Model loaded ✓")

    img_size = cfg["dataset"]["img_size"]

    # ── Layer structure ───────────────────────────────────────────────────────
    print_layer_structure(model, img_size=img_size)

    # ── Conv2d count ──────────────────────────────────────────────────────────
    counts = count_conv_layers(model)
    reported = cfg["inspect"]
    print(f"\n── Conv2d layer count ──")
    print(f"  Backbone : {counts['backbone']:>5}   (report claims: {reported['reported_conv_backbone']})")
    print(f"  Neck     : {counts['neck']:>5}   (report claims: {reported['reported_conv_neck']})")
    print(f"  Head     : {counts['head']:>5}   (report claims: {reported['reported_conv_head']})")
    print(f"  TOTAL    : {counts['total']:>5}   (report claims: {reported['reported_conv_total']})")

    if counts["total"] != reported["reported_conv_total"]:
        print(f"\n  ⚠ MISMATCH: actual total={counts['total']}, report says {reported['reported_conv_total']}.")
        print("  Update the report with the actual numbers above.")
    else:
        print(f"\n  ✓ Conv2d count matches the report ({reported['reported_conv_total']}).")

    # ── Total parameters ──────────────────────────────────────────────────────
    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n── Parameters ──")
    print(f"  Total      : {total_params:,}  ({total_params/1e6:.2f} M)")
    print(f"  Trainable  : {train_params:,}  ({train_params/1e6:.2f} M)")

    # ── Channel widths ────────────────────────────────────────────────────────
    verify_channel_widths(model, cfg["inspect"]["expected_channel_widths"])

    # ── Feature-map visualisations ────────────────────────────────────────────
    if args.sample_dir:
        sample_dir = Path(args.sample_dir)
        ext = set(cfg["dataset"]["image_extensions"])
        sample_images = [p for p in sample_dir.rglob("*") if p.suffix.lower() in ext][:2]
    else:
        sample_images = list(find_sample_images(cfg, n=2))

    if not sample_images:
        print("\n  ⚠ No sample images found. Skipping feature-map visualisation.")
    else:
        print(f"\n── Feature-map visualisation ──")
        print(f"  Using samples: {[p.name for p in sample_images]}")
        layer_groups = {
            "early": cfg["inspect"]["early_layers"],
            "mid":   cfg["inspect"]["mid_layers"],
            "late":  cfg["inspect"]["late_layers"],
        }
        fm_dir = ROOT / cfg["paths"]["feature_maps_dir"]
        save_feature_maps(model, sample_images, layer_groups, fm_dir, img_size=img_size)

    # ── Summary JSON ──────────────────────────────────────────────────────────
    summary = {
        "weights": str(weights),
        "total_params": total_params,
        "trainable_params": train_params,
        "conv2d_counts": counts,
        "reported_conv_counts": {
            "backbone": reported["reported_conv_backbone"],
            "neck":     reported["reported_conv_neck"],
            "head":     reported["reported_conv_head"],
            "total":    reported["reported_conv_total"],
        },
        "count_matches_report": counts["total"] == reported["reported_conv_total"],
    }
    out_json = ROOT / cfg["paths"]["results_dir"] / "architecture_summary.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    import json
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nArchitecture summary saved → {out_json}")
    print("\n✓ Phase 5 complete.")


if __name__ == "__main__":
    main()
