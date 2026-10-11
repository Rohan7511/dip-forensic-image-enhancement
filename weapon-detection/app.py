"""
app.py
======
Gradio demo app for weapon detection.

Usage:
    python app.py
    python app.py --share     # create public link

Features:
- Upload an image
- Confidence and IoU sliders
- Shows annotated image with bounding boxes
- Lists detections with confidence scores
- Image-level verdict: "No weapon detected" / "Detected: handgun, knife"
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent
YOLO_DIR = ROOT / "yolov5"
sys.path.insert(0, str(YOLO_DIR))


# ─────────────────────────────────────────────────────────────────────────────
# Config & model
# ─────────────────────────────────────────────────────────────────────────────

def load_config() -> dict:
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


CFG = load_config()
CLASS_NAMES = CFG["dataset"]["class_names"]

# Colour per class (BGR)
CLASS_COLORS = {
    0: (0,   200,  80),  # handgun — green
    1: (0,   120, 255),  # knife   — orange-blue
}


def find_best_weights() -> Path:
    run_dir = ROOT / CFG["train"]["project"] / CFG["train"]["name"]
    w = run_dir / "weights" / "best.pt"
    if w.exists():
        return w
    for p in (ROOT / CFG["train"]["project"]).rglob("best.pt"):
        return p
    return w


# Lazy-loaded model (global)
_model = None
_model_device = None


def get_model(device: str = "cpu"):
    global _model, _model_device
    if _model is not None and _model_device == device:
        return _model

    weights = find_best_weights()
    if not weights.exists():
        raise RuntimeError(
            f"No trained model found at {weights}.\n"
            "Run 'python scripts/train.py' first, or place best.pt in "
            f"{weights.parent}/"
        )

    _model = torch.hub.load(
        str(YOLO_DIR), "custom", path=str(weights),
        source="local", force_reload=False, verbose=False,
    )
    _model.to(device)
    _model_device = device
    return _model


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(pil_image, conf: float, iou: float) -> tuple[np.ndarray, str, str]:
    """
    Args:
        pil_image: PIL.Image from Gradio
        conf: confidence threshold
        iou:  NMS IoU threshold
    Returns:
        (annotated_rgb_np, detections_text, verdict)
    """
    if pil_image is None:
        return None, "No image uploaded.", "—"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = get_model(device)
    model.conf = conf
    model.iou  = iou

    # Convert PIL → numpy BGR
    img_np = np.array(pil_image)
    if img_np.ndim == 2:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    img_rgb = img_np  # keep RGB for model (hub expects RGB)

    results  = model(img_rgb, size=CFG["dataset"]["img_size"])
    df       = results.pandas().xyxy[0]

    annotated = img_np.copy()  # RGB
    detections_lines = []
    found_classes    = set()

    for _, row in df.iterrows():
        cid  = int(row["class"])
        c    = float(row["confidence"])
        x1, y1, x2, y2 = int(row["xmin"]), int(row["ymin"]), int(row["xmax"]), int(row["ymax"])
        name = CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else f"class_{cid}"

        # Color in RGB space
        bgr   = CLASS_COLORS.get(cid, (128, 128, 128))
        color = (bgr[2], bgr[1], bgr[0])  # BGR → RGB

        # Draw box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Label
        label = f"{name}  {c:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(annotated, (x1, max(y1-th-6, 0)), (x1+tw+4, y1), color, -1)
        cv2.putText(annotated, label, (x1+2, max(y1-4, th)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        detections_lines.append(f"• **{name}** — conf: {c:.3f}  |  box: [{x1}, {y1}, {x2}, {y2}]")
        found_classes.add(name)

    if not detections_lines:
        det_text = "_No detections above threshold._"
        verdict  = "✅  No weapon detected"
    else:
        det_text = "\n".join(detections_lines)
        verdict  = "⚠️  Detected: " + ", ".join(sorted(found_classes))

    return annotated, det_text, verdict


# ─────────────────────────────────────────────────────────────────────────────
# Gradio UI
# ─────────────────────────────────────────────────────────────────────────────

def build_ui():
    import gradio as gr

    title = "🔫 Weapon Detection — Forensic Image Analysis"
    description = (
        "Upload a forensic or surveillance image to detect **handguns** and **knives** "
        "using a YOLOv5s model trained on ~4 000 annotated images.\n\n"
        "> ⚠️ **Decision-support tool only.** Results must be reviewed by a qualified forensic expert. "
        "The model may produce false positives on look-alike objects (tools, phones, toys)."
    )

    with gr.Blocks(
        title=title,
        theme=gr.themes.Soft(
            primary_hue="slate",
            secondary_hue="blue",
        ),
        css="""
            #verdict {font-size: 1.4rem; font-weight: 700; padding: 12px 20px;
                      border-radius: 8px; text-align: center;}
            .gr-button-primary {background: #3B82F6 !important;}
        """,
    ) as demo:
        gr.Markdown(f"# {title}")
        gr.Markdown(description)

        with gr.Row():
            with gr.Column(scale=1):
                img_input = gr.Image(
                    type="pil",
                    label="Upload image",
                    sources=["upload", "clipboard"],
                )
                with gr.Accordion("Detection Settings", open=False):
                    conf_slider = gr.Slider(
                        minimum=0.10, maximum=0.90, step=0.05,
                        value=CFG["app"]["default_conf"],
                        label="Confidence threshold",
                        info="Minimum confidence to show a detection",
                    )
                    iou_slider = gr.Slider(
                        minimum=0.10, maximum=0.90, step=0.05,
                        value=CFG["app"]["default_iou"],
                        label="IoU (NMS) threshold",
                        info="Suppress overlapping boxes above this IoU",
                    )
                run_btn = gr.Button("Detect Weapons", variant="primary", size="lg")

            with gr.Column(scale=1):
                img_output  = gr.Image(label="Annotated image", type="numpy")
                verdict_box = gr.Markdown(value="", elem_id="verdict")
                det_text    = gr.Markdown(label="Detections", value="")

        run_btn.click(
            fn=run_inference,
            inputs=[img_input, conf_slider, iou_slider],
            outputs=[img_output, det_text, verdict_box],
        )

        # Run on slider change too (live updates)
        for slider in [conf_slider, iou_slider]:
            slider.change(
                fn=run_inference,
                inputs=[img_input, conf_slider, iou_slider],
                outputs=[img_output, det_text, verdict_box],
            )

        gr.Examples(
            examples=[[str(p), 0.25, 0.45] for p in (ROOT / "results" / "errors" / "false_positives").glob("*.jpg")][:3],
            inputs=[img_input, conf_slider, iou_slider],
            label="Example images (from error analysis)",
        )

        gr.Markdown(
            "---\n"
            "**Model**: YOLOv5s fine-tuned on Weapon Detection Dataset (Kaggle) · "
            "**Classes**: handgun (0), knife (1) · "
            "**Input size**: 640×640 · "
            "**Project**: DIP Case Study — Forensic Image Enhancement"
        )

    return demo


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Weapon detection Gradio app.")
    parser.add_argument("--share",   action="store_true", help="Create public Gradio link.")
    parser.add_argument("--port",    type=int, default=CFG["app"]["port"])
    parser.add_argument("--device",  type=str, default=None)
    args = parser.parse_args()

    # Pre-load model
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[app] Loading model on {device} …")
    try:
        get_model(device)
        print("[app] Model loaded ✓")
    except RuntimeError as e:
        print(f"[app] WARNING: {e}")
        print("[app] The app will start but inference will fail until a model is trained.")

    demo = build_ui()
    demo.launch(
        server_port=args.port,
        share=args.share or CFG["app"]["share"],
        inbrowser=True,
    )


if __name__ == "__main__":
    main()
