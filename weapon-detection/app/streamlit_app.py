"""
streamlit_app.py
================
Forensic Image Analysis Dashboard
DIP Case Study | YOLOv5s Weapon Detection + Classical DIP Enhancement

Run:
    streamlit run streamlit_app.py
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pathlib
import platform

# Some checkpoints trained on Linux contain pathlib.PosixPath objects in the
# pickle metadata. On Windows, map that class before importing/loading PyTorch.
if platform.system() == "Windows":
    # Checkpoints trained/saved on Linux can pickle pathlib.PosixPath objects.
    # Python 3.13+ implements these classes in pathlib._local, so aliasing only
    # pathlib.PosixPath is insufficient when the pickle references the internal
    # class path. Map both public and implementation-level names before torch loads.
    pathlib.PosixPath = pathlib.WindowsPath
    pathlib.PurePosixPath = pathlib.PureWindowsPath
    try:
        import pathlib._local as _pathlib_local
        _pathlib_local.PosixPath = pathlib.WindowsPath
        _pathlib_local.PurePosixPath = pathlib.PureWindowsPath
    except (ImportError, AttributeError):
        pass

import sys
import time
import warnings
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import yaml
from PIL import Image

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
YOLO_DIR = ROOT / "yolov5"
sys.path.insert(0, str(YOLO_DIR))

# ─────────────────────────────────────────────────────────────────────────────
# Config & Setup
# ─────────────────────────────────────────────────────────────────────────────

def load_config() -> dict:
    cfg_path = ROOT / "config.yaml"
    if cfg_path.exists():
        with open(cfg_path) as f:
            return yaml.safe_load(f)
    return {
        "dataset": {"class_names": ["Knife", "Handgun"], "img_size": 640},
        "eval": {"conf_threshold": 0.25, "iou_threshold": 0.45},
        "paths": {"weights_dir": "weights"},
        "train": {"project": "runs/train", "name": "yolov5s_baseline"},
    }

CFG = load_config()
CLASS_NAMES = CFG["dataset"]["class_names"]  # ["Knife", "Handgun"]
IMG_SIZE = CFG["dataset"]["img_size"]        # 640

CLASS_COLORS_RGB = {
    0: (50, 220, 120),   # Knife   — mint green
    1: (255, 100, 60),   # Handgun — orange-red
}

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Forensic Image Analysis — DIP Case Study",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS — Dark Forensic Lab Theme
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #0a0d14; color: #e2e8f0; }

[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0f1623 0%, #0d1320 100%);
  border-right: 1px solid #1e2d45;
}
[data-testid="stSidebar"] .stMarkdown h3 {
  color: #60a5fa; font-size: .75rem; font-weight: 600;
  letter-spacing: .12em; text-transform: uppercase; margin-top: 18px; margin-bottom: 8px;
}

.forensic-header {
  background: linear-gradient(135deg, #0f1c2e 0%, #0e2340 50%, #0a1929 100%);
  border: 1px solid #1e3a5f; border-radius: 12px;
  padding: 22px 30px; margin-bottom: 22px; position: relative; overflow: hidden;
}
.forensic-header::before {
  content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
  background: linear-gradient(90deg, #3b82f6, #06b6d4, #10b981);
}
.forensic-header h1 {
  color: #f8fafc; font-size: 1.7rem; font-weight: 700;
  letter-spacing: -.02em; margin: 0 0 4px;
}
.forensic-header .subtitle {
  color: #94a3b8; font-size: .88rem; font-weight: 400; margin-bottom: 12px;
}
.badge {
  display: inline-block; background: #1e3a5f; color: #93c5fd;
  border: 1px solid #2563eb44; border-radius: 20px;
  padding: 3px 12px; font-size: .72rem; font-weight: 500; margin-right: 6px;
}
.badge-model {
  display: inline-block; background: #1e293b; color: #38bdf8;
  border: 1px solid #0284c755; border-radius: 6px;
  padding: 4px 10px; font-size: .75rem; font-family: 'JetBrains Mono', monospace;
}

.metric-card {
  background: #0f1623; border: 1px solid #1e2d45; border-radius: 10px;
  padding: 16px 18px; text-align: center;
}
.metric-card .label {
  font-size: .68rem; font-weight: 600; letter-spacing: .1em;
  text-transform: uppercase; color: #64748b; margin-bottom: 4px;
}
.metric-card .value {
  font-size: 1.7rem; font-weight: 700; color: #f1f5f9;
  font-family: 'JetBrains Mono', monospace; line-height: 1.1;
}
.metric-card .unit { font-size: .7rem; color: #475569; margin-top: 3px; }

.verdict-danger {
  background: linear-gradient(135deg, #2d0f0f 0%, #1f0a0a 100%);
  border: 1px solid #dc2626; border-radius: 10px;
  padding: 16px 20px; margin: 16px 0; display: flex; align-items: center; gap: 14px;
}
.verdict-danger .icon { font-size: 2rem; }
.verdict-danger .title {
  color: #f87171; font-weight: 700; font-size: 1.15rem; letter-spacing: .05em;
}
.verdict-danger .detail { color: #cbd5e1; font-size: .82rem; margin-top: 2px; }

.verdict-safe {
  background: linear-gradient(135deg, #092015 0%, #06150e 100%);
  border: 1px solid #16a34a; border-radius: 10px;
  padding: 16px 20px; margin: 16px 0; display: flex; align-items: center; gap: 14px;
}
.verdict-safe .icon { font-size: 2rem; }
.verdict-safe .title {
  color: #4ade80; font-weight: 700; font-size: 1.15rem; letter-spacing: .05em;
}
.verdict-safe .detail { color: #cbd5e1; font-size: .82rem; margin-top: 2px; }

.det-card {
  background: #0f1623; border: 1px solid #1e2d45; border-radius: 8px;
  padding: 10px 14px; margin-bottom: 8px; display: flex; align-items: center;
  gap: 12px; font-size: .82rem;
}
.det-dot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }
.det-name { font-weight: 600; color: #e2e8f0; width: 90px; }
.det-conf {
  font-family: 'JetBrains Mono', monospace; color: #38bdf8;
  font-size: .78rem; width: 100px;
}
.det-box {
  font-family: 'JetBrains Mono', monospace; color: #64748b; font-size: .75rem;
}

.section-hdr {
  font-size: .78rem; font-weight: 600; letter-spacing: .1em;
  text-transform: uppercase; color: #60a5fa; margin: 20px 0 10px;
  display: flex; align-items: center; gap: 8px;
}
.section-hdr::after {
  content: ''; flex: 1; height: 1px; background: #1e2d45;
}

.weight-warn {
  background: #1c1505; border: 1px solid #b45309; border-radius: 8px;
  padding: 10px 14px; font-size: .78rem; color: #fbbf24; margin-bottom: 16px;
}

.disclaimer {
  background: #0f1420; border: 1px solid #1e2d45; border-radius: 8px;
  padding: 12px 16px; font-size: .75rem; color: #475569;
  margin-top: 24px; text-align: center; line-height: 1.6;
}
.disclaimer strong { color: #64748b; }
.img-lbl {
  text-align: center; font-size: .72rem; font-weight: 600; letter-spacing: .08em;
  text-transform: uppercase; color: #64748b; padding: 6px 0 10px;
}
.upload-hint {
  background: #0f1623; border: 2px dashed #1e3a5f; border-radius: 12px;
  padding: 48px 32px; text-align: center; color: #475569;
}
.upload-hint .icon { font-size: 3rem; margin-bottom: 12px; }

.stTabs [data-baseweb="tab-list"] {
  background: #0f1623; border-radius: 8px; padding: 4px; gap: 4px; border: 1px solid #1e2d45;
}
.stTabs [data-baseweb="tab"] {
  background: transparent; color: #64748b; border-radius: 6px;
  font-weight: 500; font-size: .85rem; padding: 8px 18px;
}
.stTabs [aria-selected="true"] { background: #1e3a5f !important; color: #60a5fa !important; }
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Model Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _locate_weights() -> tuple[Path, bool]:
    """Locate model weights. Returns (path, is_fallback)."""
    # 1. Custom trained weights in weights/best.pt
    w = ROOT / CFG["paths"].get("weights_dir", "weights") / "best.pt"
    if w.exists() and w.stat().st_size > 1_000_000:
        return w, False

    # 2. Any runs/train/ checkpoint
    runs_root = ROOT / CFG["train"]["project"]
    if runs_root.exists():
        for p in runs_root.rglob("best.pt"):
            if p.stat().st_size > 1_000_000:
                return p, False

    # 3. Pretrained yolov5s.pt
    for candidate in [ROOT / "yolov5s.pt", YOLO_DIR / "yolov5s.pt"]:
        if candidate.exists() and candidate.stat().st_size > 1_000_000:
            return candidate, True

    # 4. Fallback default
    return ROOT / "yolov5s.pt", True


@st.cache_resource(show_spinner=False)
def load_model():
    """Load the trained YOLOv5 checkpoint using the same path as the working notebook."""
    weights, is_fallback = _locate_weights()

    # This dashboard is for the trained Knife/Handgun model. Never silently
    # substitute COCO weights: their class IDs/names are not this project's classes.
    if is_fallback or not weights.is_file():
        return None, False, (
            f"Trained best.pt not found. Expected it at "
            f"'{ROOT / CFG['paths'].get('weights_dir', 'weights') / 'best.pt'}'. "
            "Place your trained checkpoint there and restart the app."
        )

    try:
        import torch

        if not YOLO_DIR.is_dir() or not (YOLO_DIR / "hubconf.py").is_file():
            return None, False, (
                f"Local YOLOv5 repository not found at '{YOLO_DIR}'. "
                "Place/clone the YOLOv5 repository in a folder named 'yolov5' "
                "beside streamlit_app.py."
            )

        # Match the working notebook: torch.hub.load(local_repo, 'custom',
        # path=best.pt, source='local'), then explicitly move to CPU/GPU.
        sys.path.insert(0, str(YOLO_DIR))
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # Repeat the Windows pickle compatibility mapping after importing torch,
        # since some torch/YOLO versions import pathlib symbols during startup.
        if platform.system() == "Windows":
            pathlib.PosixPath = pathlib.WindowsPath
            pathlib.PurePosixPath = pathlib.PureWindowsPath
            try:
                import pathlib._local as _pathlib_local
                _pathlib_local.PosixPath = pathlib.WindowsPath
                _pathlib_local.PurePosixPath = pathlib.PureWindowsPath
            except (ImportError, AttributeError):
                pass

        model = torch.hub.load(
            str(YOLO_DIR),
            "custom",
            path=str(weights.resolve()),
            source="local",
            force_reload=False,
            verbose=False,
        )
        model.to(device)
        model.eval()
        return model, False, None
    except Exception as e:
        import traceback
        details = traceback.format_exc()
        return None, False, f"Model load error: {e}\n\n{details}"

# ─────────────────────────────────────────────────────────────────────────────
# Classical DIP Helpers
# ─────────────────────────────────────────────────────────────────────────────

def apply_clahe(rgb: np.ndarray, clip_limit: float = 2.0, tile_size: int = 8) -> np.ndarray:
    """CLAHE on L-channel of CIE LAB space."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def apply_laplacian_sharpen(rgb: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """I_sharp = I - alpha * Laplacian(I)."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap = np.clip(lap, -255, 255)
    out = np.zeros_like(rgb, dtype=np.float64)
    for c in range(3):
        out[:, :, c] = rgb[:, :, c].astype(np.float64) - alpha * lap
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_canny(rgb: np.ndarray, low: int = 50, high: int = 150) -> np.ndarray:
    """Gaussian -> Sobel -> NMS -> Hysteresis Canny Edge Detection."""
    blurred = cv2.GaussianBlur(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), (5, 5), 0)
    edges = cv2.Canny(blurred, low, high)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)


def compute_l_histogram(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute histogram of L channel in LAB."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l_channel = lab[:, :, 0]
    hist = cv2.calcHist([l_channel], [0], None, [256], [0, 256]).flatten()
    return l_channel, hist

# ─────────────────────────────────────────────────────────────────────────────
# Inference Logic
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(
    model,
    rgb_img: np.ndarray,
    conf: float,
    iou: float,
    use_clahe_pre: bool,
    is_fallback: bool,
) -> tuple[np.ndarray, list[dict], float]:
    """Run the same YOLOv5 Hub inference flow used in the working notebook."""
    import torch

    inp = apply_clahe(rgb_img) if use_clahe_pre else rgb_img
    model.conf = float(conf)
    model.iou = float(iou)

    t0 = time.perf_counter()
    # Notebook passes an RGB image array to the Hub model and reads the
    # pandas xyxy result. Keep that same behavior for the dashboard.
    with torch.inference_mode():
        results = model(inp, size=int(IMG_SIZE))
    elapsed = (time.perf_counter() - t0) * 1000
    df = results.pandas().xyxy[0]

    annotated = rgb_img.copy()
    detections = []
    weapon_keywords = {"knife", "handgun", "gun", "pistol", "dagger", "sword", "weapon", "firearm"}

    # Resolve the model's own class names rather than assuming an ordering that
    # may differ from config.yaml. For the trained checkpoint, the notebook's
    # `results.pandas().xyxy[0]` supplies the correct class name in each row.
    model_names = getattr(model, "names", {})
    for _, row in df.iterrows():
        cid = int(row["class"])
        confidence = float(row["confidence"])
        x1 = max(0, min(int(row["xmin"]), rgb_img.shape[1] - 1))
        y1 = max(0, min(int(row["ymin"]), rgb_img.shape[0] - 1))
        x2 = max(0, min(int(row["xmax"]), rgb_img.shape[1] - 1))
        y2 = max(0, min(int(row["ymax"]), rgb_img.shape[0] - 1))

        raw_name = str(row.get("name", "")).strip()
        if not raw_name and isinstance(model_names, dict):
            raw_name = str(model_names.get(cid, f"Object #{cid}"))
        elif not raw_name and isinstance(model_names, (list, tuple)) and cid < len(model_names):
            raw_name = str(model_names[cid])
        name = raw_name.strip().capitalize() if raw_name else f"Object #{cid}"

        is_weapon = name.casefold() in weapon_keywords or (not is_fallback and cid < len(CLASS_NAMES))
        color = CLASS_COLORS_RGB.get(cid, (255, 80, 80)) if is_weapon else (100, 150, 240)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{name} {confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        label_top = max(y1 - th - 8, 0)
        cv2.rectangle(annotated, (x1, label_top), (min(x1 + tw + 6, rgb_img.shape[1] - 1), y1), color, -1)
        cv2.putText(
            annotated, label, (x1 + 3, max(y1 - 4, th)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (10, 10, 10), 1, cv2.LINE_AA
        )
        detections.append({
            "class_id": cid,
            "class_name": name,
            "is_weapon": is_weapon,
            "confidence": confidence,
            "box": (x1, y1, x2, y2),
            "color": color,
        })

    return annotated, detections, elapsed

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar Controls
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        """<div style="display:flex;align-items:center;gap:12px;
              padding:0 0 16px;border-bottom:1px solid #1e2d45;margin-bottom:12px;">
            <span style="font-size:1.8rem;">🔍</span>
            <div>
              <div style="font-weight:700;font-size:1.02rem;color:#f1f5f9;">ForensicAI</div>
              <div style="font-size:.7rem;color:#64748b;font-family:monospace;">DIP Case Study · v1.0</div>
            </div>
           </div>""",
        unsafe_allow_html=True,
    )

    st.markdown("### 📂 Input Source")
    input_source = st.radio(
        "Select Image Source:",
        [
            "📸 Case Study CCTV (4.png)",
            "🔫 Sample Handgun Frame",
            "🔪 Sample Knife Frame",
            "📁 Upload Custom Image",
        ],
        index=0,
    )

    uploaded_file = None
    sample_path = None

    samples_dir = ROOT / "sample_images"
    if input_source == "📸 Case Study CCTV (4.png)":
        p = samples_dir / "cctv_parking_lot.png"
        sample_path = p if p.exists() else (ROOT.parent / "4.png")
    elif input_source == "🔫 Sample Handgun Frame":
        sample_path = samples_dir / "sample_handgun.jpg"
    elif input_source == "🔪 Sample Knife Frame":
        sample_path = samples_dir / "sample_knife.jpg"
    else:
        uploaded_file = st.file_uploader(
            "Upload Image (JPG/PNG)",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
        )

    st.markdown("### ⚙️ Detection Settings")
    conf_thresh = st.slider(
        "Confidence Threshold", 0.10, 0.90,
        float(CFG["eval"].get("conf_threshold", 0.25)), 0.05,
        help="Minimum confidence score to render detection",
    )
    iou_thresh = st.slider(
        "IoU (NMS) Threshold", 0.10, 0.90,
        float(CFG["eval"].get("iou_threshold", 0.45)), 0.05,
        help="Non-maximum suppression threshold for overlapping boxes",
    )

    st.markdown("### 🔬 DIP Preprocessing")
    use_clahe = st.toggle(
        "CLAHE Preprocessing", value=True,
        help="Applies Contrast Limited Adaptive Histogram Equalization to L-channel before YOLOv5s inference",
    )
    if use_clahe:
        st.markdown(
            """<div style="background:#0a1f12;border:1px solid #166534;border-radius:6px;
                 padding:7px 10px;font-size:.72rem;color:#4ade80;margin-top:-4px;">
               ✓ CLAHE Active: <code>clipLimit=2.0, tile=8×8</code><br>Luminance channel (CIE LAB)
               </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown(
        """<div style="font-size:.68rem;color:#475569;line-height:1.7;">
           <b style="color:#64748b;">Architecture:</b> YOLOv5s (small)<br>
           <b style="color:#64748b;">Classes:</b> Handgun · Knife<br>
           <b style="color:#64748b;">Resolution:</b> 640×640 px<br>
           <b style="color:#64748b;">Preproc:</b> CIE LAB L-channel CLAHE<br>
           <b style="color:#64748b;">Ablation:</b> Gaussian / Blur / Dark
           </div>""",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Header Banner
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    """<div class="forensic-header">
         <h1>🔍 Forensic Image Analysis Dashboard</h1>
         <div class="subtitle">Digital Image Processing Case Study · YOLOv5s Weapon Detection & Classical DIP Enhancement</div>
         <div>
           <span class="badge">YOLOv5s Detector</span>
           <span class="badge">Handgun & Knife</span>
           <span class="badge">CLAHE Preprocessing</span>
           <span class="badge">Ablation Analysis</span>
           <span class="badge">CCTV Forensics</span>
         </div>
       </div>""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Load Model & Weights
# ─────────────────────────────────────────────────────────────────────────────

with st.spinner("Initializing detection engine…"):
    model, is_fallback, warn_msg = load_model()

if model is None:
    st.error(f"Failed to load YOLOv5 model: {warn_msg}")
    st.stop()

if warn_msg:
    st.warning(warn_msg)

if warn_msg:
    st.markdown(f'<div class="weight-warn">⚠️ {warn_msg}</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Load Image from Selection or Upload
# ─────────────────────────────────────────────────────────────────────────────

current_img = None
img_source_label = ""

if sample_path and sample_path.exists():
    try:
        pil_img = Image.open(sample_path).convert("RGB")
        current_img = np.array(pil_img)
        img_source_label = sample_path.name
    except Exception as e:
        st.error(f"Error loading sample image: {e}")
elif uploaded_file is not None:
    try:
        pil_img = Image.open(uploaded_file).convert("RGB")
        current_img = np.array(pil_img)
        img_source_label = uploaded_file.name
    except Exception as e:
        st.error(f"Error loading uploaded file: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────

tab_detect, tab_dip = st.tabs([
    "🎯  Forensic Weapon Detection",
    "🔬  DIP Enhancement Analysis",
])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — FORENSIC WEAPON DETECTION
# ═══════════════════════════════════════════════════════════════════════════

with tab_detect:
    if current_img is None:
        st.markdown(
            """<div class="upload-hint">
                 <div class="icon">📷</div>
                 <div style="font-size:1.05rem;font-weight:600;color:#cbd5e1;margin-bottom:6px;">No Image Selected</div>
                 <div>Select a sample test image or upload an image from the sidebar to start analysis.</div>
               </div>""",
            unsafe_allow_html=True,
        )
    else:
        # Run inference
        annotated, detections, elapsed_ms = run_inference(
            model, current_img, conf_thresh, iou_thresh, use_clahe, is_fallback
        )

        weapon_detections = [d for d in detections if d["is_weapon"]]
        other_detections  = [d for d in detections if not d["is_weapon"]]
        
        # ── Top Metrics Row ───────────────────────────────────────────
        mc1, mc2, mc3, mc4 = st.columns(4)
        peak_conf = max((d["confidence"] for d in detections), default=0.0)
        found_classes = sorted(set(d["class_name"] for d in detections))
        cls_str = " · ".join(found_classes) if found_classes else "None"

        with mc1:
            st.markdown(
                f"""<div class="metric-card">
                      <div class="label">Total Detections</div>
                      <div class="value">{len(detections)}</div>
                      <div class="unit">{"weapon(s) found" if weapon_detections else "objects found"}</div>
                    </div>""",
                unsafe_allow_html=True,
            )
        with mc2:
            st.markdown(
                f"""<div class="metric-card">
                      <div class="label">Inference Time</div>
                      <div class="value">{elapsed_ms:.0f}</div>
                      <div class="unit">milliseconds (CPU)</div>
                    </div>""",
                unsafe_allow_html=True,
            )
        with mc3:
            st.markdown(
                f"""<div class="metric-card">
                      <div class="label">Peak Confidence</div>
                      <div class="value">{peak_conf:.2f}</div>
                      <div class="unit">max detection score</div>
                    </div>""",
                unsafe_allow_html=True,
            )
        with mc4:
            st.markdown(
                f"""<div class="metric-card">
                      <div class="label">Detected Classes</div>
                      <div class="value" style="font-size:1.05rem;padding-top:6px;">{cls_str}</div>
                      <div class="unit">{len(found_classes)} class category</div>
                    </div>""",
                unsafe_allow_html=True,
            )

        # ── Forensic Verdict Badge ─────────────────────────────────────
        if weapon_detections:
            wnames = " · ".join(sorted(set(d["class_name"] for d in weapon_detections)))
            st.markdown(
                f"""<div class="verdict-danger">
                      <div class="icon">⚠️</div>
                      <div>
                        <div class="title">ALERT: WEAPON DETECTED ({wnames.upper()})</div>
                        <div class="detail">
                          Identified <b>{len(weapon_detections)}</b> potential weapon target(s) &nbsp;|&nbsp;
                          Peak confidence: <b>{peak_conf:.3f}</b> &nbsp;|&nbsp;
                          Source: <code>{img_source_label}</code>
                        </div>
                      </div>
                    </div>""",
                unsafe_allow_html=True,
            )
        else:
            other_note = f" (Found {len(other_detections)} non-weapon object(s))" if other_detections else ""
            st.markdown(
                f"""<div class="verdict-safe">
                      <div class="icon">✅</div>
                      <div>
                        <div class="title">NO WEAPON DETECTED</div>
                        <div class="detail">
                          No handgun or knife detected above confidence threshold <b>{conf_thresh:.2f}</b>{other_note}.
                          Target is clear under current threshold parameters.
                        </div>
                      </div>
                    </div>""",
                unsafe_allow_html=True,
            )

        # ── Side-by-Side Visualisation ─────────────────────────────────
        c_left, c_right = st.columns(2)
        with c_left:
            st.markdown(f'<div class="img-lbl">📷 Original Input Frame ({img_source_label})</div>', unsafe_allow_html=True)
            st.image(current_img, width="stretch")
        with c_right:
            lbl = "🎯 YOLOv5s Detection (with CLAHE L-channel preproc)" if use_clahe else "🎯 YOLOv5s Detection (Raw input)"
            st.markdown(f'<div class="img-lbl">{lbl}</div>', unsafe_allow_html=True)
            st.image(annotated, width="stretch")

        # ── Detection Breakdown ────────────────────────────────────────
        if detections:
            st.markdown('<div class="section-hdr">Detection Breakdown</div>', unsafe_allow_html=True)
            for i, det in enumerate(detections):
                x1, y1, x2, y2 = det["box"]
                r, g, b = det["color"]
                hx = f"#{r:02x}{g:02x}{b:02x}"
                area_pct = ((x2 - x1) * (y2 - y1)) / (current_img.shape[0] * current_img.shape[1]) * 100
                w_tag = " [WEAPON]" if det["is_weapon"] else ""
                st.markdown(
                    f"""<div class="det-card">
                          <div class="det-dot" style="background:{hx};"></div>
                          <div class="det-name">#{i+1} {det["class_name"]}{w_tag}</div>
                          <div class="det-conf">conf: {det["confidence"]:.3f}</div>
                          <div class="det-box">
                            Box: [{x1}, {y1}] → [{x2}, {y2}] &nbsp;|&nbsp;
                            Size: {x2-x1}×{y2-y1} px &nbsp;|&nbsp;
                            Rel. Area: {area_pct:.1f}%
                          </div>
                        </div>""",
                    unsafe_allow_html=True,
                )

        # ── Legal & Forensic Disclaimer ────────────────────────────────
        st.markdown(
            """<div class="disclaimer">
                 <strong>⚖️ DECISION-SUPPORT SYSTEM ONLY</strong><br>
                 This software is designed as a computational aid for forensic investigators and image analysts.
                 It is not an automated judicial decision-maker. All bounding boxes, edge contours, and classification verdicts
                 must be corroborated and validated by certified forensic analysts before inclusion in case proceedings.
               </div>""",
            unsafe_allow_html=True,
        )

# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — CLASSICAL DIP ENHANCEMENT ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

with tab_dip:
    if current_img is None:
        st.markdown(
            """<div class="upload-hint">
                 <div class="icon">🔬</div>
                 <div style="font-size:1.05rem;font-weight:600;color:#cbd5e1;margin-bottom:6px;">No Image Selected</div>
                 <div>Select an image from the sidebar to inspect classical Digital Image Processing techniques.</div>
               </div>""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """<p style="color:#94a3b8;font-size:.85rem;margin-bottom:16px;">
                 Interactive 4-way comparison of classical Digital Image Processing (DIP) enhancement techniques applied to the forensic scene.
                 These spatial and frequency domain operations recover degraded surveillance footage before feature extraction and deep learning inference.
               </p>""",
            unsafe_allow_html=True,
        )

        # ── Interactive Parameter Tuning Expander ─────────────────────
        with st.expander("🛠️ Interactive DIP Hyperparameter Tuning", expanded=False):
            p1, p2, p3 = st.columns(3)
            with p1:
                tune_clip = st.slider("CLAHE Clip Limit", 1.0, 5.0, 2.0, 0.5)
                tune_tile = st.selectbox("CLAHE Tile Grid Size", [4, 8, 16], index=1)
            with p2:
                tune_alpha = st.slider("Laplacian Sharpen Factor (α)", 0.2, 2.5, 1.0, 0.1)
            with p3:
                tune_canny_low = st.slider("Canny Low Threshold", 10, 150, 50, 10)
                tune_canny_high = st.slider("Canny High Threshold", 50, 300, 150, 10)

        # Compute enhanced images
        with st.spinner("Computing classical DIP transformations…"):
            clahe_img = apply_clahe(current_img, clip_limit=tune_clip, tile_size=tune_tile)
            sharp_img = apply_laplacian_sharpen(current_img, alpha=tune_alpha)
            canny_img = apply_canny(current_img, low=tune_canny_low, high=tune_canny_high)

        # ── 2×2 Image Comparison Panel ────────────────────────────────
        r1c1, r1c2 = st.columns(2)
        r2c1, r2c2 = st.columns(2)

        with r1c1:
            st.markdown('<div class="img-lbl">① Raw Surveillance Input</div>', unsafe_allow_html=True)
            st.image(current_img, width="stretch")
            st.markdown(
                """<div style="font-size:.73rem;color:#64748b;padding:6px 2px;line-height:1.5;">
                     Raw sensor acquisition. Subject to low-light underexposure, sensor noise,
                     and low dynamic range typical in CCTV parking-lot environments.
                   </div>""",
                unsafe_allow_html=True,
            )

        with r1c2:
            st.markdown(f'<div class="img-lbl">② CLAHE (CIE LAB · L-channel, clip={tune_clip})</div>', unsafe_allow_html=True)
            st.image(clahe_img, width="stretch")
            st.markdown(
                """<div style="font-size:.73rem;color:#64748b;padding:6px 2px;line-height:1.5;">
                     <b>Contrast Limited Adaptive Histogram Equalization</b> on L-channel only.
                     Redistributes luminance while preserving chromaticity (A* and B* channels),
                     preventing color distortion and noise overamplification in dark shadows.
                   </div>""",
                unsafe_allow_html=True,
            )

        with r2c1:
            st.markdown(f'<div class="img-lbl">③ Laplacian Sharpening (α = {tune_alpha:.1f})</div>', unsafe_allow_html=True)
            st.image(sharp_img, width="stretch")
            st.markdown(
                """<div style="font-size:.73rem;color:#64748b;padding:6px 2px;line-height:1.5;">
                     Second-order isotropic derivative filter: <code>I_sharp = I − α·∇²I</code>.
                     Restores high-frequency detail across weapon contours, barrel outlines, and blade edges.
                   </div>""",
                unsafe_allow_html=True,
            )

        with r2c2:
            st.markdown(f'<div class="img-lbl">④ Canny Edge Detection ({tune_canny_low} / {tune_canny_high})</div>', unsafe_allow_html=True)
            st.image(canny_img, width="stretch")
            st.markdown(
                """<div style="font-size:.73rem;color:#64748b;padding:6px 2px;line-height:1.5;">
                     Multi-stage boundary detector: Gaussian smoothing → Sobel gradient →
                     Non-Maximum Suppression (NMS) → Dual-threshold hysteresis. Isolates suspect silhouette and weapon profile.
                   </div>""",
                unsafe_allow_html=True,
            )

        # ── Histogram Comparison (Luminance Channel) ───────────────────
        st.markdown('<div class="section-hdr">Luminance (L-Channel) Histogram Analysis</div>', unsafe_allow_html=True)

        l_orig, hist_orig = compute_l_histogram(current_img)
        l_clahe, hist_clahe = compute_l_histogram(clahe_img)

        fig, ax = plt.subplots(figsize=(10, 3.2), facecolor='#0a0d14')
        ax.set_facecolor('#0f1623')
        ax.plot(hist_orig, color='#60a5fa', linewidth=1.8, label='Original Input (L-channel)')
        ax.plot(hist_clahe, color='#4ade80', linewidth=1.8, label=f'CLAHE Enhanced (clipLimit={tune_clip})')
        ax.fill_between(range(256), hist_orig, color='#60a5fa', alpha=0.15)
        ax.fill_between(range(256), hist_clahe, color='#4ade80', alpha=0.15)

        ax.set_xlim([0, 255])
        ax.set_xlabel('Pixel Intensity (0 = Dark, 255 = Bright)', color='#94a3b8', fontsize=8)
        ax.set_ylabel('Pixel Frequency', color='#94a3b8', fontsize=8)
        ax.tick_params(colors='#64748b', labelsize=8)
        for spine in ax.spines.values():
            spine.set_color('#1e2d45')
        ax.legend(facecolor='#0f1623', edgecolor='#1e2d45', labelcolor='#e2e8f0', fontsize=8)
        ax.grid(True, color='#1e2d45', linestyle='--', linewidth=0.5, alpha=0.6)
        plt.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

        st.markdown(
            """<div style="font-size:.74rem;color:#64748b;margin-top:4px;line-height:1.5;">
                 <strong>DIP Analytical Insight:</strong> The original underexposed CCTV frame clusters pixels predominantly
                 in the lower intensity brackets (0–80). CLAHE flattens and extends this distribution across the full dynamic range [0, 255],
                 recovering previously concealed forensic details without clipping highlights.
               </div>""",
            unsafe_allow_html=True,
        )

        # ── Technique Reference Cards ─────────────────────────────────
        st.markdown('<div class="section-hdr" style="margin-top:28px;">DIP Algorithmic Reference</div>', unsafe_allow_html=True)
        t1, t2, t3, t4 = st.columns(4)
        techniques = [
            ("📸", "Original CCTV", "Raw Acquisition", "Forensic baseline evidence"),
            ("🌈", "CLAHE", "Adaptive HE on L (LAB)", "Low-light recovery & contrast expansion"),
            ("✂️", "Laplacian Sharpen", "∇²I Second-Order Deriv.", "Weapon edge & texture amplification"),
            ("⬛", "Canny Filter", "Gaussian→Sobel→NMS→Hyst", "Suspect & weapon silhouette extraction"),
        ]
        for col, (icon, title, algo, use_case) in zip([t1, t2, t3, t4], techniques):
            with col:
                st.markdown(
                    f"""<div class="metric-card" style="text-align:left;padding:14px 16px;">
                          <div style="font-size:1.3rem;margin-bottom:6px;">{icon}</div>
                          <div style="font-weight:600;font-size:.82rem;color:#e2e8f0;margin-bottom:4px;">{title}</div>
                          <div style="font-family:'JetBrains Mono',monospace;font-size:.67rem;
                               color:#38bdf8;margin-bottom:8px;">{algo}</div>
                          <div style="font-size:.7rem;color:#64748b;">{use_case}</div>
                        </div>""",
                    unsafe_allow_html=True,
                )

        st.markdown(
            """<div class="disclaimer" style="margin-top:24px;">
                 <strong>📚 DIP Course Case Study Reference:</strong>
                 These classical techniques are evaluated as preprocessing ablation steps to measure
                 the quantitative impact on YOLOv5s weapon detection under simulated forensic degradation:
                 Gaussian noise (σ=25), motion blur (5×5 kernel), and low illumination (factor=0.4).
                 Refer to <code>scripts/experiments.py</code> and <code>Digital_Image_Forensics.ipynb</code> for complete experimental benchmarks.
               </div>""",
            unsafe_allow_html=True,
        )
