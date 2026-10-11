"""
Helper utilities shared across all weapon-detection scripts.
"""
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent.parent

def load_config() -> dict:
    """Load config.yaml from project root."""
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)

def get_device() -> str:
    """Auto-detect best available device."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
