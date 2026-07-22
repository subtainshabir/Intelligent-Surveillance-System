"""Loads the classic YOLOv5 model (torch.hub) once and exposes a single shared instance.

Note: genuine YOLOv5 checkpoints (trained via github.com/ultralytics/yolov5) use a
different architecture registry than the newer 'ultralytics' pip package (YOLOv8+),
so they must be loaded with torch.hub, not ultralytics.YOLO().
"""
import glob
import os

import torch

MODEL_DIR = "model"
LOCAL_YOLOV5_REPO = os.path.join(os.getcwd(), "yolov5")  # optional offline clone

_model = None


def _resolve_model_path() -> str:
    """Find the .pt weights file inside model/, preferring 'best.pt' if present."""
    default = os.path.join(MODEL_DIR, "mask_yolov5.pt")
    if os.path.exists(default):
        return default

    candidates = [
        p for p in glob.glob(os.path.join(MODEL_DIR, "*.pt"))
    ]
    if candidates:
        return candidates[0]

    raise FileNotFoundError(f"No .pt weights file found inside '{MODEL_DIR}/'")


def load_model():
    """Load the model into memory once, at app startup."""
    global _model
    if _model is not None:
        return _model

    weights_path = _resolve_model_path()

    try:
        if os.path.isdir(LOCAL_YOLOV5_REPO):
            # Offline mode: uses a local clone of ultralytics/yolov5
            _model = torch.hub.load(LOCAL_YOLOV5_REPO, "custom", path=weights_path, source="local")
        else:
            # Downloads yolov5 source once from GitHub, then caches it locally
            _model = torch.hub.load("ultralytics/yolov5", "custom", path=weights_path)
    except Exception as e:
        raise RuntimeError(
            "Failed to load the YOLOv5 model via torch.hub. "
            "If this machine has no internet access, clone "
            "https://github.com/ultralytics/yolov5 into the project root as a "
            f"'yolov5' folder and retry. Original error: {e}"
        )

    _model.conf = 0.4  # confidence threshold
    return _model


def get_model():
    """Return the already-loaded model instance."""
    if _model is None:
        raise RuntimeError("Model is not loaded. Server may still be starting.")
    return _model


def get_class_names(model) -> dict:
    """Normalize model.names (list or dict) into a {id: name} dict."""
    names = model.names
    if isinstance(names, dict):
        return names
    return {i: n for i, n in enumerate(names)}
