

from __future__ import annotations

import logging
from typing import Optional

from ultralytics import YOLO

import config

logger = logging.getLogger("surveillance.detector")


def _select_device() -> str:
    """Pick the fastest available device unless the user pinned one in config."""
    if config.DEVICE != "auto":
        return config.DEVICE
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
    except Exception:  # pragma: no cover - torch always ships with ultralytics
        pass
    return "cpu"


class VehicleDetector:
    """Loads a YOLO model once and runs class-filtered detection + tracking."""

    def __init__(self, model_path: str = config.MODEL_PATH):
        logger.info("Loading YOLO model from %s", model_path)
        self.model = YOLO(model_path)
        self.target_classes = config.TARGET_CLASSES
        self._class_ids = list(self.target_classes.keys())
        self.device = _select_device()
        logger.info("Running inference on device: %s", self.device)

    def track(self, frame):
        """
        Run detection + tracking on a single BGR frame.

        Returns the first Results object from Ultralytics, or None if the
        model produced no results at all.
        """
        results = self.model.track(
            frame,
            persist=True,
            classes=self._class_ids,
            conf=config.CONFIDENCE_THRESHOLD,
            imgsz=config.INFERENCE_IMG_SIZE,
            device=self.device,
            verbose=False,
        )
        if not results:
            return None
        return results[0]

    def get_class_name(self, class_id) -> str:
        return self.target_classes.get(int(class_id), "vehicle")
