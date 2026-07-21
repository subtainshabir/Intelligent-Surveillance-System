"""
predictor.py
------------
Wraps the Ultralytics YOLOv8 model in a singleton class so the model
is loaded into memory only once (at application startup) and reused
for every subsequent image/video inference request.
"""

import os
from ultralytics import YOLO

# Path to the trained weights, relative to this file (models/best.pt)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best.pt")


class ShawlPredictor:
    """Singleton wrapper around the YOLOv8 shawl-detection model."""

    _instance = None

    def __init__(self, model_path: str = MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model weights not found at '{model_path}'. "
                "Place your trained 'best.pt' inside the 'models/' folder."
            )
        self.model = YOLO(model_path)

    @classmethod
    def get_instance(cls) -> "ShawlPredictor":
        """Return the single shared instance, creating it if necessary."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def predict_image(self, image, conf: float = 0.25):
        """
        Run detection on a single BGR image (numpy array, as read by cv2).
        Returns an annotated BGR image with bounding boxes drawn.
        """
        results = self.model(image, conf=conf, verbose=False)[0]
        annotated = results.plot()  # returns BGR numpy array with boxes drawn
        return annotated

    def predict_frame(self, frame, conf: float = 0.25):
        """Run detection on a single video frame. Same as predict_image."""
        results = self.model(frame, conf=conf, verbose=False)[0]
        annotated = results.plot()
        return annotated
