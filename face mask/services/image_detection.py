"""Runs mask detection on a single uploaded image."""
import os
import time

import cv2

from services.model_loader import get_model, get_class_names
from app_utils.helpers import OUTPUT_DIR, unique_filename, draw_detections


def detect_image(image_path: str) -> dict:
    """Run inference on an image, save the annotated result, return stats."""
    model = get_model()

    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError("Could not read image. The file may be corrupted.")

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    start = time.time()
    results = model(rgb, size=640)
    inference_time_ms = round((time.time() - start) * 1000, 2)

    detections = results.xyxy[0].cpu().numpy()
    names = get_class_names(model)
    annotated, total_detections = draw_detections(frame, detections, names)

    out_name = unique_filename(".jpg")
    out_path = os.path.join(OUTPUT_DIR, out_name)
    cv2.imwrite(out_path, annotated)

    return {
        "output_image": f"/static/outputs/{out_name}",
        "total_detections": total_detections,
        "inference_time_ms": inference_time_ms,
    }