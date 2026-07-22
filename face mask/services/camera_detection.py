"""Manages the webcam lifecycle and live mask detection streaming."""
import time

import cv2

from services.model_loader import get_model, get_class_names
from app_utils.helpers import draw_detections

camera_state = {
    "active": False,
    "fps": 0.0,
    "detections": 0,
    "status": "stopped",  # stopped | running | unavailable
}

_cap = None


def start_camera(source: int = 0) -> bool:
    """Open the webcam. Returns False if it cannot be opened."""
    global _cap
    if camera_state["active"]:
        return True

    _cap = cv2.VideoCapture(source)
    if not _cap.isOpened():
        camera_state["status"] = "unavailable"
        return False

    camera_state["active"] = True
    camera_state["status"] = "running"
    return True


def stop_camera():
    """Release the webcam and reset state."""
    global _cap
    camera_state["active"] = False
    camera_state["status"] = "stopped"
    camera_state["fps"] = 0.0
    camera_state["detections"] = 0
    if _cap is not None:
        _cap.release()
        _cap = None


def generate_camera_frames():
    """Yield annotated JPEG frames from the webcam as a multipart stream."""
    model = get_model()
    names = get_class_names(model)
    prev_time = time.time()

    while camera_state["active"] and _cap is not None:
        ok, frame = _cap.read()
        if not ok:
            camera_state["status"] = "unavailable"
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(rgb, size=640)
        detections = results.xyxy[0].cpu().numpy()
        annotated, count = draw_detections(frame, detections, names)

        now = time.time()
        camera_state["fps"] = round(1.0 / max(now - prev_time, 1e-6), 1)
        prev_time = now
        camera_state["detections"] = count

        ok, buffer = cv2.imencode(".jpg", annotated)
        if not ok:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )