"""Streams an uploaded video frame-by-frame with live mask detection."""
import time

import cv2

from services.model_loader import get_model, get_class_names
from app_utils.helpers import draw_detections

# Shared state so the frontend can poll live FPS/detection numbers
video_state = {
    "path": None,
    "fps": 0.0,
    "detections": 0,
    "status": "idle",  # idle | ready | streaming | finished | error
}


def set_video_path(path: str):
    """Register the uploaded video that /video_feed should stream."""
    video_state["path"] = path
    video_state["status"] = "ready"
    video_state["fps"] = 0.0
    video_state["detections"] = 0


def generate_video_frames():
    """Yield annotated JPEG frames as a multipart stream."""
    path = video_state["path"]
    if not path:
        video_state["status"] = "error"
        return

    model = get_model()
    names = get_class_names(model)

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        video_state["status"] = "error"
        return

    video_state["status"] = "streaming"
    prev_time = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break  # end of video

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model(rgb, size=640)
            detections = results.xyxy[0].cpu().numpy()
            annotated, count = draw_detections(frame, detections, names)

            # Live FPS based on real processing time per frame
            now = time.time()
            video_state["fps"] = round(1.0 / max(now - prev_time, 1e-6), 1)
            prev_time = now
            video_state["detections"] = count

            ok, buffer = cv2.imencode(".jpg", annotated)
            if not ok:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            )
    finally:
        cap.release()
        video_state["status"] = "finished"