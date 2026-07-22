"""Shared helper functions used across routers/services."""
import os
import uuid

import cv2

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".jfif", ".tiff"}
VIDEO_EXTS = {".mp4", ".avi", ".mov"}

UPLOAD_DIR = os.path.join("static", "uploads")
OUTPUT_DIR = os.path.join("static", "outputs")


def is_allowed(filename: str, allowed_exts: set) -> bool:
    """Check whether a filename's extension is in the allowed set."""
    ext = os.path.splitext(filename)[1].lower()
    return ext in allowed_exts


def unique_filename(filename: str) -> str:
    """Generate a collision-free filename, keeping the original extension.

    Accepts either a full filename ('photo.jpg') or a bare extension ('.jpg').
    """
    ext = os.path.splitext(filename)[1].lower()
    if not ext and filename.startswith("."):
        ext = filename.lower()
    return f"{uuid.uuid4().hex}{ext}"


def ensure_dirs():
    """Make sure upload/output folders exist."""
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def _box_color(label: str):
    """Pick a color based on the class label (mask domain aware, generic fallback)."""
    label = label.lower()
    if "without" in label or "no_mask" in label or "no-mask" in label:
        return (0, 0, 255)  # red
    if "incorrect" in label:
        return (0, 165, 255)  # orange
    if "mask" in label:
        return (0, 200, 0)  # green
    return (255, 144, 30)  # blue fallback for unknown classes


def draw_detections(frame, detections, names: dict):
    """Draw boxes + label + confidence on a BGR frame.

    detections: iterable of [x1, y1, x2, y2, conf, cls] (numpy array or tensor rows)
    names: {class_id: class_name}
    Returns (annotated_frame, detection_count).
    """
    count = 0
    for det in detections:
        x1, y1, x2, y2, conf, cls = det[:6]
        x1, y1, x2, y2, cls = int(x1), int(y1), int(x2), int(y2), int(cls)
        label_name = names.get(cls, str(cls))
        label = f"{label_name} {conf:.2f}"
        color = _box_color(label_name)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, max(y1 - th - 8, 0)), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            frame, label, (x1 + 2, max(y1 - 5, 12)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
        )
        count += 1

    return frame, count