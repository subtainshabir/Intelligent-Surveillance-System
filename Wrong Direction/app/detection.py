"""
Person Detection (YOLOv8)

Runs YOLOv8 on a single frame and returns bounding boxes for people only
(COCO class 0). The model is loaded once and cached, since loading it per
frame would be far too slow for live streaming.
"""

from ultralytics import YOLO

PERSON_CLASS_ID = 0
DEFAULT_CONF_THRESHOLD = 0.4

_model = None


def _get_model():
    global _model
    if _model is None:
        # yolov8n.pt = the small/fast "nano" variant — good enough for
        # real-time streaming. Swap for yolov8s.pt / yolov8m.pt if you
        # want more accuracy and have the GPU/CPU budget for it.
        _model = YOLO("yolov8n.pt")
    return _model


def detect_persons(frame, conf_threshold: float = DEFAULT_CONF_THRESHOLD):
    """
    Detect people in a single BGR frame (as read by cv2.VideoCapture).

    Args:
        frame: numpy array (H, W, 3), BGR.
        conf_threshold: minimum detection confidence to keep.

    Returns:
        list of detections: [{"bbox": [x1, y1, x2, y2], "confidence": float}, ...]
    """
    model = _get_model()
    results = model.predict(
        frame,
        classes=[PERSON_CLASS_ID],
        conf=conf_threshold,
        verbose=False,
    )[0]

    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        confidence = float(box.conf[0])
        detections.append({"bbox": [x1, y1, x2, y2], "confidence": confidence})

    return detections