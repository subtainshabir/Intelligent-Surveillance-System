"""
Person detection (YOLOv8) and tracking (ByteTrack via Ultralytics'
built-in `model.track(...)`).

No custom training required — we just filter YOLO's COCO output to
class 0 ("person").
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np
from ultralytics import YOLO

PERSON_CLASS_ID = 0  # COCO class index for "person"


@dataclass
class PersonDetection:
    track_id: Optional[int]   # None if tracking disabled / not yet assigned
    box: tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float


class PersonDetector:
    """
    Thin wrapper around a pretrained YOLOv8 model, restricted to the
    "person" class, with optional ByteTrack multi-object tracking.
    """

    def __init__(self, model_name: str = "yolov8n.pt", conf: float = 0.4):
        # yolov8n.pt is auto-downloaded by ultralytics on first use.
        self.model = YOLO(model_name)
        self.conf = conf

    def detect_frame(self, frame_bgr: np.ndarray) -> list[PersonDetection]:
        """One-off detection, no tracking (no ID persistence)."""
        results = self.model.predict(
            frame_bgr, classes=[PERSON_CLASS_ID], conf=self.conf, verbose=False
        )
        return self._to_detections(results[0], with_ids=False)

    def track_video(self, video_path: str) -> Iterator[tuple[np.ndarray, list[PersonDetection]]]:
        """
        Stream frame-by-frame detections with persistent track IDs
        (ByteTrack). Yields (frame, detections) per frame.
        """
        stream = self.model.track(
            source=video_path,
            classes=[PERSON_CLASS_ID],
            conf=self.conf,
            tracker="bytetrack.yaml",
            persist=True,
            stream=True,
            verbose=False,
        )
        for result in stream:
            yield result.orig_img, self._to_detections(result, with_ids=True)

    @staticmethod
    def _to_detections(result, with_ids: bool) -> list[PersonDetection]:
        detections: list[PersonDetection] = []
        boxes = result.boxes
        if boxes is None:
            return detections

        ids = boxes.id.cpu().numpy() if (with_ids and boxes.id is not None) else None

        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
            conf = float(boxes.conf[i].cpu().numpy())
            track_id = int(ids[i]) if ids is not None else None
            detections.append(
                PersonDetection(track_id=track_id, box=(x1, y1, x2, y2), confidence=conf)
            )
        return detections
