"""
Ties detection + color analysis together into the "dress-code anomaly"
pipeline described in the spec:

    Detect persons -> track -> crop -> clothing region -> HSV ->
    dominant color -> compare vs expected/majority -> alert.

Two modes:
  - expected_color set  -> anyone NOT matching that color is an alert.
  - expected_color None -> majority vote across all persons in the
                           frame/video decides the "normal" color, and
                           anyone deviating is an alert.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from .color_utils import dominant_color_for_person
from .detector import PersonDetection, PersonDetector

ALERT_COLOR_BGR = (0, 0, 255)     # red box for anomalies
NORMAL_COLOR_BGR = (0, 200, 0)    # green box for compliant people


@dataclass
class PersonResult:
    track_id: Optional[int]
    box: tuple[int, int, int, int]
    color: str
    confidence: float
    is_anomaly: bool


@dataclass
class FrameResult:
    frame_index: int
    persons: list[PersonResult] = field(default_factory=list)
    expected_color: Optional[str] = None  # resolved expected/majority color used


class DressCodePipeline:
    def __init__(
        self,
        expected_color: Optional[str] = None,
        model_name: str = "yolov8n.pt",
        detect_conf: float = 0.4,
        use_tracking: bool = True,
    ):
        """
        expected_color: e.g. "black". If None, majority voting is used
        instead (computed per-frame, or you can pass a precomputed
        global majority via `set_global_expected`).
        """
        self.expected_color = expected_color
        self.detector = PersonDetector(model_name=model_name, conf=detect_conf)
        self.use_tracking = use_tracking

        # For avoiding repeat alerts on the same tracked person within
        # a short window (simple debounce by track_id).
        self._alerted_ids: set[int] = set()

    def set_global_expected(self, color: str) -> None:
        self.expected_color = color

    # ------------------------------------------------------------------
    # Single-frame analysis
    # ------------------------------------------------------------------
    def analyze_frame(
        self, frame_bgr: np.ndarray, detections: list[PersonDetection], frame_index: int = 0
    ) -> FrameResult:
        person_colors: list[tuple[PersonDetection, str, float]] = []

        for det in detections:
            x1, y1, x2, y2 = det.box
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame_bgr.shape[1], x2), min(frame_bgr.shape[0], y2)
            if x2 <= x1 or y2 <= y1:
                continue

            crop = frame_bgr[y1:y2, x1:x2]
            result = dominant_color_for_person(crop)
            person_colors.append((det, result.color, result.confidence))

        # Resolve what "normal" means for this frame.
        expected = self.expected_color
        if expected is None and person_colors:
            counts = Counter(c for _, c, _ in person_colors)
            expected = counts.most_common(1)[0][0]

        persons: list[PersonResult] = []
        for det, color, conf in person_colors:
            is_anomaly = (expected is not None) and (color != expected)
            persons.append(
                PersonResult(
                    track_id=det.track_id,
                    box=det.box,
                    color=color,
                    confidence=conf,
                    is_anomaly=is_anomaly,
                )
            )

        return FrameResult(frame_index=frame_index, persons=persons, expected_color=expected)

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------
    def draw_annotations(self, frame_bgr: np.ndarray, frame_result: FrameResult) -> np.ndarray:
        annotated = frame_bgr.copy()
        for p in frame_result.persons:
            x1, y1, x2, y2 = p.box
            box_color = ALERT_COLOR_BGR if p.is_anomaly else NORMAL_COLOR_BGR
            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)

            label = p.color.upper()
            if p.track_id is not None:
                label = f"ID{p.track_id} {label}"
            if p.is_anomaly:
                label += "  \u26A0 ANOMALY"

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(annotated, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), box_color, -1)
            cv2.putText(
                annotated, label, (x1 + 3, max(12, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA,
            )
        return annotated

    # ------------------------------------------------------------------
    # Debounced alert helper (avoid re-alerting same track every frame)
    # ------------------------------------------------------------------
    def new_alerts(self, frame_result: FrameResult) -> list[PersonResult]:
        fresh = []
        for p in frame_result.persons:
            if not p.is_anomaly:
                continue
            if p.track_id is None or p.track_id not in self._alerted_ids:
                fresh.append(p)
                if p.track_id is not None:
                    self._alerted_ids.add(p.track_id)
        return fresh

    # ------------------------------------------------------------------
    # Full video processing
    # ------------------------------------------------------------------
    def process_video(
        self, video_path: str, output_path: str
    ) -> dict:
        """
        Runs the full pipeline over a video file, writes an annotated
        output video, and returns a JSON-serializable summary.
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # First pass color tally, used only when no expected_color is
        # given, so the "majority" reflects the whole video rather than
        # flip-flopping frame to frame.
        color_tally: Counter = Counter()
        per_track_colors: dict[int, Counter] = defaultdict(Counter)

        all_frame_results: list[FrameResult] = []
        frame_index = 0

        for frame, detections in self.detector.track_video(video_path):
            frame_result_pre = self.analyze_frame(frame, detections, frame_index)
            for p in frame_result_pre.persons:
                color_tally[p.color] += 1
                if p.track_id is not None:
                    per_track_colors[p.track_id][p.color] += 1
            all_frame_results.append(frame_result_pre)
            frame_index += 1

        global_majority = color_tally.most_common(1)[0][0] if color_tally else None
        expected = self.expected_color or global_majority

        # Second pass: re-resolve anomaly flags against the *global*
        # expected color (instead of per-frame majority) and render.
        cap = cv2.VideoCapture(video_path)
        alert_log = []
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx >= len(all_frame_results):
                break
            fr = all_frame_results[idx]
            fr.expected_color = expected
            for p in fr.persons:
                p.is_anomaly = (expected is not None) and (p.color != expected)

            annotated = self.draw_annotations(frame, fr)
            writer.write(annotated)

            for p in self.new_alerts(fr):
                alert_log.append(
                    {
                        "frame": idx,
                        "track_id": p.track_id,
                        "color": p.color,
                        "confidence": round(p.confidence, 3),
                    }
                )
            idx += 1

        cap.release()
        writer.release()

        # Per-track dominant color (majority across that person's frames)
        # -- useful so one bad frame doesn't mislabel someone.
        track_summary = {
            str(tid): counts.most_common(1)[0][0] for tid, counts in per_track_colors.items()
        }

        return {
            "expected_color": expected,
            "total_frames": frame_index,
            "color_distribution": dict(color_tally),
            "track_dominant_colors": track_summary,
            "alerts": alert_log,
            "output_video": output_path,
        }
