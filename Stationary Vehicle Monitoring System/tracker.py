"""
VideoProcessor ties together detection, stationary tracking, frame
annotation, and MJPEG frame generation for streaming to the browser.

Processing runs frame-by-frame on a background thread so the HTTP layer
can stream whatever the latest processed frame is without blocking on
the whole video finishing.
"""

from __future__ import annotations

import logging
import math
import os
import threading
import time
from typing import Generator, Optional

import cv2

import config
from detector import VehicleDetector
from stationary import StationaryTracker, STATUS_SUSPICIOUS, STATUS_WARNING

logger = logging.getLogger("surveillance.tracker")


class VideoProcessor:
    """Owns the lifecycle of processing a single uploaded video."""

    def __init__(self, detector: VehicleDetector):
        self.detector = detector
        self.stationary_tracker = StationaryTracker()

        self.video_path: Optional[str] = None
        self.cap: Optional[cv2.VideoCapture] = None

        self.is_processing = False
        self.frame_number = 0
        self.fps = 0.0
        self.vehicles_detected = 0
        self.vehicles_tracked = 0
        self.suspicious_count = 0

        self._frame_lock = threading.Lock()
        self._latest_jpeg: Optional[bytes] = None
        self._stop_event = threading.Event()
        self._finished = threading.Event()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def load_video(self, path: str) -> None:
        self.video_path = path
        self._finished.clear()

    def reset(self) -> None:
        """Stop any active processing and clear all state for a fresh video."""
        self.stop()
        self.stationary_tracker = StationaryTracker()
        self.frame_number = 0
        self.fps = 0.0
        self.vehicles_detected = 0
        self.vehicles_tracked = 0
        self.suspicious_count = 0
        self.video_path = None
        with self._frame_lock:
            self._latest_jpeg = None
        self._stop_event.clear()
        self._finished.clear()

    def stop(self) -> None:
        self._stop_event.set()
        self.is_processing = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _color_for_status(status: str):
        if status == STATUS_SUSPICIOUS:
            return config.COLOR_SUSPICIOUS
        if status == STATUS_WARNING:
            return config.COLOR_WARNING
        return config.COLOR_NORMAL

    def _draw_vehicle(self, frame, box, vehicle_id, class_name, confidence, state):
        x1, y1, x2, y2 = (int(v) for v in box)
        color = self._color_for_status(state.status)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, config.LINE_THICKNESS)

        lines = [f"ID:{vehicle_id} {class_name} {confidence:.2f}"]
        if state.stationary_duration > 0:
            word = "SUSPICIOUS" if state.status == STATUS_SUSPICIOUS else "STATIONARY"
            lines.append(f"{int(state.stationary_duration)} sec {word}")

        ty = max(y1 - 8, 14)
        for line in reversed(lines):
            (tw, th), _ = cv2.getTextSize(
                line, cv2.FONT_HERSHEY_SIMPLEX, config.FONT_SCALE, config.LINE_THICKNESS
            )
            cv2.rectangle(frame, (x1, ty - th - 4), (x1 + tw + 4, ty + 4), (0, 0, 0), -1)
            cv2.putText(
                frame, line, (x1 + 2, ty), cv2.FONT_HERSHEY_SIMPLEX,
                config.FONT_SCALE, color, 1, cv2.LINE_AA,
            )
            ty -= th + 12

    def _draw_overlay_panel(self, frame):
        stationary = self.stationary_tracker.get_stationary_vehicles()
        if not stationary:
            return frame

        h, w = frame.shape[:2]
        panel_w = 220
        panel_h = 30 + 20 * min(len(stationary), 8)
        x0, y0 = w - panel_w - 15, 15

        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (18, 18, 18), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
        cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h), (60, 60, 60), 1)

        cv2.putText(
            frame, "STATIONARY VEHICLES", (x0 + 10, y0 + 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA,
        )

        y = y0 + 40
        for vid, duration in stationary[:8]:
            color = config.COLOR_SUSPICIOUS if duration >= config.STATIONARY_ALERT_SECONDS \
                else config.COLOR_WARNING
            cv2.putText(
                frame, f"ID {vid}   {int(duration)} sec", (x0 + 10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
            )
            y += 20

        return frame

    def _draw_status_bar(self, frame):
        h, w = frame.shape[:2]
        bar_h = 32

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - bar_h), (w, h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        text = (
            f"FPS: {self.fps:.1f}   Frame: {self.frame_number}   "
            f"Detected: {self.vehicles_detected}   Tracked: {self.vehicles_tracked}   "
            f"Suspicious: {self.suspicious_count}"
        )
        cv2.putText(
            frame, text, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
            config.FONT_SCALE, (255, 255, 255), 1, cv2.LINE_AA,
        )
        return frame

    # ------------------------------------------------------------------
    # Main processing loop
    # ------------------------------------------------------------------
    def _video_time_seconds(self, native_fps: float) -> float:
        """Seconds elapsed *in the source video*, not wall-clock time.

        Prefer the container's own timestamp (POS_MSEC); some codecs don't
        report it reliably, so fall back to frame_number / native_fps.
        """
        pos_msec = self.cap.get(cv2.CAP_PROP_POS_MSEC)
        if pos_msec and pos_msec > 0:
            return pos_msec / 1000.0
        return self.frame_number / native_fps

    def process_stream(self) -> None:
        """Runs on a background thread; processes the loaded video frame-by-frame."""
        if not self.video_path or not os.path.exists(self.video_path):
            logger.error("No valid video loaded")
            return

        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            logger.error("Failed to open video: %s", self.video_path)
            self.is_processing = False
            self._finished.set()
            return

        native_fps = self.cap.get(cv2.CAP_PROP_FPS)
        if not native_fps or native_fps <= 1:
            native_fps = 30.0  # sane fallback for videos with bad metadata

        self.is_processing = True
        self._stop_event.clear()
        self._finished.clear()
        prev_wall_time = time.time()
        detect_every_n = max(1, config.DETECT_EVERY_N_FRAMES)

        # Cache of the last frame's drawable detections, reused on skipped
        # frames so the stream still shows boxes even when we don't re-run
        # YOLO every single frame (see DETECT_EVERY_N_FRAMES).
        cached_vehicles = []  # list of (box, vid, class_name, conf)

        try:
            while not self._stop_event.is_set():
                ret, frame = self.cap.read()
                if not ret:
                    break

                self.frame_number += 1
                video_time = self._video_time_seconds(native_fps)
                active_ids = set()
                detected = tracked = suspicious = 0
                run_detection = (self.frame_number % detect_every_n) == 0 or self.frame_number == 1

                if run_detection:
                    result = self.detector.track(frame)
                    cached_vehicles = []

                    if (
                        result is not None
                        and result.boxes is not None
                        and result.boxes.id is not None
                    ):
                        boxes = result.boxes.xyxy.cpu().numpy()
                        ids = result.boxes.id.cpu().numpy().astype(int)
                        confs = result.boxes.conf.cpu().numpy()
                        classes = result.boxes.cls.cpu().numpy().astype(int)

                        detected = len(boxes)
                        tracked = len(ids)

                        for box, vid, conf, cls_id in zip(boxes, ids, confs, classes):
                            vid = int(vid)
                            active_ids.add(vid)
                            x1, y1, x2, y2 = box
                            center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                            bbox_diagonal = math.hypot(x2 - x1, y2 - y1)
                            class_name = self.detector.get_class_name(cls_id)

                            state = self.stationary_tracker.update(
                                vid, center, float(conf), class_name,
                                video_time=video_time, bbox_diagonal=bbox_diagonal,
                            )
                            if state.status == STATUS_SUSPICIOUS:
                                suspicious += 1

                            cached_vehicles.append((box, vid, class_name, float(conf)))

                    self.stationary_tracker.prune(active_ids)
                    self.vehicles_detected = detected
                    self.vehicles_tracked = tracked
                else:
                    # Reuse last known boxes/positions; only refresh the
                    # displayed duration text, since stationary_duration is
                    # keyed off video_time and only advances on real updates.
                    detected = self.vehicles_detected
                    tracked = self.vehicles_tracked

                for box, vid, class_name, conf in cached_vehicles:
                    state = self.stationary_tracker.vehicles.get(vid)
                    if state is None:
                        continue
                    if state.status == STATUS_SUSPICIOUS:
                        suspicious += 1
                    self._draw_vehicle(frame, box, vid, class_name, conf, state)

                self.suspicious_count = suspicious

                now = time.time()
                dt = now - prev_wall_time
                prev_wall_time = now
                self.fps = (1.0 / dt) if dt > 0 else 0.0

                frame = self._draw_overlay_panel(frame)
                frame = self._draw_status_bar(frame)

                ok, buffer = cv2.imencode(
                    ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY]
                )
                if ok:
                    with self._frame_lock:
                        self._latest_jpeg = buffer.tobytes()

                time.sleep(config.STREAM_POLL_INTERVAL)
        finally:
            self.is_processing = False
            self._finished.set()
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            logger.info("Finished processing %s", self.video_path)

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------
    def generate_mjpeg(self) -> Generator[bytes, None, None]:
        """Yields multipart MJPEG chunks of whatever the latest frame is."""
        boundary = b"--frame"
        while True:
            with self._frame_lock:
                jpeg = self._latest_jpeg

            if jpeg is not None:
                yield (
                    boundary + b"\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                )

            if self._finished.is_set() and not self.is_processing:
                break

            time.sleep(config.STREAM_POLL_INTERVAL)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------
    def get_table_data(self):
        return self.stationary_tracker.get_table_rows()
