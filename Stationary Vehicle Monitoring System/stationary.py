"""
Stationary-vehicle bookkeeping.

Two things had to be fixed here versus a naive first pass:

1. Duration must be measured in *video time*, not wall-clock time. If the
   host machine is slow and YOLO takes 300ms/frame instead of 30ms/frame,
   wall-clock duration would run ~10x too fast relative to what's actually
   happening on screen. The caller passes in a `video_time` (seconds of
   footage elapsed, from the video's own timestamps) and everything here is
   computed against that instead of time.time().

2. Movement must be measured against a fixed *anchor* point, not the
   previous frame. Comparing only to the previous frame lets a vehicle
   creeping at, say, 10px/frame dodge a 15px threshold forever - each single
   step is "not moving" even though it has drifted 150px over 15 frames.
   Instead we keep an anchor position (where the vehicle was when it was
   last confirmed to start sitting still) and reset it - and the timer -
   as soon as cumulative drift from that anchor exceeds the threshold. A
   light EMA smooths out per-frame detection jitter so a truly parked car
   doesn't get bumped by noisy bounding boxes.

This module has no OpenCV/YOLO dependency so it can be tested in isolation.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import config

Center = Tuple[float, float]

STATUS_MOVING = "moving"
STATUS_WARNING = "warning"
STATUS_SUSPICIOUS = "suspicious"


@dataclass
class VehicleState:
    smoothed_center: Center
    anchor_center: Center
    class_name: str
    confidence: float
    stationary_start_time: Optional[float]  # video-time seconds, not wall clock
    stationary_duration: float = 0.0
    status: str = STATUS_MOVING
    last_seen_wall: float = field(default_factory=time.time)  # only for prune()


class StationaryTracker:
    """Thread-safe registry of per-vehicle stationary state."""

    def __init__(
        self,
        movement_threshold: float = config.MOVEMENT_THRESHOLD,
        movement_threshold_ratio: float = config.MOVEMENT_THRESHOLD_RATIO,
        warning_seconds: float = config.STATIONARY_WARNING_SECONDS,
        alert_seconds: float = config.STATIONARY_ALERT_SECONDS,
        stale_seconds: float = config.STALE_TRACK_SECONDS,
        smoothing_alpha: float = config.CENTER_SMOOTHING_ALPHA,
    ):
        self.vehicles: Dict[int, VehicleState] = {}
        self.movement_threshold = movement_threshold
        self.movement_threshold_ratio = movement_threshold_ratio
        self.warning_seconds = warning_seconds
        self.alert_seconds = alert_seconds
        self.stale_seconds = stale_seconds
        self.smoothing_alpha = smoothing_alpha
        self._lock = threading.Lock()

    def update(
        self,
        vehicle_id: int,
        center: Center,
        confidence: float,
        class_name: str,
        video_time: float,
        bbox_diagonal: float = 0.0,
    ) -> VehicleState:
        """Update (or create) the state for one vehicle for the current frame.

        video_time: seconds elapsed *in the video itself* (from the source's
        own timestamps/frame count), not wall-clock time. This is what makes
        stationary duration independent of how fast the machine can process
        frames.
        bbox_diagonal: diagonal size of the current bounding box in pixels,
        used to scale the movement threshold so a vehicle close to the
        camera (large, fast-moving in pixel terms) and one far away (small,
        slow-moving in pixel terms) are judged fairly.
        """
        with self._lock:
            state = self.vehicles.get(vehicle_id)

            if state is None:
                state = VehicleState(
                    smoothed_center=center,
                    anchor_center=center,
                    class_name=class_name,
                    confidence=confidence,
                    stationary_start_time=video_time,
                    stationary_duration=0.0,
                )
                self.vehicles[vehicle_id] = state
                return state

            # Smooth the raw detection center to reduce bounding-box jitter.
            alpha = self.smoothing_alpha
            smoothed = (
                alpha * center[0] + (1 - alpha) * state.smoothed_center[0],
                alpha * center[1] + (1 - alpha) * state.smoothed_center[1],
            )

            effective_threshold = max(
                self.movement_threshold, self.movement_threshold_ratio * bbox_diagonal
            )

            drift = math.hypot(
                smoothed[0] - state.anchor_center[0],
                smoothed[1] - state.anchor_center[1],
            )

            if drift > effective_threshold:
                # Vehicle has moved meaningfully from where it last "stopped".
                state.anchor_center = smoothed
                state.stationary_start_time = video_time
                state.stationary_duration = 0.0
            else:
                if state.stationary_start_time is None:
                    state.stationary_start_time = video_time
                state.stationary_duration = max(
                    0.0, video_time - state.stationary_start_time
                )

            state.smoothed_center = smoothed
            state.confidence = confidence
            state.class_name = class_name
            state.last_seen_wall = time.time()
            state.status = self._status_for_duration(state.stationary_duration)

            return state

    def _status_for_duration(self, duration: float) -> str:
        if duration >= self.alert_seconds:
            return STATUS_SUSPICIOUS
        if duration >= self.warning_seconds:
            return STATUS_WARNING
        return STATUS_MOVING

    def prune(self, active_ids: set) -> None:
        """Drop vehicles that haven't been seen recently (left the frame).

        This uses wall-clock time deliberately - it's about how long we've
        gone without a fresh detection for this id at all, which is a
        real-time concern regardless of video speed.
        """
        now = time.time()
        with self._lock:
            stale = [
                vid
                for vid, state in self.vehicles.items()
                if vid not in active_ids and (now - state.last_seen_wall) > self.stale_seconds
            ]
            for vid in stale:
                del self.vehicles[vid]

    def get_stationary_vehicles(self) -> List[Tuple[int, float]]:
        """Vehicles currently stationary (duration > 0), longest first."""
        with self._lock:
            items = [
                (vid, state.stationary_duration)
                for vid, state in self.vehicles.items()
                if state.stationary_duration > 0
            ]
        return sorted(items, key=lambda item: -item[1])

    def get_table_rows(self) -> List[dict]:
        """Full snapshot for the UI table, longest-stationary first."""
        with self._lock:
            rows = [
                {
                    "id": vid,
                    "class_name": state.class_name,
                    "confidence": round(state.confidence, 2),
                    "stationary_time": round(state.stationary_duration, 1),
                    "status": state.status,
                }
                for vid, state in self.vehicles.items()
            ]
        rows.sort(key=lambda r: -r["stationary_time"])
        return rows

    def suspicious_count(self) -> int:
        with self._lock:
            return sum(
                1 for state in self.vehicles.values() if state.status == STATUS_SUSPICIOUS
            )
