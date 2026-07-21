"""
Person Tracking

A lightweight centroid tracker: matches each new detection to the closest
existing track (by centroid distance) so people keep a consistent ID across
frames. This keeps the project dependency-light (no extra CUDA/lap
requirements) while giving you real per-person IDs and history.

TODO (optional upgrade): swap this for real ByteTrack, e.g. via the
`supervision` package (`pip install supervision`,
`supervision.ByteTrack`), for more robust tracking through occlusion.
This class's `update()` interface can stay the same if you do.
"""

import math
from typing import Dict, List


class CentroidTracker:
    """Stateful tracker — create one instance per video/job and call
    update() once per frame."""

    def __init__(self, max_distance: float = 80.0, max_missed: int = 15, history_len: int = 10):
        self.next_id = 0
        self.tracks: Dict[int, dict] = {}
        # tracks[id] = {"bbox": [...], "centroid": (x, y), "missed": int, "history": [(x, y), ...]}
        self.max_distance = max_distance
        self.max_missed = max_missed
        self.history_len = history_len

    @staticmethod
    def _centroid(bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @staticmethod
    def _distance(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def update(self, detections: List[dict]) -> List[dict]:
        """
        Match this frame's detections to existing tracks (or start new ones).

        Args:
            detections: output of app.detection.detect_persons()

        Returns:
            list of active tracks this frame:
            [{"track_id": int, "bbox": [...], "centroid": (x, y), "history": [(x, y), ...]}, ...]
        """
        detection_centroids = [self._centroid(d["bbox"]) for d in detections]
        unmatched_detections = set(range(len(detections)))
        matched_track_ids = set()

        # Greedy nearest-centroid matching.
        for track_id, track in self.tracks.items():
            best_idx, best_dist = None, self.max_distance
            for idx in unmatched_detections:
                dist = self._distance(track["centroid"], detection_centroids[idx])
                if dist < best_dist:
                    best_idx, best_dist = idx, dist

            if best_idx is not None:
                bbox = detections[best_idx]["bbox"]
                centroid = detection_centroids[best_idx]
                track["bbox"] = bbox
                track["centroid"] = centroid
                track["missed"] = 0
                track["history"].append(centroid)
                if len(track["history"]) > self.history_len:
                    track["history"].pop(0)
                unmatched_detections.discard(best_idx)
                matched_track_ids.add(track_id)
            else:
                track["missed"] += 1

        # Drop tracks that have gone unmatched for too long.
        self.tracks = {
            tid: t for tid, t in self.tracks.items() if t["missed"] <= self.max_missed
        }

        # Start new tracks for leftover detections.
        for idx in unmatched_detections:
            bbox = detections[idx]["bbox"]
            centroid = detection_centroids[idx]
            self.tracks[self.next_id] = {
                "bbox": bbox,
                "centroid": centroid,
                "missed": 0,
                "history": [centroid],
            }
            self.next_id += 1

        return [
            {
                "track_id": tid,
                "bbox": t["bbox"],
                "centroid": t["centroid"],
                "history": list(t["history"]),
            }
            for tid, t in self.tracks.items()
            if t["missed"] == 0
        ]