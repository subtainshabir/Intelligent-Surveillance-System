"""
Movement Analysis

Computes a movement direction (angle in degrees) for each track from its
recent centroid history. Needs at least 2 points of history to produce a
direction, so brand-new tracks are skipped until they've been seen for a
couple of frames.

Angle convention (image coordinates, y increases downward):
    0°    -> moving right
    90°   -> moving down
    180°  -> moving left
    -90°  -> moving up
"""

import math
from typing import List


MIN_DISPLACEMENT_PX = 4.0  # ignore near-stationary people (jitter, not real movement)


def analyze_movement(tracks: List[dict]) -> List[dict]:
    """
    Args:
        tracks: output of app.tracking.CentroidTracker.update()

    Returns:
        list of movements: [{"track_id": int, "bbox": [...], "angle_deg": float}, ...]
        (tracks without enough history or without meaningful displacement
        are omitted)
    """
    movements = []

    for track in tracks:
        history = track["history"]
        if len(history) < 2:
            continue

        (x1, y1), (x2, y2) = history[0], history[-1]
        dx, dy = x2 - x1, y2 - y1

        if math.hypot(dx, dy) < MIN_DISPLACEMENT_PX:
            continue

        angle_deg = math.degrees(math.atan2(dy, dx))

        movements.append(
            {
                "track_id": track["track_id"],
                "bbox": track["bbox"],
                "angle_deg": round(angle_deg, 1),
            }
        )

    return movements