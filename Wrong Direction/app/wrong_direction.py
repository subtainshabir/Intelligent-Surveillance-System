"""
Wrong Direction Detection (crowd-relative)

This does NOT compare people against a fixed, pre-configured direction.
It compares each person against what the CROWD AROUND THEM is doing right
now — which is what you actually want for something like a panic-rush
scene: nobody configures "the exit is at 90 degrees" in advance, and the
"correct" direction only exists because most people are moving that way.

How it works, per frame:
    1. Take every track's movement angle (from app.movement.analyze_movement).
    2. Compute the crowd's dominant direction using circular statistics
       (a vector average of everyone's direction, not a plain average of
       angles, which breaks near the 180/-180 wraparound).
    3. Compute "coherence" (0-1): how unified the crowd's motion is.
       - Near 1.0: almost everyone is moving the same way (a rush, a stream
         of foot traffic).
       - Near 0.0: motion is scattered / people are standing still and
         drifting randomly — there IS no meaningful "crowd direction" to
         compare against, so nobody gets flagged.
    4. Only when there's both (a) enough moving people and (b) high enough
       coherence do we treat "dominant direction" as real. Then anyone
       whose own angle deviates from it by more than tolerance_deg is
       flagged.
    5. The dominant direction is smoothed frame-to-frame (circular EMA) so
       it doesn't jitter/flip on a noisy single frame.

A standing crowd, or a crowd with no clear collective direction, produces
no alerts at all — there's nothing to be "opposite" to.
"""

import math
from typing import List, Optional


def _angle_diff(a: float, b: float) -> float:
    """Smallest absolute difference between two angles in degrees, 0-180."""
    diff = (a - b + 180) % 360 - 180
    return abs(diff)


class CrowdFlowAnalyzer:
    """
    Stateful — create one instance per video/job and call update() once per
    frame (state is only the smoothed dominant direction, so results don't
    flicker frame to frame).
    """

    def __init__(
        self,
        min_crowd_size: int = 4,     # need at least this many moving people to call it "a crowd"
        min_coherence: float = 0.45,  # 0-1, how unified motion must be to count as a real flow
        tolerance_deg: float = 55.0,  # how far off the crowd flow still counts as "with the crowd"
        smoothing: float = 0.85,      # higher = more stable/slower-to-change dominant direction
    ):
        self.min_crowd_size = min_crowd_size
        self.min_coherence = min_coherence
        self.tolerance_deg = tolerance_deg
        self.smoothing = smoothing
        self.smoothed_dominant_deg: Optional[float] = None

    def update(self, movements: List[dict]) -> dict:
        """
        Args:
            movements: output of app.movement.analyze_movement() for this frame
                       (already excludes near-stationary people)

        Returns:
            {
                "dominant_deg": float or None,   # the crowd's current direction of travel
                "coherence": float,               # 0-1, how unified the motion is
                "crowd_detected": bool,           # whether a real collective flow exists right now
                "alerts": [{"track_id": int, "angle_deg": float, "deviation_deg": float}, ...]
            }
        """
        if len(movements) < self.min_crowd_size:
            return {
                "dominant_deg": self.smoothed_dominant_deg,
                "coherence": 0.0,
                "crowd_detected": False,
                "alerts": [],
            }

        # Circular mean: average unit vectors, not raw angles, so e.g. 179°
        # and -179° correctly average to ~180° instead of ~0°.
        angles_rad = [math.radians(m["angle_deg"]) for m in movements]
        sin_sum = sum(math.sin(a) for a in angles_rad)
        cos_sum = sum(math.cos(a) for a in angles_rad)
        n = len(angles_rad)

        coherence = math.hypot(sin_sum, cos_sum) / n  # resultant vector length, 0-1
        raw_dominant_deg = math.degrees(math.atan2(sin_sum, cos_sum))

        if coherence < self.min_coherence:
            # People are moving, but not together in any one direction
            # (scattered movement / standing around drifting) — there's no
            # real "crowd direction" to be wrong-against this frame.
            return {
                "dominant_deg": self.smoothed_dominant_deg,
                "coherence": round(coherence, 2),
                "crowd_detected": False,
                "alerts": [],
            }

        self._smooth_dominant_direction(raw_dominant_deg)

        alerts = []
        for m in movements:
            deviation = _angle_diff(m["angle_deg"], self.smoothed_dominant_deg)
            if deviation > self.tolerance_deg:
                alerts.append(
                    {
                        "track_id": m["track_id"],
                        "angle_deg": m["angle_deg"],
                        "deviation_deg": round(deviation, 1),
                    }
                )

        return {
            "dominant_deg": round(self.smoothed_dominant_deg, 1),
            "coherence": round(coherence, 2),
            "crowd_detected": True,
            "alerts": alerts,
        }

    def _smooth_dominant_direction(self, raw_deg: float) -> None:
        if self.smoothed_dominant_deg is None:
            self.smoothed_dominant_deg = raw_deg
            return
        # Circular EMA: blend unit vectors, not raw degrees, again to avoid
        # wraparound artifacts (blending 179 and -179 directly would give 0).
        prev_rad = math.radians(self.smoothed_dominant_deg)
        raw_rad = math.radians(raw_deg)
        blended_sin = self.smoothing * math.sin(prev_rad) + (1 - self.smoothing) * math.sin(raw_rad)
        blended_cos = self.smoothing * math.cos(prev_rad) + (1 - self.smoothing) * math.cos(raw_rad)
        self.smoothed_dominant_deg = math.degrees(math.atan2(blended_sin, blended_cos))


# ---------------------------------------------------------------------------
# Optional alternative: fixed-direction mode.
#
# Useful for scenes where you *do* know the correct direction in advance
# regardless of crowd size (e.g. a one-way turnstile corridor that's often
# nearly empty) — NOT used by default, since the panic-rush use case needs
# the crowd-relative logic above instead.
# ---------------------------------------------------------------------------

def detect_wrong_direction_fixed(
    movements: List[dict],
    allowed_direction_deg: float,
    tolerance_deg: float = 55.0,
) -> List[dict]:
    alerts = []
    for m in movements:
        if _angle_diff(m["angle_deg"], allowed_direction_deg) > tolerance_deg:
            alerts.append(m)
    return alerts