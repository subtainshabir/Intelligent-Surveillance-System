"""
Video Utilities

Shared drawing helpers used to annotate frames before they're streamed to
the frontend.
"""

import math

import cv2

COLOR_OK = (80, 210, 90)      # green (BGR) - moving with the crowd
COLOR_ALERT = (50, 50, 230)   # red (BGR)   - moving against the crowd
COLOR_FLOW = (60, 200, 255)   # amber (BGR) - crowd flow indicator


def draw_box(frame, bbox, label: str, color=COLOR_OK, thickness: int = 2):
    """Draw a labeled bounding box on a frame in place."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    if label:
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - text_h - 8), (x1 + text_w + 6, y1), color, -1)
        cv2.putText(
            frame, label, (x1 + 3, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA,
        )

    return frame


def draw_flow_indicator(frame, dominant_deg, coherence: float, crowd_detected: bool):
    """
    Draw a small arrow + readout in the bottom-left corner showing the
    crowd's current dominant direction of travel and how unified it is.
    Makes it visible *why* someone did or didn't get flagged.
    """
    h = frame.shape[0]
    origin = (60, h - 40)

    if not crowd_detected or dominant_deg is None:
        cv2.putText(
            frame, "Crowd flow: no clear direction",
            (12, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (170, 170, 170), 1, cv2.LINE_AA,
        )
        return frame

    length = 32
    angle_rad = math.radians(dominant_deg)
    end = (
        int(origin[0] + length * math.cos(angle_rad)),
        int(origin[1] + length * math.sin(angle_rad)),
    )
    cv2.arrowedLine(frame, origin, end, COLOR_FLOW, 2, tipLength=0.35)
    cv2.putText(
        frame,
        f"Crowd flow {dominant_deg:.0f}deg  coherence {coherence:.2f}",
        (origin[0] + 45, origin[1] + 5),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_FLOW, 1, cv2.LINE_AA,
    )
    return frame