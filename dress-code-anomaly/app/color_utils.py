

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
from sklearn.cluster import KMeans

# ---------------------------------------------------------------------------
# Named color buckets. Order matters: achromatic checks (black/white/gray)
# are evaluated before hue-based colors.
# ---------------------------------------------------------------------------

COLOR_NAMES = [
    "black", "white", "gray",
    "red", "orange", "yellow", "green",
    "cyan", "blue", "purple", "pink",
]


def classify_hsv_pixel(h: float, s: float, v: float) -> str:

    if v < 65:
        return "black"
    if s < 40:
        if v > 200:
            return "white"
        return "gray"

    # Chromatic -> bucket by hue (OpenCV hue is 0-179)
    if h < 5 or h >= 170:
        return "red"
    if h < 15:
        return "orange"
    if h < 33:
        return "yellow"
    if h < 78:
        return "green"
    if h < 95:
        return "cyan"
    if h < 130:
        return "blue"
    if h < 150:
        return "purple"
    return "pink"


@dataclass
class DominantColorResult:
    color: str
    confidence: float                 # share of clothing pixels matching color
    breakdown: dict[str, float]       # color -> percentage


def extract_clothing_region(
    person_bgr: np.ndarray,
    top_ratio: float = 0.20,
    bottom_ratio: float = 0.55,
) -> np.ndarray:

    h, w = person_bgr.shape[:2]
    if h == 0 or w == 0:
        return person_bgr

    y1 = int(h * top_ratio)
    y2 = int(h * bottom_ratio)
    y1, y2 = max(0, y1), max(y1 + 1, min(h, y2))

    # Also trim a bit off the sides to reduce background bleed for
    # loosely-fit bounding boxes.
    x1 = int(w * 0.15)
    x2 = int(w * 0.85)
    x1, x2 = max(0, x1), max(x1 + 1, min(w, x2))

    return person_bgr[y1:y2, x1:x2]


def find_dominant_color(
    clothing_bgr: np.ndarray,
    k: int = 3,
    sample_max: int = 4000,
) -> DominantColorResult:
    """
    Convert clothing region to HSV, cluster pixels with K-Means,
    and return the dominant named color plus a full breakdown.
    """
    if clothing_bgr.size == 0:
        return DominantColorResult(color="unknown", confidence=0.0, breakdown={})

    hsv = cv2.cvtColor(clothing_bgr, cv2.COLOR_BGR2HSV)
    pixels = hsv.reshape(-1, 3).astype(np.float32)

    # Subsample for speed on large crops.
    if pixels.shape[0] > sample_max:
        idx = np.random.choice(pixels.shape[0], sample_max, replace=False)
        pixels = pixels[idx]

    n_clusters = min(k, max(1, pixels.shape[0]))
    if n_clusters == 1:
        centers = pixels.mean(axis=0, keepdims=True)
        labels = np.zeros(pixels.shape[0], dtype=int)
    else:
        km = KMeans(n_clusters=n_clusters, n_init=4, random_state=42)
        labels = km.fit_predict(pixels)
        centers = km.cluster_centers_

    counts = np.bincount(labels, minlength=n_clusters)
    weights = counts / counts.sum()

    # Map every cluster center to a named color, then aggregate weight
    # across clusters that map to the same name (e.g. two "black" clusters
    # from shadow vs. lit fabric).
    breakdown: dict[str, float] = {}
    for center, weight in zip(centers, weights):
        h, s, v = center
        name = classify_hsv_pixel(h, s, v)
        breakdown[name] = breakdown.get(name, 0.0) + float(weight)

    dominant_name = max(breakdown, key=breakdown.get)
    confidence = breakdown[dominant_name]

    # Sort breakdown for readability (largest share first).
    breakdown = dict(sorted(breakdown.items(), key=lambda kv: kv[1], reverse=True))

    return DominantColorResult(color=dominant_name, confidence=confidence, breakdown=breakdown)


def dominant_color_for_person(person_bgr: np.ndarray) -> DominantColorResult:
    """Convenience wrapper: crop clothing region + find dominant color."""
    region = extract_clothing_region(person_bgr)
    return find_dominant_color(region)
