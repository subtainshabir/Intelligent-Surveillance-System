"""
Central configuration for the AI Surveillance System.
All tunable thresholds and paths live here so behavior can be adjusted
without touching business logic.
"""

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODEL_PATH = "models/yolov8n.pt"
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"

# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
# COCO class ids -> readable names. Only these classes are detected/tracked.
TARGET_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}
CONFIDENCE_THRESHOLD = 0.35

# ---------------------------------------------------------------------------
# Stationary detection
# ---------------------------------------------------------------------------
# A vehicle is "stationary" as long as it hasn't drifted more than this many
# pixels from the spot where it last stopped (an anchor point, not just the
# previous frame - see stationary.py for why that distinction matters).
MOVEMENT_THRESHOLD = 15
# The threshold above also scales with the vehicle's own size in pixels
# (effective_threshold = max(MOVEMENT_THRESHOLD, ratio * bbox_diagonal)) so a
# vehicle close to the camera and one far away are judged fairly.
MOVEMENT_THRESHOLD_RATIO = 0.04
# EMA smoothing applied to each vehicle's center before comparing to its
# anchor, to absorb bounding-box jitter without masking real movement.
# Closer to 1.0 = more responsive/less smoothing; closer to 0.0 = smoother/more lag.
CENTER_SMOOTHING_ALPHA = 0.5

STATIONARY_WARNING_SECONDS = 30  # yellow - approaching the alert threshold
STATIONARY_ALERT_SECONDS = 60    # red - flagged as suspicious ("left unattended")
STALE_TRACK_SECONDS = 5.0        # forget a track if unseen for this long

# ---------------------------------------------------------------------------
# Performance (tune these down if detection is too slow on your machine)
# ---------------------------------------------------------------------------
# "auto" picks CUDA > MPS > CPU. Set explicitly to "cpu", "cuda", or "mps" to
# override.
DEVICE = "auto"
# Downscale the frame YOLO runs inference on (in pixels, longest side).
# Smaller = faster but less accurate on small/far vehicles. None = model default.
INFERENCE_IMG_SIZE = 480
# Run detection+tracking every Nth frame instead of every frame; frames in
# between reuse the last known boxes. Raise this (e.g. 2 or 3) on a slow
# CPU-only laptop to keep the stream responsive. Stationary timing is still
# correct either way since it's based on video time, not frames processed.
DETECT_EVERY_N_FRAMES = 1

# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
LINE_THICKNESS = 2
FONT_SCALE = 0.55
JPEG_QUALITY = 80

COLOR_NORMAL = (0, 200, 0)        # green (BGR)
COLOR_WARNING = (0, 210, 255)     # yellow/amber (BGR)
COLOR_SUSPICIOUS = (0, 0, 255)    # red (BGR)

# ---------------------------------------------------------------------------
# Uploads
# ---------------------------------------------------------------------------
ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
MAX_UPLOAD_SIZE_MB = 500

# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------
STREAM_POLL_INTERVAL = 0.01
STATUS_POLL_INTERVAL_MS = 1000  # how often the frontend polls /vehicle_status
