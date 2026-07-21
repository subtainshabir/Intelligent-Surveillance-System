"""
Intelligent Surveillance System - Wrong Direction Detection
FastAPI backend entrypoint.

Flow:
    1. POST /upload      -> save the uploaded video, return a job_id immediately.
    2. GET  /stream/{id} -> MJPEG live stream. Reads the video frame by frame,
                             runs each frame through the detection pipeline,
                             draws boxes/labels, and streams JPEGs to the
                             browser in real time (multipart/x-mixed-replace).
                             The frontend just points an <img> tag at this URL
                             and it plays like a live feed.
    3. GET  /alerts/{id} -> returns the wrong-direction alerts logged so far
                             for that job. The frontend polls this while the
                             stream plays, so the alert log fills in live.

The real computer-vision pipeline (YOLOv8 detection, ByteTrack tracking,
movement analysis, wrong-direction logic) lives in app/*.py as empty
placeholder modules — see process_frame() below for exactly where each one
plugs in. Until they're implemented, every frame is streamed through as-is
with a "LIVE" overlay, so the full upload -> live-play -> alert-log pipeline
already works end to end.
"""

import time
import uuid
from pathlib import Path
from typing import Dict, List

import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from app.detection import detect_persons
from app.tracking import CentroidTracker
from app.movement import analyze_movement
from app.wrong_direction import CrowdFlowAnalyzer
from app.video_utils import draw_box, draw_flow_indicator, COLOR_OK, COLOR_ALERT

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

# A track must read as "wrong direction" for this many consecutive frames
# before it's actually confirmed and logged as an alert. This absorbs the
# occasional single-frame noise (a brief ID switch in a crowd, a jittery
# box) that would otherwise trigger a false positive.
CONFIRM_FRAMES = 6

app = FastAPI(title="Wrong Direction Detection System")

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

templates = Jinja2Templates(directory=BASE_DIR / "templates")

# In-memory per-job state. Fine for a single-process demo; swap for Redis /
# a DB if you need multiple workers or persistence across restarts.
JOBS: Dict[str, dict] = {}
# JOBS[job_id] = {
#     "video_path": Path,
#     "alerts": List[dict],   # appended live as frames are processed
#     "status": "streaming" | "done",
# }


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the single-page frontend."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload")
async def upload(video: UploadFile = File(...)):
    """
    Save the uploaded video and hand back a job_id. This is intentionally
    fast (no processing here) — processing happens frame-by-frame once the
    frontend opens the /stream/{job_id} connection, so "Run Detection"
    feels instant and the video appears to play live.
    """
    suffix = Path(video.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    job_id = uuid.uuid4().hex[:12]
    video_path = UPLOAD_DIR / f"{job_id}{suffix}"

    with video_path.open("wb") as buffer:
        while chunk := await video.read(1024 * 1024):
            buffer.write(chunk)

    JOBS[job_id] = {
        "video_path": video_path,
        "alerts": [],
        "status": "streaming",
        "tracker": CentroidTracker(),
        "crowd_analyzer": CrowdFlowAnalyzer(),
        "wrong_streaks": {},        # track_id -> consecutive wrong-direction frame count
        "confirmed_wrong_ids": set(),  # track_ids currently confirmed wrong-direction
    }

    return {"job_id": job_id, "stream_url": f"/stream/{job_id}", "alerts_url": f"/alerts/{job_id}"}


@app.get("/stream/{job_id}")
def stream(job_id: str):
    """
    Live MJPEG stream of the processed video. Each frame is pulled from the
    source video, run through the detection pipeline, annotated, JPEG-encoded,
    and yielded as a multipart chunk — the browser renders each chunk as it
    arrives, so it looks like a live feed rather than a finished file.
    """
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job_id")

    return StreamingResponse(
        frame_generator(job_id, job["video_path"]),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/alerts/{job_id}")
def get_alerts(job_id: str):
    """Return alerts logged so far for a job. Frontend polls this while streaming."""
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job_id")
    return {"status": job["status"], "alerts": job["alerts"]}


def frame_generator(job_id: str, video_path: Path):
    """
    Generator that yields MJPEG multipart chunks, one per processed frame.
    This is where the pipeline runs live, frame by frame, instead of
    processing the whole video up front.
    """
    job = JOBS[job_id]
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_delay = 1.0 / fps
    frame_idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            annotated, new_alerts = process_frame(frame, frame_idx, job_id)
            job["alerts"].extend(new_alerts)

            ok, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ok:
                frame_idx += 1
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            )

            frame_idx += 1
            time.sleep(frame_delay)  # pace playback to the source video's fps
    finally:
        cap.release()
        job["status"] = "done"


def process_frame(frame, frame_idx: int, job_id: str):
    """
    Run one frame through the full detection pipeline and return the
    annotated frame plus any *newly confirmed* wrong-direction alerts.

    Pipeline:
        1. detect_persons(frame)            -> bounding boxes for people this frame
        2. tracker.update(detections)       -> stable track_id + position history per person
        3. analyze_movement(tracks)         -> movement angle per track
        4. crowd_analyzer.update(movements) -> the CROWD's own dominant direction this
                                                frame (computed live, not hardcoded), plus
                                                which individuals are moving against it.
                                                No alerts are produced at all if there's no
                                                real crowd or no coherent collective motion
                                                (e.g. a standing crowd).
        5. debounce                          -> only confirm + draw red once a track has
                                                 read wrong for CONFIRM_FRAMES consecutive
                                                 frames, so one noisy frame can't fire an
                                                 alert on its own.
        6. draw a box per tracked person (green = with the crowd, red = against it) plus
           a small arrow showing the crowd's current dominant direction.
    """
    job = JOBS[job_id]
    tracker: CentroidTracker = job["tracker"]
    crowd_analyzer: CrowdFlowAnalyzer = job["crowd_analyzer"]
    streaks: Dict[int, int] = job["wrong_streaks"]
    confirmed: set = job["confirmed_wrong_ids"]

    detections = detect_persons(frame)
    tracks = tracker.update(detections)
    movements = analyze_movement(tracks)
    flow = crowd_analyzer.update(movements)
    raw_wrong_ids = {a["track_id"] for a in flow["alerts"]}
    movements_by_id = {m["track_id"]: m for m in movements}

    active_ids = {t["track_id"] for t in tracks}
    new_alerts: List[dict] = []

    for track_id in active_ids:
        if track_id in raw_wrong_ids:
            streaks[track_id] = streaks.get(track_id, 0) + 1
        else:
            streaks[track_id] = 0
            confirmed.discard(track_id)  # they fell back in line with the crowd

        if streaks[track_id] >= CONFIRM_FRAMES and track_id not in confirmed:
            confirmed.add(track_id)
            movement = movements_by_id.get(track_id, {})
            new_alerts.append(
                {
                    "track_id": track_id,
                    "angle_deg": movement.get("angle_deg"),
                    "crowd_flow_deg": flow["dominant_deg"],
                    "frame_idx": frame_idx,
                }
            )

    # Forget streak/confirmed state for tracks that left the frame.
    for track_id in list(streaks.keys()):
        if track_id not in active_ids:
            streaks.pop(track_id, None)
            confirmed.discard(track_id)

    for track in tracks:
        is_wrong = track["track_id"] in confirmed
        color = COLOR_ALERT if is_wrong else COLOR_OK
        label = f"#{track['track_id']} WRONG DIR" if is_wrong else f"#{track['track_id']}"
        draw_box(frame, track["bbox"], label, color)

    draw_flow_indicator(frame, flow["dominant_deg"], flow["coherence"], flow["crowd_detected"])

    cv2.putText(
        frame,
        f"LIVE  frame {frame_idx}  people: {len(tracks)}",
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )

    return frame, new_alerts


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)