"""
FastAPI service for the "dress-code anomaly" pipeline.

Endpoints
---------
GET  /health
POST /analyze/image   - single image, returns JSON + optionally annotated image
POST /analyze/video   - full video, returns JSON summary + downloadable annotated video
GET  /download/{name} - fetch a previously generated output file

Run:
    uvicorn app.main:app --reload --port 8000
"""

from __future__ import annotations

import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .pipeline import DressCodePipeline

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "storage" / "uploads"
OUTPUT_DIR = BASE_DIR / "storage" / "outputs"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VALID_COLORS = {
    "black", "white", "gray", "red", "orange", "yellow",
    "green", "cyan", "blue", "purple", "pink",
}

app = FastAPI(
    title="Dress-Code Anomaly Detection API",
    description="Detects people whose clothing color deviates from an expected "
                 "or majority dress-code color (e.g. spotting non-black attire "
                 "at a funeral or formal event).",
    version="1.0.0",
)
app.mount("/files", StaticFiles(directory=str(OUTPUT_DIR)), name="files")

# In-memory registry of video processing sessions: video_id -> session dict.
# Each session tracks the source file, the requested/resolved expected
# color, a running alert log, and a status flag the frontend polls.
_video_sessions: dict[str, dict] = {}
_sessions_lock = threading.Lock()


def get_pipeline(expected_color: Optional[str], use_tracking: bool = True) -> DressCodePipeline:
    pipeline = DressCodePipeline(expected_color=expected_color, use_tracking=use_tracking)
    return pipeline


@app.get("/", response_class=HTMLResponse)
def index():
    html_path = Path(__file__).parent / "static" / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze/image")
async def analyze_image(
    file: UploadFile = File(...),
    expected_color: Optional[str] = Form(None),
):
    """
    Analyze a single image. If `expected_color` is omitted, the majority
    color among detected people in the image is used as the baseline.
    """
    if expected_color and expected_color.lower() not in VALID_COLORS:
        raise HTTPException(400, f"expected_color must be one of {sorted(VALID_COLORS)}")

    data = await file.read()
    npimg = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(400, "Could not decode image")

    pipeline = get_pipeline(expected_color.lower() if expected_color else None, use_tracking=False)
    detections = pipeline.detector.detect_frame(frame)
    frame_result = pipeline.analyze_frame(frame, detections)
    annotated = pipeline.draw_annotations(frame, frame_result)

    out_name = f"{uuid.uuid4().hex}.jpg"
    out_path = OUTPUT_DIR / out_name
    cv2.imwrite(str(out_path), annotated)

    return JSONResponse(
        {
            "expected_color": frame_result.expected_color,
            "num_people": len(frame_result.persons),
            "people": [
                {
                    "track_id": p.track_id,
                    "box": p.box,
                    "color": p.color,
                    "confidence": round(p.confidence, 3),
                    "is_anomaly": p.is_anomaly,
                }
                for p in frame_result.persons
            ],
            "annotated_image_url": f"/files/{out_name}",
        }
    )


@app.post("/analyze/video")
async def analyze_video(
    file: UploadFile = File(...),
    expected_color: Optional[str] = Form(None),
):
    """
    Batch mode: analyze a full video, write an annotated .mp4 to disk, and
    return one JSON summary at the end. Use this for automation/scripting.
    For the live-in-browser experience the frontend uses
    /upload/video + /stream/video/{id} instead (see below).
    """
    if expected_color and expected_color.lower() not in VALID_COLORS:
        raise HTTPException(400, f"expected_color must be one of {sorted(VALID_COLORS)}")

    suffix = Path(file.filename).suffix or ".mp4"
    in_name = f"{uuid.uuid4().hex}{suffix}"
    in_path = UPLOAD_DIR / in_name
    with in_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    out_name = f"{uuid.uuid4().hex}_annotated.mp4"
    out_path = OUTPUT_DIR / out_name

    pipeline = get_pipeline(expected_color.lower() if expected_color else None, use_tracking=True)
    summary = pipeline.process_video(str(in_path), str(out_path))
    summary["annotated_video_url"] = f"/files/{out_name}"

    return JSONResponse(summary)


# ---------------------------------------------------------------------------
# Live streaming mode (used by the built-in frontend at "/"):
#   1. POST /upload/video                    -> saves the file, returns a video_id
#   2. GET  /stream/video/{video_id}          -> MJPEG stream, frame by frame,
#                                                 with boxes drawn live as each
#                                                 frame finishes processing.
#   3. GET  /stream/video/{video_id}/status   -> polled by the frontend for
#                                                 the running alert log + state.
# ---------------------------------------------------------------------------

@app.post("/upload/video")
async def upload_video(
    file: UploadFile = File(...),
    expected_color: Optional[str] = Form(None),
):
    if expected_color and expected_color.lower() not in VALID_COLORS:
        raise HTTPException(400, f"expected_color must be one of {sorted(VALID_COLORS)}")

    suffix = Path(file.filename).suffix or ".mp4"
    video_id = uuid.uuid4().hex
    in_path = UPLOAD_DIR / f"{video_id}{suffix}"
    with in_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    with _sessions_lock:
        _video_sessions[video_id] = {
            "path": str(in_path),
            "expected_color": expected_color.lower() if expected_color else None,
            "alerts": [],
            "status": "ready",       # ready -> processing -> done | error
            "frame_index": 0,
            "started_at": None,
            "error": None,
        }

    return {"video_id": video_id}


def _frame_generator(video_id: str):
    session = _video_sessions[video_id]
    session["status"] = "processing"
    session["started_at"] = time.time()
    pipeline = DressCodePipeline(expected_color=session["expected_color"], use_tracking=True)

    try:
        for frame, detections in pipeline.detector.track_video(session["path"]):
            frame_result = pipeline.analyze_frame(frame, detections, frame_index=session["frame_index"])
            annotated = pipeline.draw_annotations(frame, frame_result)

            for p in pipeline.new_alerts(frame_result):
                session["alerts"].append(
                    {
                        "frame": session["frame_index"],
                        "track_id": p.track_id,
                        "color": p.color,
                        "confidence": round(p.confidence, 3),
                    }
                )

            session["frame_index"] += 1
            session["expected_color_resolved"] = frame_result.expected_color

            ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ok:
                continue
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
            )
        session["status"] = "done"
    except Exception as exc:  # keep the stream + status endpoint informative on failure
        session["status"] = "error"
        session["error"] = str(exc)
        raise


@app.get("/stream/video/{video_id}")
def stream_video(video_id: str):
    session = _video_sessions.get(video_id)
    if session is None:
        raise HTTPException(404, "Unknown video_id — upload it first via /upload/video")
    if session["status"] != "ready":
        raise HTTPException(409, f"Session already {session['status']} — upload a new video to restart")

    return StreamingResponse(
        _frame_generator(video_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/stream/video/{video_id}/status")
def stream_status(video_id: str):
    session = _video_sessions.get(video_id)
    if session is None:
        raise HTTPException(404, "Unknown video_id")
    return {
        "status": session["status"],
        "frame_index": session["frame_index"],
        "expected_color": session.get("expected_color_resolved") or session["expected_color"],
        "alerts": session["alerts"],
        "error": session["error"],
    }


@app.get("/download/{filename}")
def download(filename: str):
    path = OUTPUT_DIR / filename
    if not path.exists():
        raise HTTPException(404, "File not found")
    return FileResponse(str(path))
