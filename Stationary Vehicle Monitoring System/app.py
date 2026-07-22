"""
AI Surveillance System - FastAPI application entrypoint.

Endpoints
---------
GET  /                 Dashboard UI
POST /upload            Upload a video file, saved into uploads/
POST /start_detection   Kick off frame-by-frame processing on a background thread
GET  /video_feed        MJPEG stream of the annotated video
GET  /vehicle_status     JSON snapshot of tracked vehicles + status-bar stats
POST /reset              Stop processing, clear state, delete the temp upload
"""

from __future__ import annotations

import logging
import os
import shutil
import threading
import uuid

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import config
from detector import VehicleDetector
from tracker import VideoProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("surveillance.app")

app = FastAPI(title="AI Surveillance System")

os.makedirs(config.UPLOAD_DIR, exist_ok=True)
os.makedirs(config.OUTPUT_DIR, exist_ok=True)
os.makedirs("static", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Loaded once at startup and shared across requests.
detector = VehicleDetector()
processor = VideoProcessor(detector)

# Tracks the currently uploaded file so /reset can clean it up.
_state = {"upload_path": None}
_processing_lock = threading.Lock()


def _validate_extension(filename: str) -> str:
    ext = os.path.splitext(filename)[1].lower()
    if ext not in config.ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(config.ALLOWED_EXTENSIONS))
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {allowed}",
        )
    return ext


def _cleanup_current_upload() -> None:
    path = _state["upload_path"]
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except OSError as exc:
            logger.warning("Failed to remove upload %s: %s", path, exc)
    _state["upload_path"] = None


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    ext = _validate_extension(file.filename)

    # A new upload replaces any previous one.
    processor.reset()
    _cleanup_current_upload()

    unique_name = f"{uuid.uuid4().hex}{ext}"
    dest_path = os.path.join(config.UPLOAD_DIR, unique_name)

    try:
        with open(dest_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as exc:
        logger.exception("Failed to save upload")
        raise HTTPException(status_code=500, detail="Failed to save file") from exc
    finally:
        file.file.close()

    size_mb = os.path.getsize(dest_path) / (1024 * 1024)
    if size_mb > config.MAX_UPLOAD_SIZE_MB:
        os.remove(dest_path)
        raise HTTPException(
            status_code=400,
            detail=f"File exceeds {config.MAX_UPLOAD_SIZE_MB} MB limit",
        )

    _state["upload_path"] = dest_path
    processor.load_video(dest_path)
    logger.info("Uploaded video saved to %s", dest_path)

    return {"status": "ok", "filename": file.filename}


@app.post("/start_detection")
def start_detection():
    if not _state["upload_path"]:
        raise HTTPException(status_code=400, detail="Upload a video first")

    with _processing_lock:
        if processor.is_processing:
            return {"status": "already_running"}

        thread = threading.Thread(target=processor.process_stream, daemon=True)
        thread.start()

    return {"status": "started"}


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        processor.generate_mjpeg(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/vehicle_status")
def vehicle_status():
    return JSONResponse(
        {
            "is_processing": processor.is_processing,
            "fps": round(processor.fps, 1),
            "frame_number": processor.frame_number,
            "vehicles_detected": processor.vehicles_detected,
            "vehicles_tracked": processor.vehicles_tracked,
            "suspicious_count": processor.suspicious_count,
            "vehicles": processor.get_table_data(),
            "warning_threshold": config.STATIONARY_WARNING_SECONDS,
            "alert_threshold": config.STATIONARY_ALERT_SECONDS,
        }
    )


@app.post("/reset")
def reset():
    processor.reset()
    _cleanup_current_upload()
    return {"status": "reset"}


@app.on_event("shutdown")
def shutdown_event():
    processor.stop()
