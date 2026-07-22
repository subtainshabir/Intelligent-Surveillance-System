"""Routes for video upload and live frame-by-frame streaming."""
import os

from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse, StreamingResponse

from services.video_detection import set_video_path, generate_video_frames, video_state
from app_utils.helpers import (
    VIDEO_EXTS,
    UPLOAD_DIR,
    is_allowed,
    unique_filename,
    ensure_dirs,
)

router = APIRouter(tags=["video"])


@router.post("/detect/video")
async def detect_video_route(file: UploadFile = File(...)):
    """Accept a video upload and register it for streaming (no full processing here)."""
    ensure_dirs()

    if not file.filename or not is_allowed(file.filename, VIDEO_EXTS):
        return JSONResponse(
            status_code=400,
            content={"error": "Unsupported format. Use mp4, avi or mov."},
        )

    contents = await file.read()
    if not contents:
        return JSONResponse(status_code=400, content={"error": "Empty file uploaded."})

    save_path = os.path.join(UPLOAD_DIR, unique_filename(file.filename))
    with open(save_path, "wb") as f:
        f.write(contents)

    set_video_path(save_path)
    return JSONResponse(content={"message": "Video ready. Starting stream."})


@router.get("/video_feed")
async def video_feed():
    """Live MJPEG stream of the uploaded video with detections drawn per frame."""
    if not video_state["path"]:
        return JSONResponse(status_code=400, content={"error": "No video uploaded yet."})

    return StreamingResponse(
        generate_video_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.get("/video/stats")
async def video_stats():
    """Live FPS/detection/status info for the frontend to poll."""
    return JSONResponse(content=video_state)