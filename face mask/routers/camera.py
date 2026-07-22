"""Routes for starting, stopping and streaming the live webcam."""
from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse

from services.camera_detection import (
    start_camera,
    stop_camera,
    generate_camera_frames,
    camera_state,
)

router = APIRouter(prefix="/camera", tags=["camera"])


@router.get("/start")
async def camera_start():
    """Open the webcam."""
    ok = start_camera()
    if not ok:
        return JSONResponse(status_code=500, content={"error": "Camera unavailable."})
    return JSONResponse(content={"message": "Camera started."})


@router.get("/stop")
async def camera_stop():
    """Release the webcam."""
    stop_camera()
    return JSONResponse(content={"message": "Camera stopped."})


@router.get("/feed")
async def camera_feed():
    """Live MJPEG stream from the webcam with detections drawn per frame."""
    if not camera_state["active"]:
        return JSONResponse(status_code=400, content={"error": "Camera is not running."})

    return StreamingResponse(
        generate_camera_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.get("/stats")
async def camera_stats():
    """Live FPS/detection/status info for the frontend to poll."""
    return JSONResponse(content=camera_state)
