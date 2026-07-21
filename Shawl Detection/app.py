"""
app.py
------
FastAPI application for Shawl Detection using a YOLOv8 model.

Endpoints:
    GET  /                -> Renders the main UI (upload form + result area)
    POST /predict/image   -> Accepts an image, runs detection, returns result image URL
    POST /predict/video   -> Accepts a video, saves it, returns a streaming URL
    GET  /video_feed      -> MJPEG stream of the processed video (frame-by-frame YOLO)
"""

import os
import shutil
import uuid

import cv2
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from models.predictor import ShawlPredictor
from typing import Optional

# --------------------------------------------------------------------------
# Paths & setup
# --------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALLOWED_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ALLOWED_VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv"}

app = FastAPI(title="Shawl Detection App")

# Static assets (css/js) and processed output files (images) are served directly
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
app.mount("/output", StaticFiles(directory=OUTPUT_DIR), name="output")

templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# Model is loaded once here (module-level global) and reused for every request.
predictor: Optional[ShawlPredictor] = None


@app.on_event("startup")
def load_model() -> None:
    """Load the YOLOv8 model a single time when the app starts."""
    global predictor
    predictor = ShawlPredictor.get_instance()
    print("[startup] YOLOv8 model loaded successfully.")


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.get("/")
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    """Run YOLOv8 detection on an uploaded image and return the result URL."""
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_IMAGE_EXT:
        raise HTTPException(status_code=400, detail="Unsupported image format.")

    uid = uuid.uuid4().hex
    input_path = os.path.join(UPLOAD_DIR, f"{uid}{ext}")
    output_filename = f"{uid}_out.jpg"
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    image = cv2.imread(input_path)
    if image is None:
        raise HTTPException(status_code=400, detail="Could not read the uploaded image.")

    annotated = predictor.predict_image(image)
    cv2.imwrite(output_path, annotated)

    return JSONResponse({"type": "image", "result_url": f"/output/{output_filename}"})


@app.post("/predict/video")
async def upload_video(file: UploadFile = File(...)):
    """Save an uploaded video and return the URL of its live-processed stream."""
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_VIDEO_EXT:
        raise HTTPException(status_code=400, detail="Unsupported video format.")

    uid = uuid.uuid4().hex
    saved_name = f"{uid}{ext}"
    input_path = os.path.join(UPLOAD_DIR, saved_name)

    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return JSONResponse({"type": "video", "stream_url": f"/video_feed?path={saved_name}"})


def generate_frames(video_path: str):
    """Generator that reads a video frame-by-frame, runs YOLOv8, and yields MJPEG chunks."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            annotated = predictor.predict_frame(frame)
            ok, buffer = cv2.imencode(".jpg", annotated)
            if not ok:
                continue

            frame_bytes = buffer.tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )
    finally:
        cap.release()


@app.get("/video_feed")
def video_feed(path: str):
    """Stream the processed video as multipart/x-mixed-replace (MJPEG)."""
    video_path = os.path.join(UPLOAD_DIR, path)
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video not found.")

    return StreamingResponse(
        generate_frames(video_path),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
