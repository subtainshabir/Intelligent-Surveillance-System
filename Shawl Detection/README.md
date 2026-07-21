# Shawl Detection App

A simple FastAPI + YOLOv8 web app for detecting shawls in images and videos.

## Setup

1. Place your trained weights at `models/best.pt` (already expected by `models/predictor.py`).
2. Create a virtual environment and install dependencies:

   ```bash
   python -m venv venv
   source venv/bin/activate   # on Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Run the app:

   ```bash
   uvicorn app:app --reload
   ```

4. Open your browser at http://127.0.0.1:8000

## Notes

- The YOLOv8 model is loaded exactly once, in the FastAPI `startup` event, and reused for every request.
- Image uploads are processed synchronously and the annotated result is saved to `output/` and shown as an `<img>`.
- Video uploads are saved to `uploads/`, then streamed back frame-by-frame (with live YOLOv8 detection) via `StreamingResponse` using MJPEG, rendered in the browser with an `<img>` tag pointed at `/video_feed`.
- Uploaded/generated files accumulate in `uploads/` and `output/` — clear them periodically in production.
