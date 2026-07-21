from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import torch
import cv2
import numpy as np
from PIL import Image
import io
import base64

app = FastAPI()

# This was missing — without it, /static/style.css and /static/script.js
# have nothing serving them, hence the 404s.
app.mount("/static", StaticFiles(directory="app/static"), name="static")

templates = Jinja2Templates(directory="app/templates")

# Load YOLOv5 model once at startup
model = torch.hub.load(
    'ultralytics/yolov5',
    'custom',
    path='app/models/mask_yolov5.pt',
    source='github'
)


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(request, "index.html", {})


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):

    image_bytes = await file.read()
    image = np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))

    results = model(image)
    annotated = results.render()[0]

    _, buffer = cv2.imencode(".jpg", cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
    img_base64 = base64.b64encode(buffer).decode("utf-8")

    # Per-class detection counts, e.g. {"mask": 2, "no_mask": 1} —
    # the index.html template already knows how to render these as chips.
    df = results.pandas().xyxy[0]
    counts = df["name"].value_counts().to_dict() if not df.empty else {}

    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "image": img_base64,
            "counts": counts,
        },
    )
