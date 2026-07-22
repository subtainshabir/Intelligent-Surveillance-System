"""FastAPI entrypoint. Loads the model once at startup and wires up routers."""
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

from services.model_loader import load_model
from routers import image, video, camera

templates = Jinja2Templates(directory="templates")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load YOLO model once, not per-request
    load_model()
    yield


app = FastAPI(title="Face Mask Detection System", lifespan=lifespan)

app.mount("/static", StaticFiles(directory="static"), name="static")

app.include_router(image.router)
app.include_router(video.router)
app.include_router(camera.router)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the dashboard."""
    return templates.TemplateResponse("index.html", {"request": request})
