"""Routes for image-based mask detection."""
import os

from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse

from services.image_detection import detect_image
from app_utils.helpers import (
    IMAGE_EXTS,
    UPLOAD_DIR,
    is_allowed,
    unique_filename,
    ensure_dirs,
)

router = APIRouter(prefix="/detect", tags=["image"])


@router.post("/image")
async def detect_image_route(file: UploadFile = File(...)):
    """Accept an image upload, run detection, return annotated result + stats."""
    ensure_dirs()

    if not file.filename or not is_allowed(file.filename, IMAGE_EXTS):
        return JSONResponse(
            status_code=400,
            content={"error": "Unsupported format. Use jpg, jpeg or png."},
        )

    contents = await file.read()
    if not contents:
        return JSONResponse(status_code=400, content={"error": "Empty file uploaded."})

    save_path = os.path.join(UPLOAD_DIR, unique_filename(file.filename))
    with open(save_path, "wb") as f:
        f.write(contents)

    try:
        result = detect_image(save_path)
    except FileNotFoundError:
        return JSONResponse(status_code=500, content={"error": "Model not found on server."})
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Detection failed: {e}"})

    return JSONResponse(content=result)