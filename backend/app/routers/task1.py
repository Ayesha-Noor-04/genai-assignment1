import io

from fastapi import APIRouter, File, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image

from ..services.task1_service import task1_service


router = APIRouter(prefix="/api/task1", tags=["Task 1"])


@router.post("/restore")
async def restore(file: UploadFile = File(...)):
    image_bytes = await file.read()

    try:
        image = Image.open(io.BytesIO(image_bytes))
    except Exception:
        return {"error": "Invalid image file."}

    restored = task1_service.predict(image)

    buffer = io.BytesIO()
    restored.save(buffer, format="PNG")
    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="image/png",
        headers={
            "Content-Disposition": "inline; filename=task1_restored.png"
        },
    )
