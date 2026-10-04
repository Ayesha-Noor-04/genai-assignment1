import time
from typing import Optional

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from PIL import Image

from . import imaging
from .models import run

router = APIRouter(prefix="/api")

PRED_LABELS = ["Clean", "Salt-and-pepper", "Gaussian blur", "Occlusion"]
EXPERTS = ["Identity bypass", "Salt-and-pepper expert", "Blur expert", "Occlusion expert"]
EXPERT_KEYS = [None, "salt", "blur", "occlusion"]


async def prepare(file, sample_id, corruption, severity):
    img = await imaging.load_image(file, sample_id)
    corrupted, settings = imaging.corrupt(img, corruption, severity)
    return corrupted, settings


@router.post("/task1/restore")
async def task1(
    file: Optional[UploadFile] = File(None),
    sample_id: Optional[int] = Form(None),
    corruption: str = Form("none"),
    severity: str = Form("medium"),
):
    corrupted, settings = await prepare(file, sample_id, corruption, severity)
    t = time.perf_counter()
    out = run("universal", imaging.to_input(corrupted))[0]
    ms = round((time.perf_counter() - t) * 1000, 1)
    return {
        "input_image": imaging.to_data_url(corrupted),
        "restored_image": imaging.to_data_url(imaging.out_to_hwc(out)),
        "settings": settings,
        "inference_ms": ms,
    }


@router.post("/task2/restore")
async def task2(
    file: Optional[UploadFile] = File(None),
    sample_id: Optional[int] = Form(None),
    corruption: str = Form("none"),
    severity: str = Form("medium"),
    mode: str = Form("predicted"),
):
    corrupted, _ = await prepare(file, sample_id, corruption, severity)
    x = imaging.to_input(corrupted)

    t = time.perf_counter()
    probs = np.asarray(run("classifier", x)[0][0], dtype=np.float64)
    if abs(probs.sum() - 1.0) > 1e-3:  # model returned logits -> softmax
        e = np.exp(probs - probs.max())
        probs = e / e.sum()

    if mode == "oracle" and corruption != "none":
        idx = imaging.LABEL_INDEX[corruption]
    else:
        idx = int(probs.argmax())

    if EXPERT_KEYS[idx] is None:
        restored = x
    else:
        restored = run(EXPERT_KEYS[idx], x)[0]
    ms = round((time.perf_counter() - t) * 1000, 1)

    return {
        "input_image": imaging.to_data_url(corrupted),
        "restored_image": imaging.to_data_url(imaging.out_to_hwc(restored)),
        "probabilities": [float(p) for p in probs],
        "predicted": PRED_LABELS[int(probs.argmax())],
        "expert": EXPERTS[idx],
        "inference_ms": ms,
    }


@router.post("/task3/restore")
async def task3(
    file: Optional[UploadFile] = File(None),
    sample_id: Optional[int] = Form(None),
    corruption: str = Form("none"),
    severity: str = Form("medium"),
):
    corrupted, _ = await prepare(file, sample_id, corruption, severity)
    t = time.perf_counter()
    restored, weights = run("soft", imaging.to_input(corrupted))[:2]
    ms = round((time.perf_counter() - t) * 1000, 1)
    return {
        "input_image": imaging.to_data_url(corrupted),
        "restored_image": imaging.to_data_url(imaging.out_to_hwc(restored)),
        "weights": [float(w) for w in np.asarray(weights)[0]],
        "inference_ms": ms,
    }


@router.post("/task4/generate")
async def task4(
    file: Optional[UploadFile] = File(None),
    style: int = Form(0),
):
    if style not in (0, 1, 2):
        raise HTTPException(400, "Style must be 0, 1 or 2.")
    img = await imaging.load_image(file, None)

    # --- FIX: sketch model expects 256x256 input ---
    img = img.resize((256, 256), Image.BILINEAR)
    # ------------------------------------------------

    a = np.asarray(img, dtype=np.float32) / 255.0
    photo = np.transpose((a - 0.5) / 0.5, (2, 0, 1))[None].astype(np.float32)

    t = time.perf_counter()
    out = run("sketch", photo, np.array([style], dtype=np.int64))[0]
    ms = round((time.perf_counter() - t) * 1000, 1)

    sketch = (out[0, 0] + 1.0) / 2.0  # [-1,1] -> [0,1], shape HxW
    return {"sketch_image": imaging.to_data_url(sketch), "inference_ms": ms}