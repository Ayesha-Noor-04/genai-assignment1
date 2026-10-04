import base64
import io
import random
from typing import Optional

import numpy as np
from fastapi import HTTPException, UploadFile
from PIL import Image

from .config import SAMPLE_DIR

S = 128
MAX_BYTES = 10 * 1024 * 1024

LEVELS = {
    "salt": {"low": 0.03, "medium": 0.08, "high": 0.15},
    "blur": {"low": (3, 0.7), "medium": (5, 1.5), "high": (7, 2.5)},
    "occlusion": {"low": (1, 0.10), "medium": (2, 0.20), "high": (3, 0.35)},
}
LABEL_INDEX = {"none": 0, "salt": 1, "blur": 2, "occlusion": 3}


async def load_image(file: Optional[UploadFile], sample_id: Optional[int]) -> Image.Image:
    if file is not None:
        data = await file.read()
        if len(data) > MAX_BYTES:
            raise HTTPException(413, "Image is larger than 10 MB.")
        try:
            img = Image.open(io.BytesIO(data))
            img.load()
        except Exception:
            raise HTTPException(400, "The uploaded file is not a valid image.")
    elif sample_id is not None:
        files = sorted(p for p in SAMPLE_DIR.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
        if not files:
            raise HTTPException(400, "No sample images found in backend/samples.")
        img = Image.open(files[sample_id % len(files)])
    else:
        raise HTTPException(400, "Provide an image file or a sample.")
    return img.convert("RGB").resize((S, S), Image.BICUBIC)


def _salt_pepper(a, p, rng):
    hit = rng.random((S, S)) < p
    white = rng.random((S, S)) < 0.5
    a[hit & white] = 1.0
    a[hit & ~white] = 0.0
    return a


def _blur(a, k, sigma):
    x = np.arange(k) - k // 2
    g = np.exp(-(x ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    r = k // 2
    p = np.pad(a, ((r, r), (r, r), (0, 0)), mode="reflect")
    h = sum(g[i] * p[:, i:i + S, :] for i in range(k))
    return sum(g[i] * h[i:i + S, :, :] for i in range(k))


def _occlude(a, n, frac):
    weights = [random.uniform(0.5, 1.5) for _ in range(n)]
    for w in weights:
        area = frac * S * S * w / sum(weights)
        ar = random.uniform(0.5, 2.0)
        h = min(S, max(2, int((area / ar) ** 0.5)))
        wd = min(S, max(2, int(area / h)))
        x, y = random.randint(0, S - wd), random.randint(0, S - h)
        a[y:y + h, x:x + wd, :] = 0.0
    return a


def corrupt(img: Image.Image, kind: str, severity: str):
    """Returns (corrupted float array HWC in [0,1], settings text)."""
    a = np.asarray(img, dtype=np.float32) / 255.0
    if kind == "none":
        return a, "None (image used as is)"
    if severity not in ("low", "medium", "high") or kind not in LEVELS:
        raise HTTPException(400, "Invalid corruption or severity.")
    lvl = LEVELS[kind][severity]
    if kind == "salt":
        return _salt_pepper(a.copy(), lvl, np.random.default_rng()), f"Salt-and-pepper, p = {lvl}"
    if kind == "blur":
        k, s = lvl
        return np.clip(_blur(a, k, s), 0, 1).astype(np.float32), f"Gaussian blur, kernel {k}, sigma {s}"
    n, frac = lvl
    return _occlude(a.copy(), n, frac), f"Rectangular occlusion, {n} rectangle(s), ~{int(frac * 100)}% area"


def to_input(a: np.ndarray) -> np.ndarray:
    return np.transpose(a, (2, 0, 1))[None].astype(np.float32)


def to_data_url(a: np.ndarray) -> str:
    """a: HWC (or HW) float array in [0,1]."""
    u8 = (np.clip(a, 0, 1) * 255).round().astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(u8).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def out_to_hwc(out: np.ndarray) -> np.ndarray:
    return np.transpose(out[0], (1, 2, 0))