from functools import lru_cache

import onnxruntime as ort
from fastapi import HTTPException

from .config import MODEL_DIR, MODELS


@lru_cache(maxsize=None)
def get_session(key: str) -> ort.InferenceSession:
    path = MODEL_DIR / MODELS[key]
    if not path.exists():
        raise HTTPException(503, f"Model file not found: {path.name}")
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def run(key: str, *arrays):
    """Run a model. Input names are read from the model itself."""
    sess = get_session(key)
    feeds = {inp.name: arr for inp, arr in zip(sess.get_inputs(), arrays)}
    return sess.run(None, feeds)


def availability() -> dict:
    return {k: (MODEL_DIR / v).exists() for k, v in MODELS.items()}