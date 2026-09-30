"""Corruption functions for Tasks 1-3 (numpy only, deterministic given explicit parameters).

Images are uint8 arrays of shape [H, W, 3] (H = W = 128).

Corruption definitions follow the assignment table exactly:
  clean            : no corruption
  salt_pepper      : p ~ U(0.02, 0.15); selected pixels -> black or white with equal probability
  blur             : kernel size in {3, 5, 7}; sigma ~ U(0.5, 2.5)
  occlusion        : 1-3 black rectangles jointly covering 10%-35% of the image area

Every function is split in two parts:
  * a *sampler*   (draws random parameters from an RNG)   -> used by the runtime training pipeline
  * an *applier*  (applies explicit parameters, no RNG needed except salt_pepper which takes a seed)
                                                          -> used by the fixed val/test manifests
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

CLASSES = ["clean", "salt_pepper", "blur", "occlusion"]  # label index = position in this list
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}

IMG_SIZE = 128

# ---- training ranges (assignment table) ------------------------------------------------
SP_P_RANGE = (0.02, 0.15)
BLUR_KERNELS = (3, 5, 7)
BLUR_SIGMA_RANGE = (0.5, 2.5)
OCC_N_RECTS = (1, 3)
OCC_COVER_RANGE = (0.10, 0.35)

# ---- fixed test severities (assignment, "final testing" paragraph) ---------------------
SEVERITIES = ["low", "medium", "high"]
TEST_SP_P = {"low": 0.03, "medium": 0.08, "high": 0.15}
TEST_BLUR = {"low": (3, 0.7), "medium": (5, 1.5), "high": (7, 2.5)}  # (kernel size, sigma)
TEST_OCC = {"low": (1, 0.10), "medium": (2, 0.20), "high": (3, 0.35)}  # (n rectangles, coverage)


# ======================================================================================
# Salt-and-pepper
# ======================================================================================
def sample_salt_pepper(rng: np.random.Generator) -> dict:
    return {"p": float(rng.uniform(*SP_P_RANGE))}


def apply_salt_pepper(img: np.ndarray, p: float, seed: int) -> np.ndarray:
    """Pixel-level noise: each pixel is selected with probability p and set to black (0,0,0)
    or white (255,255,255) with equal probability (all channels together)."""
    rng = np.random.default_rng(seed)
    h, w = img.shape[:2]
    selected = rng.random((h, w)) < p
    white = rng.random((h, w)) < 0.5
    out = img.copy()
    out[selected & white] = 255
    out[selected & ~white] = 0
    return out


# ======================================================================================
# Gaussian blur
# ======================================================================================
def sample_blur(rng: np.random.Generator) -> dict:
    k = int(rng.choice(BLUR_KERNELS))
    sigma = float(rng.uniform(*BLUR_SIGMA_RANGE))
    return {"kernel": k, "sigma": sigma}


def apply_blur(img: np.ndarray, kernel: int, sigma: float) -> np.ndarray:
    """Gaussian blur with an explicit kernel window of size `kernel` (radius (kernel-1)/2)
    and standard deviation `sigma`; borders are reflected. Channels are blurred independently."""
    radius = (kernel - 1) // 2
    truncate = radius / sigma  # scipy kernel radius = int(truncate * sigma + 0.5) == radius
    f = img.astype(np.float32)
    out = gaussian_filter(f, sigma=(sigma, sigma, 0), truncate=truncate, mode="reflect")
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


# ======================================================================================
# Rectangular occlusion
# ======================================================================================
def _coverage(rects, size=IMG_SIZE) -> float:
    m = np.zeros((size, size), dtype=bool)
    for (x0, y0, x1, y1) in rects:
        m[y0:y1, x0:x1] = True
    return float(m.mean())


def _place_rects(rng: np.random.Generator, n: int, coverage: float, size=IMG_SIZE):
    """Draw n rectangles whose *areas* sum to `coverage * size^2`, with random aspect ratios and
    random positions. Rectangles are half-open pixel boxes (x0, y0, x1, y1)."""
    total = coverage * size * size
    shares = rng.dirichlet(np.full(n, 4.0)) if n > 1 else np.array([1.0])  # fairly even split
    rects = []
    for share in shares:
        area = share * total
        aspect = float(np.exp(rng.uniform(np.log(0.5), np.log(2.0))))  # w/h in [0.5, 2]
        w = int(np.clip(round(np.sqrt(area * aspect)), 4, size))
        h = int(np.clip(round(area / max(w, 1)), 4, size))
        # place; try to avoid overlap with earlier rectangles so coverage adds up
        best = None
        for _ in range(50):
            x0 = int(rng.integers(0, size - w + 1))
            y0 = int(rng.integers(0, size - h + 1))
            cand = (x0, y0, x0 + w, y0 + h)
            overlap = any(not (cand[2] <= r[0] or cand[0] >= r[2] or cand[3] <= r[1] or cand[1] >= r[3])
                          for r in rects)
            if best is None:
                best = cand
            if not overlap:
                best = cand
                break
        rects.append(best)
    return rects


def sample_occlusion(rng: np.random.Generator) -> dict:
    """Training sampler: n ~ U{1,2,3}, target coverage ~ U(0.10, 0.35). Rejection-sampled until the
    *actual union coverage* is inside [0.10, 0.35] (overlap / rounding can shift it)."""
    while True:
        n = int(rng.integers(OCC_N_RECTS[0], OCC_N_RECTS[1] + 1))
        target = float(rng.uniform(*OCC_COVER_RANGE))
        rects = _place_rects(rng, n, target)
        cov = _coverage(rects)
        if OCC_COVER_RANGE[0] <= cov <= OCC_COVER_RANGE[1]:
            return {"rects": [list(map(int, r)) for r in rects], "coverage": cov}


def sample_occlusion_fixed(rng: np.random.Generator, n: int, target: float, tol: float = 0.015) -> dict:
    """Test-manifest sampler: n rectangles covering `target` +- tol (actual union coverage)."""
    while True:
        rects = _place_rects(rng, n, target)
        cov = _coverage(rects)
        if abs(cov - target) <= tol:
            return {"rects": [list(map(int, r)) for r in rects], "coverage": cov}


def apply_occlusion(img: np.ndarray, rects) -> np.ndarray:
    out = img.copy()
    for (x0, y0, x1, y1) in rects:
        out[y0:y1, x0:x1] = 0
    return out


# ======================================================================================
# Generic dispatch used by both the runtime pipeline and the manifests
# ======================================================================================
def apply_corruption(img: np.ndarray, corruption: str, params: dict, seed: int = 0) -> np.ndarray:
    if corruption == "clean":
        return img
    if corruption == "salt_pepper":
        return apply_salt_pepper(img, params["p"], seed)
    if corruption == "blur":
        return apply_blur(img, params["kernel"], params["sigma"])
    if corruption == "occlusion":
        return apply_occlusion(img, params["rects"])
    raise ValueError(f"unknown corruption {corruption!r}")


def sample_corruption(rng: np.random.Generator, corruption: str | None = None):
    """Runtime training sampler. If `corruption` is None, pick one of the four conditions with
    equal probability (as required by the assignment). Returns (name, params, seed)."""
    if corruption is None:
        corruption = CLASSES[int(rng.integers(0, len(CLASSES)))]
    if corruption == "clean":
        params = {}
    elif corruption == "salt_pepper":
        params = sample_salt_pepper(rng)
    elif corruption == "blur":
        params = sample_blur(rng)
    elif corruption == "occlusion":
        params = sample_occlusion(rng)
    else:
        raise ValueError(corruption)
    seed = int(rng.integers(0, 2**31 - 1))  # only used by salt_pepper
    return corruption, params, seed


# ======================================================================================
# Severity label (low / medium / high) for a continuous parameter = nearest test level.
# Used for the validation manifest so val results can also be split by severity.
# ======================================================================================
def severity_of(corruption: str, params: dict) -> str:
    if corruption == "clean":
        return "none"
    if corruption == "salt_pepper":
        levels = {s: TEST_SP_P[s] for s in SEVERITIES}
        value = params["p"]
    elif corruption == "blur":
        levels = {s: TEST_BLUR[s][1] for s in SEVERITIES}
        value = params["sigma"]
    else:
        levels = {s: TEST_OCC[s][1] for s in SEVERITIES}
        value = params["coverage"]
    return min(levels, key=lambda s: abs(levels[s] - value))