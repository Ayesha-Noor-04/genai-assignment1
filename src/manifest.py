"""Fixed (deterministic) corruption manifests for validation and test.

Assignment requirement: "Generate a validation corruption manifest and a test corruption manifest
once and store the corruption type, severity, mask coordinates, blur settings, and random seed
associated with every image."

Validation manifest : one entry per validation image; condition sampled ONCE from the training
                      distribution with a fixed seed (severity = nearest test level, for reporting).
Test manifest       : per test image -> clean + 3 severities x 3 corruptions = 10 entries.
                      salt_pepper p = 0.03 / 0.08 / 0.15
                      blur (kernel, sigma) = (3, 0.7) / (5, 1.5) / (7, 2.5)
                      occlusion = 1 / 2 / 3 rectangles covering ~10% / 20% / 35%

Each entry (a plain dict, JSON-serialisable):
    {"image_idx": int,        # index into the corresponding image array
     "corruption": "clean" | "salt_pepper" | "blur" | "occlusion",
     "label": 0..3,           # index in corruptions.CLASSES
     "severity": "none" | "low" | "medium" | "high",
     "params": {...},         # p | kernel, sigma | rects, coverage
     "seed": int}             # drives salt-and-pepper noise; stored for every entry
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import corruptions as C

MANIFEST_SEED = 42


def _entry(image_idx, corruption, severity, params, seed):
    return {
        "image_idx": int(image_idx),
        "corruption": corruption,
        "label": C.CLASS_TO_ID[corruption],
        "severity": severity,
        "params": params,
        "seed": int(seed),
    }


def build_val_manifest(val_image_indices, seed: int = MANIFEST_SEED):
    """`val_image_indices`: indices (into trainval_128.npy) of the validation images."""
    rng = np.random.default_rng([seed, 1])
    entries = []
    for idx in val_image_indices:
        name, params, s = C.sample_corruption(rng)
        entries.append(_entry(idx, name, C.severity_of(name, params), params, s))
    return entries


def build_test_manifest(n_test_images: int, seed: int = MANIFEST_SEED):
    """10 entries per test image: clean + {salt_pepper, blur, occlusion} x {low, medium, high}."""
    rng = np.random.default_rng([seed, 2])
    entries = []
    for idx in range(n_test_images):
        entries.append(_entry(idx, "clean", "none", {}, rng.integers(0, 2**31 - 1)))
        for sev in C.SEVERITIES:
            entries.append(_entry(idx, "salt_pepper", sev, {"p": C.TEST_SP_P[sev]},
                                  rng.integers(0, 2**31 - 1)))
        for sev in C.SEVERITIES:
            k, sigma = C.TEST_BLUR[sev]
            entries.append(_entry(idx, "blur", sev, {"kernel": k, "sigma": sigma},
                                  rng.integers(0, 2**31 - 1)))
        for sev in C.SEVERITIES:
            n, cov = C.TEST_OCC[sev]
            params = C.sample_occlusion_fixed(rng, n, cov)
            entries.append(_entry(idx, "occlusion", sev, params, rng.integers(0, 2**31 - 1)))
    return entries


def apply_entry(clean_img: np.ndarray, entry: dict) -> np.ndarray:
    """Deterministically corrupt a clean uint8 [H,W,3] image according to a manifest entry."""
    return C.apply_corruption(clean_img, entry["corruption"], entry["params"], entry["seed"])


def save_manifest(entries, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"seed": MANIFEST_SEED, "n_entries": len(entries), "entries": entries}, f)


def load_manifest(path):
    with open(path) as f:
        return json.load(f)["entries"]