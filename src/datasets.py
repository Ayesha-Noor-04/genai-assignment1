"""PyTorch datasets for Tasks 1-3.

  RuntimeCorruptionDataset : training. Every __getitem__ call draws a NEW corruption type and
                             severity (nothing corrupted is ever saved to disk).
  ManifestDataset          : validation / test. Deterministic, driven by a stored manifest.

Both return (corrupted, clean, label):
    corrupted : float32 tensor [3,128,128] in [0,1]
    clean     : float32 tensor [3,128,128] in [0,1]   (the reconstruction target)
    label     : int64 scalar, index in corruptions.CLASSES (used by the Task 2/3 classifier)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from . import corruptions as C
from .manifest import apply_entry, load_manifest


def load_arrays(processed_dir):
    """Returns (trainval uint8 [N,128,128,3], test uint8 [M,128,128,3], split dict)."""
    d = Path(processed_dir)
    trainval = np.load(d / "trainval_128.npy")
    test = np.load(d / "test_128.npy")
    with open(d / "split.json") as f:
        split = json.load(f)
    return trainval, test, split


def to_tensor(img_uint8: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(img_uint8).permute(2, 0, 1).float().div_(255.0)


class RuntimeCorruptionDataset(Dataset):
    """Training dataset with on-the-fly corruption.

    images : uint8 array [N,128,128,3] (the full trainval array)
    indices: which rows of `images` belong to this split (the 80% training indices)
    corruption: None -> each load picks clean / salt_pepper / blur / occlusion with equal
                probability (Task 1). Pass a name to force one type (e.g. to train a Task 2
                specialist, or to build class-balanced batches).
    """

    def __init__(self, images, indices, corruption: str | None = None, seed: int = 42):
        self.images = images
        self.indices = np.asarray(indices)
        self.corruption = corruption
        self.seed = seed
        self._rng = None
        self._rng_pid = None

    def __len__(self):
        return len(self.indices)

    def _get_rng(self):
        # One generator per worker process. torch.initial_seed() differs per worker (and per epoch
        # unless persistent_workers=True, where the generator simply keeps advancing), so every
        # load of an image gets a fresh corruption.
        import os
        pid = os.getpid()
        if self._rng is None or self._rng_pid != pid:
            self._rng = np.random.default_rng([self.seed, torch.initial_seed() % (2**32)])
            self._rng_pid = pid
        return self._rng

    def __getitem__(self, i):
        clean = self.images[self.indices[i]]
        name, params, seed = C.sample_corruption(self._get_rng(), self.corruption)
        corrupted = C.apply_corruption(clean, name, params, seed)
        return to_tensor(corrupted), to_tensor(clean), C.CLASS_TO_ID[name]


class ManifestDataset(Dataset):
    """Deterministic dataset defined by a manifest (validation or test).

    images  : uint8 array the manifest's `image_idx` values index into
              (trainval array for the validation manifest, test array for the test manifest)
    manifest: list of entries (see manifest.py) or a path to a manifest JSON
    """

    def __init__(self, images, manifest):
        self.images = images
        self.entries = load_manifest(manifest) if isinstance(manifest, (str, Path)) else manifest

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, i):
        e = self.entries[i]
        clean = self.images[e["image_idx"]]
        return to_tensor(apply_entry(clean, e)), to_tensor(clean), e["label"]

    def entry(self, i):
        return self.entries[i]