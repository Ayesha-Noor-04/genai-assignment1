from __future__ import annotations

import numpy as np
from torch.utils.data import Dataset, Sampler

from src import corruptions as C
from src.datasets import to_tensor


class ClassifierDataset(Dataset):
    def __init__(self, images, indices, seed=42):
        self.images = images
        self.indices = np.asarray(indices)
        self.seed = seed

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        clean = self.images[self.indices[i]]

        rng = np.random.default_rng([self.seed, i])

        label = int(rng.integers(0, len(C.CLASSES)))
        corruption = C.CLASSES[label]

        name, params, seed = C.sample_corruption(rng, corruption)
        corrupted = C.apply_corruption(clean, name, params, seed)

        return to_tensor(corrupted), label


class BalancedBatchSampler(Sampler):
    def __init__(self, labels, batch_size, seed=42):
        if batch_size % len(C.CLASSES) != 0:
            raise ValueError("batch_size must be divisible by 4")

        self.labels = np.asarray(labels)
        self.batch_size = batch_size
        self.per_class = batch_size // len(C.CLASSES)
        self.seed = seed

        self.class_indices = []

        for label in range(len(C.CLASSES)):
            indices = np.where(self.labels == label)[0]

            if len(indices) == 0:
                raise ValueError(f"no samples for class {label}")

            self.class_indices.append(indices)

    def __iter__(self):
        rng = np.random.default_rng(self.seed)

        shuffled = []

        for indices in self.class_indices:
            values = indices.copy()
            rng.shuffle(values)
            shuffled.append(values)

        n_batches = min(
            len(values) // self.per_class
            for values in shuffled
        )

        for batch_number in range(n_batches):
            batch = []

            start = batch_number * self.per_class
            end = start + self.per_class

            for values in shuffled:
                batch.extend(values[start:end])

            rng.shuffle(batch)
            yield batch

    def __len__(self):
        return min(
            len(values) // self.per_class
            for values in self.class_indices
        )