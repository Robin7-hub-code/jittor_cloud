from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Tuple

import numpy as np

from utils.noise import add_noise


@dataclass
class ModelNetDenoiseDataset:
    points: np.ndarray
    labels: np.ndarray
    noise_type: str = "gaussian"
    noise_std: float = 0.02
    noise_range: float = 0.03

    def __len__(self) -> int:
        return int(self.points.shape[0])

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray, int]:
        clean = self.points[index].astype(np.float32)
        noisy = add_noise(clean, self.noise_type, std=self.noise_std, noise_range=self.noise_range)
        label = int(self.labels[index])
        return noisy, clean, label


def load_processed_split(path: str | Path):
    data = np.load(path)
    return data["points"].astype(np.float32), data["labels"].astype(np.int64)


def split_iterator(
    dataset: ModelNetDenoiseDataset,
    batch_size: int,
    shuffle: bool = True,
    seed: int = 42,
) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(dataset))
    if shuffle:
        rng.shuffle(idx)
    for st in range(0, len(dataset), batch_size):
        batch_idx = idx[st : st + batch_size]
        noisy, clean, labels = [], [], []
        for i in batch_idx:
            n, c, y = dataset[int(i)]
            noisy.append(n)
            clean.append(c)
            labels.append(y)
        yield np.stack(noisy, axis=0), np.stack(clean, axis=0), np.array(labels, dtype=np.int64)
