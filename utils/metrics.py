from __future__ import annotations

import numpy as np

from .losses import emd_approx_numpy


def chamfer_distance_numpy(pred: np.ndarray, target: np.ndarray) -> float:
    dist = ((pred[:, None, :] - target[None, :, :]) ** 2).sum(axis=-1)
    d1 = dist.min(axis=1).mean()
    d2 = dist.min(axis=0).mean()
    return float(d1 + d2)


def evaluate_pair(pred: np.ndarray, target: np.ndarray, with_emd: bool = False):
    result = {"cd": chamfer_distance_numpy(pred, target)}
    if with_emd:
        result["emd"] = emd_approx_numpy(pred, target)
    return result
