from __future__ import annotations

import numpy as np


def add_noise(
    points: np.ndarray,
    noise_type: str = "gaussian",
    std: float = 0.02,
    noise_range: float = 0.03,
) -> np.ndarray:
    if noise_type == "gaussian":
        noise = np.random.normal(0.0, std, size=points.shape).astype(np.float32)
    elif noise_type == "uniform":
        noise = np.random.uniform(-noise_range, noise_range, size=points.shape).astype(np.float32)
    elif noise_type == "mixed":
        use_gaussian = np.random.rand() > 0.5
        return add_noise(points, "gaussian" if use_gaussian else "uniform", std=std, noise_range=noise_range)
    else:
        raise ValueError(f"Unsupported noise_type={noise_type}, expected gaussian|uniform|mixed")
    return (points + noise).astype(np.float32)
