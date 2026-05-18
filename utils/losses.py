from __future__ import annotations

import numpy as np


def chamfer_distance(pred, target):
    pred_expand = pred.unsqueeze(2)
    target_expand = target.unsqueeze(1)
    dist = ((pred_expand - target_expand) ** 2).sum(dim=-1)
    pred_to_target = dist.min(dim=2)
    target_to_pred = dist.min(dim=1)
    if isinstance(pred_to_target, tuple):
        pred_to_target = pred_to_target[0]
    if isinstance(target_to_pred, tuple):
        target_to_pred = target_to_pred[0]
    return pred_to_target.mean() + target_to_pred.mean()


def emd_approx_numpy(pred: np.ndarray, target: np.ndarray) -> float:
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        return float("nan")
    cost = ((pred[:, None, :] - target[None, :, :]) ** 2).sum(axis=-1)
    ridx, cidx = linear_sum_assignment(cost)
    return float(cost[ridx, cidx].mean())
