from __future__ import annotations

import numpy as np


def masked_mae(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.asarray(mask, dtype=bool)

    if y_true.shape != y_pred.shape or y_true.shape != mask.shape:
        raise ValueError("y_true, y_pred, mask must have the same shape")

    if int(mask.sum()) == 0:
        return float("nan")

    err = np.abs(y_true[mask] - y_pred[mask])
    return float(np.mean(err))


def masked_rmse(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.asarray(mask, dtype=bool)

    if y_true.shape != y_pred.shape or y_true.shape != mask.shape:
        raise ValueError("y_true, y_pred, mask must have the same shape")

    if int(mask.sum()) == 0:
        return float("nan")

    err2 = (y_true[mask] - y_pred[mask]) ** 2
    return float(np.sqrt(np.mean(err2)))
