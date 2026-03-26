from __future__ import annotations

import numpy as np

# 要mask掉缺失值, 不能把缺失的也计算在内
def masked_mae(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> float:
    """
    计算带掩码的平均绝对误差 (Mean Absolute Error, MAE)
    
    参数:
        y_true: 真实值数组，形状为 (n_samples,) 或 (n_samples, n_features)
        y_pred: 预测值数组，形状必须与 y_true 相同
        mask: 布尔掩码数组，True 表示该位置有效，False 表示该位置需要被忽略
              形状必须与 y_true 和 y_pred 相同
    
    公式: MAE = mean(|y_true[mask] - y_pred[mask]|)
    """
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
    """
    计算带掩码的均方根误差 (Root Mean Square Error, RMSE)
    
    参数:
        y_true: 真实值数组，形状为 (n_samples,) 或 (n_samples, n_features)
        y_pred: 预测值数组，形状必须与 y_true 相同
        mask: 布尔掩码数组，True 表示该位置有效，False 表示该位置需要被忽略
              形状必须与 y_true 和 y_pred 相同
    
    公式: RMSE = sqrt(mean((y_true[mask] - y_pred[mask])^2))
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.asarray(mask, dtype=bool)

    if y_true.shape != y_pred.shape or y_true.shape != mask.shape:
        raise ValueError("y_true, y_pred, mask must have the same shape")

    if int(mask.sum()) == 0:
        return float("nan")

    err2 = (y_true[mask] - y_pred[mask]) ** 2
    return float(np.sqrt(np.mean(err2)))
