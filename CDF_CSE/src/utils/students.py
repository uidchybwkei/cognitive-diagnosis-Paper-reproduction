from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Optional

import numpy as np


def _format_float(x: float) -> str:
    value = float(x)
    if np.isnan(value):
        return "nan"
    return f"{value:.6f}"


def write_students_csv(path: str | Path, c: np.ndarray, alpha: np.ndarray) -> None:
    out_path = Path(path)
    c_arr = np.asarray(c, dtype=float).reshape(-1)
    alpha_arr = np.asarray(alpha, dtype=float)

    if alpha_arr.ndim != 2:
        raise ValueError(f"alpha must be 2D of shape (N,K), got shape={alpha_arr.shape}")
    if int(alpha_arr.shape[0]) != int(c_arr.shape[0]):
        raise ValueError(f"c has shape {c_arr.shape} but alpha has shape {alpha_arr.shape}")

    n_students = int(alpha_arr.shape[0])
    n_skills = int(alpha_arr.shape[1])

    header = ["student_id", "global_trait"]
    header.extend([f"alpha_k{i}" for i in range(n_skills)])
    header.extend(["alpha_mean", "alpha_sum", "alpha_topk"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for student_id in range(n_students):
            alpha_row = alpha_arr[student_id]
            alpha_mean = float(np.mean(alpha_row))
            alpha_sum = float(np.sum(alpha_row))
            top_n = min(3, n_skills)
            topk_idx = np.argsort(alpha_row)[::-1][:top_n]
            alpha_topk = "|".join([f"k{int(idx)}" for idx in topk_idx])

            row = [str(student_id), _format_float(c_arr[student_id])]
            row.extend([_format_float(x) for x in alpha_row.tolist()])
            row.extend([
                _format_float(alpha_mean),
                _format_float(alpha_sum),
                alpha_topk,
            ])
            writer.writerow(row)


def save_predictions_npz(
    path: str | Path,
    *,
    eta_theory: Optional[np.ndarray] = None,
    eta_experiment: Optional[np.ndarray] = None,
    c: Optional[np.ndarray] = None,
    alpha: Optional[np.ndarray] = None,
    extra_arrays: Optional[Dict[str, np.ndarray]] = None,
) -> None:
    out_path = Path(path)
    arrays: Dict[str, np.ndarray] = {}

    if eta_theory is not None:
        arrays["eta_theory"] = np.asarray(eta_theory, dtype=float)
    if eta_experiment is not None:
        arrays["eta_experiment"] = np.asarray(eta_experiment, dtype=float)
    if eta_theory is not None and eta_experiment is not None:
        arrays["rhat_all"] = np.concatenate(
            [np.asarray(eta_theory, dtype=float), np.asarray(eta_experiment, dtype=float)],
            axis=1,
        )
    if c is not None:
        arrays["c"] = np.asarray(c, dtype=float)
    if alpha is not None:
        arrays["alpha"] = np.asarray(alpha, dtype=float)
    if extra_arrays is not None:
        for key, value in extra_arrays.items():
            arrays[str(key)] = np.asarray(value)

    if not arrays:
        raise ValueError("At least one array must be provided to save_predictions_npz")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **arrays)
