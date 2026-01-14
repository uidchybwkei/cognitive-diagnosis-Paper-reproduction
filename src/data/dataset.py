from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CDFCSEDataset:
    name: str
    r_theory: np.ndarray
    r_experiment: np.ndarray
    q_theory: np.ndarray
    q_experiment: np.ndarray
    mask_theory: np.ndarray
    mask_experiment: np.ndarray
    problem_ids_theory: np.ndarray
    problem_ids_experiment: np.ndarray
    skill_ids: np.ndarray

    @property
    def n_students(self) -> int:
        return int(self.r_theory.shape[0])

    @property
    def n_theory(self) -> int:
        return int(self.r_theory.shape[1])

    @property
    def n_experiment(self) -> int:
        return int(self.r_experiment.shape[1])

    @property
    def n_skills(self) -> int:
        return int(self.q_theory.shape[1])

    def combined_r(self) -> np.ndarray:
        return np.concatenate([self.r_theory, self.r_experiment], axis=1)

    def combined_mask(self) -> np.ndarray:
        return np.concatenate([self.mask_theory, self.mask_experiment], axis=1)

    def combined_q(self) -> np.ndarray:
        return np.concatenate([self.q_theory, self.q_experiment], axis=0)


@dataclass(frozen=True)
class SplitMasks:
    train_theory: np.ndarray
    train_experiment: np.ndarray
    val_theory: np.ndarray
    val_experiment: np.ndarray
    test_theory: np.ndarray
    test_experiment: np.ndarray


def _normalize_q_rows(q: np.ndarray) -> np.ndarray:
    row_sum = q.sum(axis=1, keepdims=True)
    out = np.zeros_like(q, dtype=float)
    nonzero = row_sum.squeeze(axis=1) > 0
    out[nonzero] = q[nonzero] / row_sum[nonzero]
    return out


def _ensure_expected_k(
    q: pd.DataFrame,
    expected_k: int,
    q_zero_row_strategy: Literal["unknown_skill", "error"],
) -> pd.DataFrame:
    if q.shape[1] > expected_k:
        raise ValueError(
            f"Q has {q.shape[1]} skill columns but expected_k={expected_k}."
        )

    if q.shape[1] < expected_k:
        for col_idx in range(q.shape[1], expected_k):
            q[str(col_idx)] = 0

    if q_zero_row_strategy not in ("unknown_skill", "error"):
        raise ValueError(f"Unsupported q_zero_row_strategy={q_zero_row_strategy}")

    row_sum = q.sum(axis=1)
    zero_rows = row_sum == 0
    if int(zero_rows.sum()) == 0:
        return q

    if q_zero_row_strategy == "error":
        raise ValueError(
            f"Q contains {int(zero_rows.sum())} all-zero rows: {list(q.index[zero_rows])}"
        )

    unknown_col = str(expected_k - 1)
    q.loc[zero_rows, unknown_col] = 1
    return q


def load_real_dataset(
    dataset_root: str | Path,
    dataset_name: str,
    n_theory: int,
    n_experiment: int,
    expected_k: int,
    score_scale: float = 10.0,
    missing_value: float = -1.0,
    q_normalize: bool = True,
    q_zero_row_strategy: Literal["unknown_skill", "error"] = "unknown_skill",
) -> CDFCSEDataset:
    ds_dir = Path(dataset_root) / dataset_name
    r_path = ds_dir / "R.csv"
    q_path = ds_dir / "q.csv"

    r_df = pd.read_csv(r_path, index_col=0)
    q_df = pd.read_csv(q_path, index_col=0)

    n_total = n_theory + n_experiment
    if r_df.shape[1] != n_total:
        raise ValueError(
            f"R has {r_df.shape[1]} problems but expected {n_total} (=n_theory+n_experiment)."
        )
    if q_df.shape[0] != n_total:
        raise ValueError(
            f"Q has {q_df.shape[0]} problem rows but expected {n_total} (=n_theory+n_experiment)."
        )

    q_df = _ensure_expected_k(q_df, expected_k=expected_k, q_zero_row_strategy=q_zero_row_strategy)

    problem_ids = np.asarray(q_df.index)
    skill_ids = np.asarray(q_df.columns)

    r = r_df.to_numpy(dtype=float)
    mask = r != float(missing_value)
    r_norm = np.zeros_like(r, dtype=float)
    r_norm[mask] = r[mask] / float(score_scale)

    q = q_df.to_numpy(dtype=float)
    if q_normalize:
        q = _normalize_q_rows(q)

    r_theory = r_norm[:, :n_theory]
    r_experiment = r_norm[:, n_theory:n_total]
    mask_theory = mask[:, :n_theory]
    mask_experiment = mask[:, n_theory:n_total]

    q_theory = q[:n_theory, :]
    q_experiment = q[n_theory:n_total, :]

    problem_ids_theory = problem_ids[:n_theory]
    problem_ids_experiment = problem_ids[n_theory:n_total]

    return CDFCSEDataset(
        name=dataset_name,
        r_theory=r_theory,
        r_experiment=r_experiment,
        q_theory=q_theory,
        q_experiment=q_experiment,
        mask_theory=mask_theory,
        mask_experiment=mask_experiment,
        problem_ids_theory=problem_ids_theory,
        problem_ids_experiment=problem_ids_experiment,
        skill_ids=skill_ids,
    )


def _split_observed_entries(
    observed_mask: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if train_ratio <= 0 or train_ratio >= 1:
        raise ValueError("train_ratio must be in (0,1)")
    if val_ratio < 0 or val_ratio >= 1:
        raise ValueError("val_ratio must be in [0,1)")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be < 1")

    obs_idx = np.argwhere(observed_mask)
    n_obs = int(obs_idx.shape[0])
    rng = np.random.default_rng(int(seed))
    rng.shuffle(obs_idx)

    n_train = int(np.floor(train_ratio * n_obs))
    n_val = int(np.floor(val_ratio * n_obs))

    train_idx = obs_idx[:n_train]
    val_idx = obs_idx[n_train : n_train + n_val]
    test_idx = obs_idx[n_train + n_val :]

    train_mask = np.zeros_like(observed_mask, dtype=bool)
    val_mask = np.zeros_like(observed_mask, dtype=bool)
    test_mask = np.zeros_like(observed_mask, dtype=bool)

    train_mask[train_idx[:, 0], train_idx[:, 1]] = True
    val_mask[val_idx[:, 0], val_idx[:, 1]] = True
    test_mask[test_idx[:, 0], test_idx[:, 1]] = True

    return train_mask, val_mask, test_mask


def make_splits(
    dataset: CDFCSEDataset,
    train_ratio: float,
    val_ratio: float,
    seed: int,
    split_mode: Literal["combined", "per_matrix"] = "combined",
) -> SplitMasks:
    if split_mode not in ("combined", "per_matrix"):
        raise ValueError(f"Unsupported split_mode={split_mode}")

    if split_mode == "combined":
        combined_obs = dataset.combined_mask()
        train_all, val_all, test_all = _split_observed_entries(
            combined_obs, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed
        )
        train_theory = train_all[:, : dataset.n_theory]
        train_experiment = train_all[:, dataset.n_theory :]
        val_theory = val_all[:, : dataset.n_theory]
        val_experiment = val_all[:, dataset.n_theory :]
        test_theory = test_all[:, : dataset.n_theory]
        test_experiment = test_all[:, dataset.n_theory :]
        return SplitMasks(
            train_theory=train_theory,
            train_experiment=train_experiment,
            val_theory=val_theory,
            val_experiment=val_experiment,
            test_theory=test_theory,
            test_experiment=test_experiment,
        )

    train_theory, val_theory, test_theory = _split_observed_entries(
        dataset.mask_theory, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed
    )
    train_experiment, val_experiment, test_experiment = _split_observed_entries(
        dataset.mask_experiment,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed + 1,
    )
    return SplitMasks(
        train_theory=train_theory,
        train_experiment=train_experiment,
        val_theory=val_theory,
        val_experiment=val_experiment,
        test_theory=test_theory,
        test_experiment=test_experiment,
    )


def load_dataset_from_config(cfg: Dict[str, Any]) -> CDFCSEDataset:
    data_cfg = cfg.get("data", {})
    name = str(data_cfg.get("dataset_name"))
    specs = data_cfg.get("dataset_specs", {})
    if name not in specs:
        raise ValueError(f"dataset_specs missing entry for dataset_name={name}")

    spec = specs[name]
    return load_real_dataset(
        dataset_root=data_cfg.get("dataset_root", "dataset"),
        dataset_name=name,
        n_theory=int(spec["n_theory"]),
        n_experiment=int(spec["n_experiment"]),
        expected_k=int(spec.get("expected_k", spec.get("n_skills"))),
        score_scale=float(data_cfg.get("score_scale", 10.0)),
        missing_value=float(data_cfg.get("missing_value", -1.0)),
        q_normalize=bool(data_cfg.get("q_normalize", True)),
        q_zero_row_strategy=str(data_cfg.get("q_zero_row_strategy", "unknown_skill")),
    )
