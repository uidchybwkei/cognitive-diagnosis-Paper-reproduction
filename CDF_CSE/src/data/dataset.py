from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CDFCSEDataset:
    """
    Attributes:
        name: 数据集名（如 `dataStructure` / `networkSecurity`）
        r_theory: 理论题得分矩阵，shape (M, N_i)，已归一化到 [0,1]，缺失位置填 0，但会被 mask 排除
        r_experiment: 实验题得分矩阵，shape (M, N_e)，已归一化到 [0,1]，缺失位置填 0，但会被 mask 排除
        q_theory: 理论题 Q 矩阵，shape (N_i, K)，默认按行归一化，每行和为 1
        q_experiment: 实验题 Q′ 矩阵，shape (N_e, K)，默认按行归一化，每行和为 1
        mask_theory: 理论题观测掩码，dtype bool，shape (M, N_i)，True 表示该学生-该理论题的分数是"观测到的"（不是 -1）
        mask_experiment: 实验题观测掩码，dtype bool，shape (M, N_e)，True 表示该学生-该实验题的分数是"观测到的"（不是 -1）
        problem_ids_theory: 理论题 ID，shape (N_i,)，来自 q.csv 的索引
        problem_ids_experiment: 实验题 ID，shape (N_e,)，来自 q.csv 的索引
        skill_ids: 技能 ID，shape (K,)，来自 q.csv 的列名
    """
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

    # 算数量的
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

    # 把理论题与实验题当成“同一种题” 论文的4.3节出现的情况
    def combined_r(self) -> np.ndarray:
        return np.concatenate([self.r_theory, self.r_experiment], axis=1)

    def combined_mask(self) -> np.ndarray:
        return np.concatenate([self.mask_theory, self.mask_experiment], axis=1)

    def combined_q(self) -> np.ndarray:
        return np.concatenate([self.q_theory, self.q_experiment], axis=0)


@dataclass(frozen=True)
class SplitMasks:
    """
    分割掩码, 用于将数据集分成训练集、验证集和测试集
    Attributes:
        train_theory: 训练集理论题掩码，shape (M, N_i)
        train_experiment: 训练集实验题掩码，shape (M, N_e)
        val_theory: 验证集理论题掩码，shape (M, N_i)
        val_experiment: 验证集实验题掩码，shape (M, N_e)
        test_theory: 测试集理论题掩码，shape (M, N_i)
        test_experiment: 测试集实验题掩码，shape (M, N_e)
    """
    train_theory: np.ndarray
    train_experiment: np.ndarray
    val_theory: np.ndarray
    val_experiment: np.ndarray
    test_theory: np.ndarray
    test_experiment: np.ndarray

# Q 矩阵的行归一化
def _normalize_q_rows(q: np.ndarray) -> np.ndarray:
    row_sum = q.sum(axis=1, keepdims=True)
    out = np.zeros_like(q, dtype=float)
    nonzero = row_sum.squeeze(axis=1) > 0 # 移除列维度后判断行的sum是否大于0
    out[nonzero] = q[nonzero] / row_sum[nonzero]
    return out

# 处理技能列数不一致与 Q 全零行
def _ensure_expected_k(
    q: pd.DataFrame,
    expected_k: int,
    q_zero_row_strategy: Literal["unknown_skill", "error"],
) -> pd.DataFrame:
    """确保 Q 矩阵的技能列数符合期望，并处理全零行
    
    Args:
        q: Q 矩阵，shape (N, K_raw)
        expected_k: 期望技能数，来自 configs/default.yaml 的 expected_k
        q_zero_row_strategy: 处理全零行的策略，"unknown_skill" 或 "error"
    
    Returns:
        q: 处理后的 Q 矩阵，shape (N, expected_k)
    """
    if q.shape[1] > expected_k: # 这表示数据文件包含“比配置更多的技能列”，属于严重不一致
        raise ValueError(
            f"Q has {q.shape[1]} skill columns but expected_k={expected_k}."
        )

    if q.shape[1] < expected_k: # 补全0列知道满足
        for col_idx in range(q.shape[1], expected_k):
            q[str(col_idx)] = 0

    if q_zero_row_strategy not in ("unknown_skill", "error"):
        raise ValueError(f"Unsupported q_zero_row_strategy={q_zero_row_strategy}")

    # 处理全零行, 没有任何的技能不对, 如果策略是 `unknown_skill` 时：把这些题目的“最后一列技能”置为 1
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
    """加载真实数据集并构建 CDFCSEDataset 对象
    
    Args:
        dataset_root: 数据集根目录路径（默认 dataset）
        dataset_name: 数据集名称（如 dataStructure）
        n_theory: 理论题数量 N_i，来自配置 dataset_specs，对应论文 Table 1
        n_experiment: 实验题数量 N_e，来自配置 dataset_specs，对应论文 Table 1
        expected_k: 期望技能数 K，来自配置 dataset_specs，对应论文 Table 1
        score_scale: 分数缩放因子（默认 10.0）
        missing_value: 缺失值标记（默认 -1.0）
        q_normalize: 是否对 Q 矩阵进行行归一化（默认 True，对应 Eq.3.1-1）
        q_zero_row_strategy: Q 矩阵全零行处理策略（默认 unknown_skill）
    
    Returns:
        CDFCSEDataset: 包含切分后的理论题和实验题数据的对象，其中 r_* 与 q_* 都已按理论/实验切分
    """
    ds_dir = Path(dataset_root) / dataset_name
    r_path = ds_dir / "R.csv"
    q_path = ds_dir / "q.csv"

    r_df = pd.read_csv(r_path, index_col=0)
    q_df = pd.read_csv(q_path, index_col=0)

    # 校验题目数量的一致性
    n_total = n_theory + n_experiment
    if r_df.shape[1] != n_total:
        raise ValueError(
            f"R has {r_df.shape[1]} problems but expected {n_total} (=n_theory+n_experiment)."
        )
    if q_df.shape[0] != n_total:
        raise ValueError(
            f"Q has {q_df.shape[0]} problem rows but expected {n_total} (=n_theory+n_experiment)."
        )

    # 处理技能列数的全0行
    q_df = _ensure_expected_k(q_df, expected_k=expected_k, q_zero_row_strategy=q_zero_row_strategy)

    problem_ids = np.asarray(q_df.index)
    skill_ids = np.asarray(q_df.columns)

    r = r_df.to_numpy(dtype=float)
    mask = r != float(missing_value) # 缺失值mask
    r_norm = np.zeros_like(r, dtype=float)
    r_norm[mask] = r[mask] / float(score_scale) # 归一化到 0-1

    q = q_df.to_numpy(dtype=float)
    if q_normalize:
        q = _normalize_q_rows(q) # q 的行归一化

    # 按理论/实验切分
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
    """将观测到的条目分割为训练、验证和测试集
    
    Args:
        observed_mask: 观测掩码，shape (M, N)，True 表示有值
        train_ratio: 训练集占比（必须在 (0,1)）
        val_ratio: 验证集占比（必须在 [0,1)）
        seed: 随机种子
    
    Returns:
        train_mask, val_mask, test_mask: 分割后的掩码，shape 均为 (M, N)
    """
    if train_ratio <= 0 or train_ratio >= 1:
        raise ValueError("train_ratio must be in (0,1)")
    if val_ratio < 0 or val_ratio >= 1:
        raise ValueError("val_ratio must be in [0,1)")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be < 1")

    obs_idx = np.argwhere(observed_mask) # 找true的index
    n_obs = int(obs_idx.shape[0]) 
    rng = np.random.default_rng(int(seed))
    # 打乱, 划分, 做train/val mask
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

# 划分数据集, combine是理论实验一起划分, 共享随机种子; per_matrix是分别划分, 独立随机种子
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

# load config
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
