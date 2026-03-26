from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class CDFCSEResult:
    """CDF-CSE 模型的预测结果
    
    Attributes:
        eta_theory: 理论题的预测分数/掌握度 η，shape (M, N_i)，对应论文 Eq.3.2-2
        eta_experiment: 实验题的预测分数/掌握度 η'，shape (M, N_e)，对应论文 Eq.3.2-3
    """
    eta_theory: np.ndarray
    eta_experiment: np.ndarray


def _check_shapes(
    c: np.ndarray,
    alpha: np.ndarray,
    q_theory: np.ndarray,
    q_experiment: np.ndarray,
    r_theory: np.ndarray,
    r_experiment: np.ndarray,
    mask_theory: np.ndarray,
    mask_experiment: np.ndarray,
) -> Tuple[int, int, int, int]:
    """检查输入数组的形状是否一致
    
    Args:
        c: 学生能力参数，shape (M,)
        alpha: 学生技能掌握度，shape (M, K)
        q_theory: 理论题 Q 矩阵，shape (N_i, K)
        q_experiment: 实验题 Q 矩阵，shape (N_e, K)
        r_theory: 理论题得分矩阵，shape (M, N_i)
        r_experiment: 实验题得分矩阵，shape (M, N_e)
        mask_theory: 理论题观测掩码，shape (M, N_i)
        mask_experiment: 实验题观测掩码，shape (M, N_e)
    
    Returns:
        (m, k, n_theory, n_experiment): 学生数、技能数、理论题数、实验题数
    """
    c = np.asarray(c)
    alpha = np.asarray(alpha)

    if c.ndim != 1:
        raise ValueError(f"c must be 1D of shape (M,), got shape={c.shape}")
    if alpha.ndim != 2:
        raise ValueError(f"alpha must be 2D of shape (M,K), got shape={alpha.shape}")

    m = int(alpha.shape[0])
    k = int(alpha.shape[1])

    if int(c.shape[0]) != m:
        raise ValueError(f"c has shape {c.shape} but alpha has shape {alpha.shape}")

    if q_theory.ndim != 2 or q_experiment.ndim != 2:
        raise ValueError("q_theory and q_experiment must be 2D")

    n_theory = int(q_theory.shape[0])
    n_experiment = int(q_experiment.shape[0])

    if int(q_theory.shape[1]) != k:
        raise ValueError(
            f"q_theory has shape {q_theory.shape} but alpha expects K={k}"
        )
    if int(q_experiment.shape[1]) != k:
        raise ValueError(
            f"q_experiment has shape {q_experiment.shape} but alpha expects K={k}"
        )

    if r_theory.shape != (m, n_theory):
        raise ValueError(
            f"r_theory must have shape {(m, n_theory)}, got {r_theory.shape}"
        )
    if r_experiment.shape != (m, n_experiment):
        raise ValueError(
            f"r_experiment must have shape {(m, n_experiment)}, got {r_experiment.shape}"
        )

    if mask_theory.shape != r_theory.shape:
        raise ValueError(
            f"mask_theory must have shape {r_theory.shape}, got {mask_theory.shape}"
        )
    if mask_experiment.shape != r_experiment.shape:
        raise ValueError(
            f"mask_experiment must have shape {r_experiment.shape}, got {mask_experiment.shape}"
        )

    return m, k, n_theory, n_experiment


def predict(
    c: np.ndarray,
    alpha: np.ndarray,
    q_theory: np.ndarray,
    q_experiment: np.ndarray,
) -> CDFCSEResult:
    """根据学生参数和 Q 矩阵预测得分
    
    Args:
        c: 学生能力参数，shape (M,)
        alpha: 学生技能掌握度，shape (M, K)
        q_theory: 理论题 Q 矩阵，shape (N_i, K)
        q_experiment: 实验题 Q 矩阵，shape (N_e, K)
    
    Returns:
        CDFCSERult: 包含预测得分的结果对象
            - eta_theory: 理论题预测得分，shape (M, N_i)
            - eta_experiment: 实验题预测得分，shape (M, N_e)
    """
    c = np.asarray(c, dtype=float)
    alpha = np.asarray(alpha, dtype=float)

    eta_theory = alpha @ q_theory.T # 对理论题目的掌握程度
    eta_experiment = (c[:, None] * alpha) @ q_experiment.T # 对实验题目的掌握程度

    return CDFCSEResult(eta_theory=eta_theory, eta_experiment=eta_experiment)


def objective_F(
    c: np.ndarray,
    alpha: np.ndarray,
    r_theory: np.ndarray,
    r_experiment: np.ndarray,
    q_theory: np.ndarray,
    q_experiment: np.ndarray,
    mask_theory: np.ndarray,
    mask_experiment: np.ndarray,
    sigma_r: float,
    sigma_rp: float,
    sigma_alpha: float,
    sigma_c: float,
    mu_alpha: float,
    mu_c: float,
) -> float:
    """计算目标函数 F(c,α)
    
    Args:
        c: 学生能力参数，shape (M,)
        alpha: 学生技能掌握度，shape (M, K)
        r_theory: 理论题得分矩阵，shape (M, N_i)
        r_experiment: 实验题得分矩阵，shape (M, N_e)
        q_theory: 理论题 Q 矩阵，shape (N_i, K)
        q_experiment: 实验题 Q 矩阵，shape (N_e, K)
        mask_theory: 理论题观测掩码，shape (M, N_i)
        mask_experiment: 实验题观测掩码，shape (M, N_e)
        sigma_r: 理论题得分噪声标准差
        sigma_rp: 实验题得分噪声标准差
        sigma_alpha: 技能掌握度先验标准差
        sigma_c: 能力参数先验标准差
        mu_alpha: 技能掌握度先验均值
        mu_c: 能力参数先验均值
    
    Returns:
        float: 目标函数值 F(c,α)
    """
    _check_shapes(
        c=c,
        alpha=alpha,
        q_theory=q_theory,
        q_experiment=q_experiment,
        r_theory=r_theory,
        r_experiment=r_experiment,
        mask_theory=mask_theory,
        mask_experiment=mask_experiment,
    )

    pred = predict(c=c, alpha=alpha, q_theory=q_theory, q_experiment=q_experiment)

    # 把缺失值mask掉, 这里是按照公式写的 R_{ji}-\eta_{ji}
    diff_theory = (r_theory - pred.eta_theory) * mask_theory
    diff_experiment = (r_experiment - pred.eta_experiment) * mask_experiment

    # 算理论题和实验题的损失, 对应论文 Eq.3.3-5 的前两项
    loss_theory = 0.5 * float(sigma_r) * float(np.sum(diff_theory * diff_theory))
    loss_experiment = 0.5 * float(sigma_rp) * float(
        np.sum(diff_experiment * diff_experiment)
    )

    # 算先验项, 对应论文 Eq.3.3-5 的后两项
    prior_alpha = 0.5 * float(sigma_alpha) * float(
        np.sum((alpha - float(mu_alpha)) ** 2)
    )
    prior_c = 0.5 * float(sigma_c) * float(np.sum((c - float(mu_c)) ** 2))

    # 汇总所有损失, 就是完整的 Eq.3.3-5
    return float(loss_theory + loss_experiment + prior_alpha + prior_c)


def grad_c(
    c: np.ndarray,
    alpha: np.ndarray,
    r_experiment: np.ndarray,
    q_experiment: np.ndarray,
    mask_experiment: np.ndarray,
    sigma_rp: float,
    sigma_c: float,
    mu_c: float,
) -> np.ndarray:
    """计算目标函数关于 c 的梯度
    
    Args:
        c: 学生能力参数，shape (M,)
        alpha: 学生技能掌握度，shape (M, K)
        r_experiment: 实验题得分矩阵，shape (M, N_e)
        q_experiment: 实验题 Q 矩阵，shape (N_e, K)
        mask_experiment: 实验题观测掩码，shape (M, N_e)
        sigma_rp: 实验题得分噪声标准差
        sigma_c: 能力参数先验标准差
        mu_c: 能力参数先验均值
    
    Returns:
        np.ndarray: 关于 c 的梯度，shape (M,)
    """
    c = np.asarray(c, dtype=float)
    alpha = np.asarray(alpha, dtype=float)

    if c.ndim != 1 or alpha.ndim != 2:
        raise ValueError("c must be (M,), alpha must be (M,K)")
    if c.shape[0] != alpha.shape[0]:
        raise ValueError("c and alpha must share the same M dimension")
    if r_experiment.shape != mask_experiment.shape:
        raise ValueError("r_experiment and mask_experiment must have the same shape")
    if r_experiment.shape[0] != c.shape[0]:
        raise ValueError("r_experiment must have shape (M,Ne)")
    if q_experiment.shape != (r_experiment.shape[1], alpha.shape[1]):
        raise ValueError("q_experiment must have shape (Ne,K)")

    # \sum_k \alpha_{jk} q'_{ek}
    alpha_q = alpha @ q_experiment.T
    # \eta'_{je} = \sum_{k=1}^{K} \beta_{jk} q'_{ke}. 实验题的预测得分
    eta_experiment = c[:, None] * alpha_q
    # \eta'_{je}-R_{je}. 实验题的预测得分减去实际得分
    err = (eta_experiment - r_experiment) * mask_experiment

    # 
    g = float(sigma_rp) * np.sum(err * alpha_q, axis=1) + float(sigma_c) * (
        c - float(mu_c)
    )

    return g


def grad_alpha(
    c: np.ndarray,
    alpha: np.ndarray,
    r_theory: np.ndarray,
    r_experiment: np.ndarray,
    q_theory: np.ndarray,
    q_experiment: np.ndarray,
    mask_theory: np.ndarray,
    mask_experiment: np.ndarray,
    sigma_r: float,
    sigma_rp: float,
    sigma_alpha: float,
    mu_alpha: float,
) -> np.ndarray:
    """计算目标函数关于 alpha 的梯度
    
    Args:
        c: 学生能力参数，shape (M,)
        alpha: 学生技能掌握度，shape (M, K)
        r_theory: 理论题得分矩阵，shape (M, N_i)
        r_experiment: 实验题得分矩阵，shape (M, N_e)
        q_theory: 理论题 Q 矩阵，shape (N_i, K)
        q_experiment: 实验题 Q 矩阵，shape (N_e, K)
        mask_theory: 理论题观测掩码，shape (M, N_i)
        mask_experiment: 实验题观测掩码，shape (M, N_e)
        sigma_r: 理论题得分噪声标准差
        sigma_rp: 实验题得分噪声标准差
        sigma_alpha: 技能掌握度先验标准差
        mu_alpha: 技能掌握度先验均值
    
    Returns:
        np.ndarray: 关于 alpha 的梯度，shape (M, K)
    """
    c = np.asarray(c, dtype=float)
    alpha = np.asarray(alpha, dtype=float)

    if c.ndim != 1 or alpha.ndim != 2:
        raise ValueError("c must be (M,), alpha must be (M,K)")
    if c.shape[0] != alpha.shape[0]:
        raise ValueError("c and alpha must share the same M dimension")

    m = int(alpha.shape[0])
    k = int(alpha.shape[1])

    if q_theory.shape[1] != k or q_experiment.shape[1] != k:
        raise ValueError("Q matrices must have K columns")
    if r_theory.shape != mask_theory.shape:
        raise ValueError("r_theory and mask_theory must have the same shape")
    if r_experiment.shape != mask_experiment.shape:
        raise ValueError("r_experiment and mask_experiment must have the same shape")
    if r_theory.shape[0] != m or r_experiment.shape[0] != m:
        raise ValueError("R matrices must have M rows")
    if r_theory.shape[1] != q_theory.shape[0]:
        raise ValueError("r_theory must have Ni columns matching q_theory rows")
    if r_experiment.shape[1] != q_experiment.shape[0]:
        raise ValueError("r_experiment must have Ne columns matching q_experiment rows")

    # 第一项理论题部分: - \sigma_R \sum_{i=1}^{N_i} q_{ki} (R_{ji} - \eta_{ji})
    eta_theory = alpha @ q_theory.T
    err_theory = (eta_theory - r_theory) * mask_theory
    g_theory = float(sigma_r) * (err_theory @ q_theory)

    # 第二项实验题部分: - \sigma_{R'} \sum_{e=1}^{N_e} c_j q'_{ke} (R'_{je} - \eta'_{je})
    alpha_q = alpha @ q_experiment.T
    eta_experiment = c[:, None] * alpha_q
    err_experiment = (eta_experiment - r_experiment) * mask_experiment
    g_experiment = float(sigma_rp) * ((err_experiment @ q_experiment) * c[:, None])

    # 第三项先验项: - \sigma_α \sum_{k=1}^{K} (α_{jk} - μ_α)^2
    g_prior = float(sigma_alpha) * (alpha - float(mu_alpha))

    return g_theory + g_experiment + g_prior

# 更新函数: c^{\text{new}} = c^{\text{old}} - r_1 \, g(c),
def update_c(
    c: np.ndarray,
    alpha: np.ndarray,
    r_experiment: np.ndarray,
    q_experiment: np.ndarray,
    mask_experiment: np.ndarray,
    sigma_rp: float,
    sigma_c: float,
    mu_c: float,
    lr: float,
) -> np.ndarray:
    """更新学生能力参数 c
    
    Args:
        c: 当前学生能力参数，shape (M,)
        alpha: 学生技能掌握度，shape (M, K)
        r_experiment: 实验题得分矩阵，shape (M, N_e)
        q_experiment: 实验题 Q 矩阵，shape (N_e, K)
        mask_experiment: 实验题观测掩码，shape (M, N_e)
        sigma_rp: 实验题得分噪声标准差
        sigma_c: 能力参数先验标准差
        mu_c: 能力参数先验均值
        lr: 学习率
    
    Returns:
        np.ndarray: 更新后的学生能力参数，shape (M,)
    """
    g = grad_c(
        c=c,
        alpha=alpha,
        r_experiment=r_experiment,
        q_experiment=q_experiment,
        mask_experiment=mask_experiment,
        sigma_rp=sigma_rp,
        sigma_c=sigma_c,
        mu_c=mu_c,
    )
    return np.asarray(c, dtype=float) - float(lr) * g

# \alpha^{\text{new}} = \alpha^{\text{old}} - r_2 \frac{\partial}{\partial \alpha} g(\alpha),
def update_alpha(
    c: np.ndarray,
    alpha: np.ndarray,
    r_theory: np.ndarray,
    r_experiment: np.ndarray,
    q_theory: np.ndarray,
    q_experiment: np.ndarray,
    mask_theory: np.ndarray,
    mask_experiment: np.ndarray,
    sigma_r: float,
    sigma_rp: float,
    sigma_alpha: float,
    mu_alpha: float,
    lr: float,
) -> np.ndarray:
    """更新学生技能掌握度 alpha
    
    Args:
        c: 学生能力参数，shape (M,)
        alpha: 当前学生技能掌握度，shape (M, K)
        r_theory: 理论题得分矩阵，shape (M, N_i)
        r_experiment: 实验题得分矩阵，shape (M, N_e)
        q_theory: 理论题 Q 矩阵，shape (N_i, K)
        q_experiment: 实验题 Q 矩阵，shape (N_e, K)
        mask_theory: 理论题观测掩码，shape (M, N_i)
        mask_experiment: 实验题观测掩码，shape (M, N_e)
        sigma_r: 理论题得分噪声标准差
        sigma_rp: 实验题得分噪声标准差
        sigma_alpha: 技能掌握度先验标准差
        mu_alpha: 技能掌握度先验均值
        lr: 学习率
    
    Returns:
        np.ndarray: 更新后的学生技能掌握度，shape (M, K)
    """
    g = grad_alpha(
        c=c,
        alpha=alpha,
        r_theory=r_theory,
        r_experiment=r_experiment,
        q_theory=q_theory,
        q_experiment=q_experiment,
        mask_theory=mask_theory,
        mask_experiment=mask_experiment,
        sigma_r=sigma_r,
        sigma_rp=sigma_rp,
        sigma_alpha=sigma_alpha,
        mu_alpha=mu_alpha,
    )
    return np.asarray(alpha, dtype=float) - float(lr) * g
