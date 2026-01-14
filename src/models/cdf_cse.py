from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class CDFCSEResult:
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
    c = np.asarray(c, dtype=float)
    alpha = np.asarray(alpha, dtype=float)

    eta_theory = alpha @ q_theory.T
    eta_experiment = (c[:, None] * alpha) @ q_experiment.T

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

    diff_theory = (r_theory - pred.eta_theory) * mask_theory
    diff_experiment = (r_experiment - pred.eta_experiment) * mask_experiment

    loss_theory = 0.5 * float(sigma_r) * float(np.sum(diff_theory * diff_theory))
    loss_experiment = 0.5 * float(sigma_rp) * float(
        np.sum(diff_experiment * diff_experiment)
    )

    prior_alpha = 0.5 * float(sigma_alpha) * float(
        np.sum((alpha - float(mu_alpha)) ** 2)
    )
    prior_c = 0.5 * float(sigma_c) * float(np.sum((c - float(mu_c)) ** 2))

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

    alpha_q = alpha @ q_experiment.T
    eta_experiment = c[:, None] * alpha_q
    err = (eta_experiment - r_experiment) * mask_experiment

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

    eta_theory = alpha @ q_theory.T
    err_theory = (eta_theory - r_theory) * mask_theory
    g_theory = float(sigma_r) * (err_theory @ q_theory)

    alpha_q = alpha @ q_experiment.T
    eta_experiment = c[:, None] * alpha_q
    err_experiment = (eta_experiment - r_experiment) * mask_experiment
    g_experiment = float(sigma_rp) * ((err_experiment @ q_experiment) * c[:, None])

    g_prior = float(sigma_alpha) * (alpha - float(mu_alpha))

    return g_theory + g_experiment + g_prior


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
