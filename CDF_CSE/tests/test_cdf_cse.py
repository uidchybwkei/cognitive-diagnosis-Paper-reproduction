import numpy as np

from src.models.cdf_cse import grad_alpha, grad_c, objective_F


def _row_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    row_sum = q.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    return q / row_sum


def test_gradients_match_finite_differences_small_random():
    rng = np.random.default_rng(0)

    m = 5
    k = 4
    n_theory = 3
    n_experiment = 2

    c = rng.normal(loc=0.0, scale=1.0, size=(m,))
    alpha = rng.normal(loc=0.0, scale=1.0, size=(m, k))

    q_theory = _row_normalize(rng.uniform(0.1, 1.0, size=(n_theory, k)))
    q_experiment = _row_normalize(rng.uniform(0.1, 1.0, size=(n_experiment, k)))

    r_theory = rng.uniform(0.0, 1.0, size=(m, n_theory))
    r_experiment = rng.uniform(0.0, 1.0, size=(m, n_experiment))

    mask_theory = rng.uniform(0.0, 1.0, size=(m, n_theory)) > 0.3
    mask_experiment = rng.uniform(0.0, 1.0, size=(m, n_experiment)) > 0.3

    sigma_r = 1.7
    sigma_rp = 2.3
    sigma_alpha = 0.9
    sigma_c = 1.1
    mu_alpha = -0.2
    mu_c = 0.3

    g_c = grad_c(
        c=c,
        alpha=alpha,
        r_experiment=r_experiment,
        q_experiment=q_experiment,
        mask_experiment=mask_experiment,
        sigma_rp=sigma_rp,
        sigma_c=sigma_c,
        mu_c=mu_c,
    )
    g_a = grad_alpha(
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

    def F(c_vec: np.ndarray, a_mat: np.ndarray) -> float:
        return objective_F(
            c=c_vec,
            alpha=a_mat,
            r_theory=r_theory,
            r_experiment=r_experiment,
            q_theory=q_theory,
            q_experiment=q_experiment,
            mask_theory=mask_theory,
            mask_experiment=mask_experiment,
            sigma_r=sigma_r,
            sigma_rp=sigma_rp,
            sigma_alpha=sigma_alpha,
            sigma_c=sigma_c,
            mu_alpha=mu_alpha,
            mu_c=mu_c,
        )

    eps = 1e-6

    for j in range(3):
        c_pos = c.copy()
        c_neg = c.copy()
        c_pos[j] += eps
        c_neg[j] -= eps
        num = (F(c_pos, alpha) - F(c_neg, alpha)) / (2.0 * eps)
        assert np.isfinite(num)
        assert np.isclose(num, g_c[j], rtol=1e-4, atol=1e-4)

    check_indices = [(0, 0), (1, 2), (3, 1)]
    for j, kk in check_indices:
        a_pos = alpha.copy()
        a_neg = alpha.copy()
        a_pos[j, kk] += eps
        a_neg[j, kk] -= eps
        num = (F(c, a_pos) - F(c, a_neg)) / (2.0 * eps)
        assert np.isfinite(num)
        assert np.isclose(num, g_a[j, kk], rtol=1e-4, atol=1e-4)
