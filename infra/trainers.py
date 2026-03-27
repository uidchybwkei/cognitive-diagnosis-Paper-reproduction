from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from infra.artifacts import save_run_bundle
from infra.common import CDF_CSE_ROOT, NEURALCDM_ROOT, load_python_module
from src.data.dataset import CDFCSEDataset, load_dataset_from_config, make_splits
from src.utils.metrics import masked_mae, masked_rmse
from src.utils.unified_models import UnifiedArtifacts


@dataclass(frozen=True)
class TrainResult:
    metrics: Dict[str, Any]
    history: List[Dict[str, Any]]
    artifacts: UnifiedArtifacts
    meta: Dict[str, Any]


def _metrics_dict(prefix: str, y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    return {
        f"{prefix}_mae": masked_mae(y_true, y_pred, mask),
        f"{prefix}_rmse": masked_rmse(y_true, y_pred, mask),
        f"{prefix}_n": float(np.asarray(mask, dtype=bool).sum()),
    }


def _combined_predictions(ds: CDFCSEDataset, eta_theory: np.ndarray, eta_experiment: np.ndarray) -> np.ndarray:
    return np.concatenate([eta_theory, eta_experiment], axis=1)


def _split_combined_predictions(ds: CDFCSEDataset, rhat_all: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return rhat_all[:, : ds.n_theory], rhat_all[:, ds.n_theory :]


def _evaluate_predictions(
    *,
    ds: CDFCSEDataset,
    pred_theory: np.ndarray,
    pred_experiment: np.ndarray,
    splits,
    split_name: str,
    true_theory: np.ndarray | None = None,
    true_experiment: np.ndarray | None = None,
) -> Dict[str, float]:
    if split_name == "train":
        mask_theory = splits.train_theory
        mask_experiment = splits.train_experiment
    elif split_name == "val":
        mask_theory = splits.val_theory
        mask_experiment = splits.val_experiment
    elif split_name == "test":
        mask_theory = splits.test_theory
        mask_experiment = splits.test_experiment
    else:
        raise ValueError(f"Unsupported split_name={split_name}")

    obs_theory = ds.r_theory if true_theory is None else true_theory
    obs_experiment = ds.r_experiment if true_experiment is None else true_experiment
    metrics: Dict[str, float] = {}
    metrics.update(_metrics_dict(f"{split_name}_theory", obs_theory, pred_theory, mask_theory))
    metrics.update(_metrics_dict(f"{split_name}_experiment", obs_experiment, pred_experiment, mask_experiment))
    combined_true = np.concatenate([obs_theory, obs_experiment], axis=1)
    combined_pred = np.concatenate([pred_theory, pred_experiment], axis=1)
    combined_mask = np.concatenate([mask_theory, mask_experiment], axis=1)
    metrics.update(_metrics_dict(f"{split_name}_all", combined_true, combined_pred, combined_mask))
    return metrics


def train_cdf_cse(cfg: Dict[str, Any], run_dir: str | Path) -> TrainResult:
    cdf_train = load_python_module("cdf_cse_train_module", CDF_CSE_ROOT / "train.py")
    result = cdf_train.train(cfg)
    ds = load_dataset_from_config(cfg)

    artifacts = UnifiedArtifacts(
        c=np.asarray(result["c"], dtype=float),
        alpha=np.asarray(result["alpha"], dtype=float),
        rhat_all=_combined_predictions(ds, result["pred"].eta_theory, result["pred"].eta_experiment),
        eta_theory=np.asarray(result["pred"].eta_theory, dtype=float),
        eta_experiment=np.asarray(result["pred"].eta_experiment, dtype=float),
    )
    metrics = dict(result["final_metrics"])
    metrics["model"] = "cdf_cse"
    history = list(result["history"])
    meta = {"model": "cdf_cse"}
    save_run_bundle(run_dir, cfg=cfg, metrics=metrics, artifacts=artifacts, history=history, meta=meta)
    return TrainResult(metrics=metrics, history=history, artifacts=artifacts, meta=meta)


def _load_dataset_raw_q(cfg: Dict[str, Any]) -> CDFCSEDataset:
    cfg_copy = copy.deepcopy(cfg)
    cfg_copy.setdefault("data", {})["q_normalize"] = False
    return load_dataset_from_config(cfg_copy)


def train_neuralcdm(cfg: Dict[str, Any], run_dir: str | Path) -> TrainResult:
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
    except ImportError as exc:
        raise ImportError("NeuralCDM training requires torch") from exc

    neural_model = load_python_module("neuralcdm_model_module", NEURALCDM_ROOT / "model.py")
    Net = neural_model.Net

    ds = _load_dataset_raw_q(cfg)
    split_cfg = cfg.get("split", {})
    splits = make_splits(
        ds,
        train_ratio=float(split_cfg.get("train_ratio", 0.8)),
        val_ratio=float(split_cfg.get("val_ratio", 0.0)),
        seed=int(split_cfg.get("seed", 42)),
        split_mode=str(cfg.get("data", {}).get("split_mode", "combined")),
    )

    combined_r = ds.combined_r()
    combined_q = ds.combined_q()
    train_mask = np.concatenate([splits.train_theory, splits.train_experiment], axis=1)
    test_mask = np.concatenate([splits.test_theory, splits.test_experiment], axis=1)
    threshold = float(cfg.get("neuralcdm", {}).get("label_threshold", 0.5))
    labels = (combined_r >= threshold).astype(np.float32)

    train_idx = np.argwhere(train_mask)
    device_name = str(cfg.get("neuralcdm", {}).get("device", "cpu"))
    device = torch.device(device_name if device_name != "auto" else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    epoch_n = int(cfg.get("neuralcdm", {}).get("epochs", 20))
    batch_size = int(cfg.get("neuralcdm", {}).get("batch_size", 64))
    lr = float(cfg.get("neuralcdm", {}).get("lr", 0.002))
    seed = int(cfg.get("training", {}).get("seed", 42))
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    net = Net(ds.n_students, ds.n_theory + ds.n_experiment, ds.n_skills).to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    loss_function = nn.BCELoss()

    history: List[Dict[str, Any]] = []
    q_tensor = torch.tensor(combined_q, dtype=torch.float32, device=device)

    for epoch in range(1, epoch_n + 1):
        rng.shuffle(train_idx)
        net.train()
        epoch_loss = 0.0
        for start in range(0, len(train_idx), batch_size):
            batch = train_idx[start : start + batch_size]
            stu_ids = torch.tensor(batch[:, 0], dtype=torch.long, device=device)
            exer_ids = torch.tensor(batch[:, 1], dtype=torch.long, device=device)
            kn_emb = q_tensor[exer_ids]
            y = torch.tensor(labels[batch[:, 0], batch[:, 1]], dtype=torch.float32, device=device).view(-1, 1)

            optimizer.zero_grad()
            pred = net(stu_ids, exer_ids, kn_emb)
            loss = loss_function(pred, y)
            loss.backward()
            optimizer.step()
            net.apply_clipper()
            epoch_loss += float(loss.item()) * len(batch)

        net.eval()
        with torch.no_grad():
            alpha = net.get_knowledge_status(torch.arange(ds.n_students, dtype=torch.long, device=device)).cpu().numpy()
            c = np.full((ds.n_students,), np.nan, dtype=float)
            k_difficulty, e_discrimination = net.get_exer_params(torch.arange(ds.n_theory + ds.n_experiment, dtype=torch.long, device=device))
            kd = k_difficulty.cpu().numpy()
            ed = e_discrimination.cpu().numpy().reshape(-1)
            predictions = np.zeros((ds.n_students, ds.n_theory + ds.n_experiment), dtype=float)
            for item_idx in range(ds.n_theory + ds.n_experiment):
                input_x = ed[item_idx] * (alpha - kd[item_idx][None, :]) * combined_q[item_idx][None, :]
                hidden1 = 1.0 / (1.0 + np.exp(-(input_x @ net.prednet_full1.weight.detach().cpu().numpy().T + net.prednet_full1.bias.detach().cpu().numpy()[None, :])))
                hidden2 = 1.0 / (1.0 + np.exp(-(hidden1 @ net.prednet_full2.weight.detach().cpu().numpy().T + net.prednet_full2.bias.detach().cpu().numpy()[None, :])))
                output = 1.0 / (1.0 + np.exp(-(hidden2 @ net.prednet_full3.weight.detach().cpu().numpy().T + net.prednet_full3.bias.detach().cpu().numpy()[None, :])))
                predictions[:, item_idx] = output.reshape(-1)

        pred_theory, pred_experiment = _split_combined_predictions(ds, predictions)
        true_theory = (ds.r_theory >= threshold).astype(float)
        true_experiment = (ds.r_experiment >= threshold).astype(float)
        epoch_metrics = {
            "epoch": int(epoch),
            "train_loss": float(epoch_loss / max(len(train_idx), 1)),
        }
        epoch_metrics.update(
            _evaluate_predictions(
                ds=ds,
                pred_theory=pred_theory,
                pred_experiment=pred_experiment,
                splits=splits,
                split_name="test",
                true_theory=true_theory,
                true_experiment=true_experiment,
            )
        )
        history.append(epoch_metrics)

    artifacts = UnifiedArtifacts(
        c=c,
        alpha=alpha,
        rhat_all=predictions,
        eta_theory=pred_theory,
        eta_experiment=pred_experiment,
    )
    metrics = {
        "model": "neuralcdm",
        "dataset": ds.name,
        "train_ratio": float(split_cfg.get("train_ratio", 0.8)),
        "val_ratio": float(split_cfg.get("val_ratio", 0.0)),
        "split_seed": int(split_cfg.get("seed", 42)),
        "epochs": int(epoch_n),
        "label_threshold": float(threshold),
    }
    metrics.update(history[-1])

    model_path = Path(run_dir) / "model.pt"
    torch.save(net.state_dict(), model_path)
    meta = {"model": "neuralcdm", "snapshot": str(model_path)}
    save_run_bundle(run_dir, cfg=cfg, metrics=metrics, artifacts=artifacts, history=history, meta=meta)
    return TrainResult(metrics=metrics, history=history, artifacts=artifacts, meta=meta)


def _compute_subjective_n(alpha: np.ndarray, q: np.ndarray) -> np.ndarray:
    masked = alpha[:, None, :] * q[None, :, :]
    masked = np.where(q[None, :, :] > 0, masked, -np.inf)
    n = np.max(masked, axis=2)
    n[np.isneginf(n)] = 0.0
    return n


def _normal_logpdf(x: np.ndarray, mean: np.ndarray, sigma: float) -> np.ndarray:
    sigma_value = max(float(sigma), 1e-6)
    return -0.5 * np.log(2.0 * np.pi * sigma_value * sigma_value) - ((x - mean) ** 2) / (2.0 * sigma_value * sigma_value)


def train_fuzzycdf(cfg: Dict[str, Any], run_dir: str | Path) -> TrainResult:
    ds = _load_dataset_raw_q(cfg)
    split_cfg = cfg.get("split", {})
    splits = make_splits(
        ds,
        train_ratio=float(split_cfg.get("train_ratio", 0.8)),
        val_ratio=float(split_cfg.get("val_ratio", 0.0)),
        seed=int(split_cfg.get("seed", 42)),
        split_mode=str(cfg.get("data", {}).get("split_mode", "combined")),
    )

    r = ds.combined_r()
    q = ds.combined_q()
    train_mask = np.concatenate([splits.train_theory, splits.train_experiment], axis=1)
    fuzzy_cfg = cfg.get("fuzzycdf", {})
    n_iters = int(fuzzy_cfg.get("n_iters", 200))
    burnin = int(fuzzy_cfg.get("burnin", max(n_iters // 2, 1)))
    log_every = int(fuzzy_cfg.get("log_every", 20))
    seed = int(cfg.get("training", {}).get("seed", 42))
    rng = np.random.default_rng(seed)

    n_students = ds.n_students
    n_items = ds.n_theory + ds.n_experiment
    n_skills = ds.n_skills

    theta = rng.normal(0.0, 1.0, size=(n_students,))
    a = np.exp(rng.normal(0.0, 1.0, size=(n_students, n_skills)))
    b = rng.normal(0.0, 1.0, size=(n_students, n_skills))
    s = rng.beta(1.0, 2.0, size=(n_items,)) * 0.6
    g = rng.beta(1.0, 2.0, size=(n_items,)) * 0.6
    variance = float(fuzzy_cfg.get("sigma", 0.2))

    ea = np.zeros_like(a)
    eb = np.zeros_like(b)
    etheta = np.zeros_like(theta)
    ealpha = np.zeros((n_students, n_skills), dtype=float)
    es = np.zeros_like(s)
    eg = np.zeros_like(g)
    history: List[Dict[str, Any]] = []

    def compute_alpha(theta_value: np.ndarray, a_value: np.ndarray, b_value: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-1.7 * a_value * (theta_value[:, None] - b_value)))

    def compute_predictions(alpha_value: np.ndarray, s_value: np.ndarray, g_value: np.ndarray) -> np.ndarray:
        n_value = _compute_subjective_n(alpha_value, q)
        return (1.0 - s_value[None, :]) * n_value + g_value[None, :] * (1.0 - n_value)

    def train_loglik(pred_value: np.ndarray) -> float:
        logpdf = _normal_logpdf(r, pred_value, variance)
        return float(np.sum(logpdf[train_mask]))

    best_pred = None
    alpha = compute_alpha(theta, a, b)
    pred = compute_predictions(alpha, s, g)
    current_ll = train_loglik(pred)

    for it in range(1, n_iters + 1):
        cand_a = np.clip(a + rng.normal(0.0, 0.05, size=a.shape), 1e-4, None)
        cand_b = b + rng.normal(0.0, 0.05, size=b.shape)
        cand_theta = theta + rng.normal(0.0, 0.05, size=theta.shape)
        cand_s = np.clip(s + rng.normal(0.0, 0.02, size=s.shape), 0.0, 0.6)
        cand_g = np.clip(g + rng.normal(0.0, 0.02, size=g.shape), 0.0, 0.6)

        cand_alpha = compute_alpha(cand_theta, cand_a, cand_b)
        cand_pred = compute_predictions(cand_alpha, cand_s, cand_g)
        cand_ll = train_loglik(cand_pred)
        accept_prob = min(1.0, float(np.exp(np.clip(cand_ll - current_ll, -50.0, 50.0))))
        if rng.random() < accept_prob:
            a, b, theta, s, g = cand_a, cand_b, cand_theta, cand_s, cand_g
            alpha = cand_alpha
            pred = cand_pred
            current_ll = cand_ll

        if it > burnin:
            ea += a
            eb += b
            etheta += theta
            ealpha += alpha
            es += s
            eg += g
            best_pred = pred.copy()

        if it == 1 or it == n_iters or (log_every > 0 and it % log_every == 0):
            pred_theory, pred_experiment = _split_combined_predictions(ds, pred)
            record = {"iter": int(it), "train_loglik": float(current_ll)}
            record.update(
                _evaluate_predictions(
                    ds=ds,
                    pred_theory=pred_theory,
                    pred_experiment=pred_experiment,
                    splits=splits,
                    split_name="test",
                )
            )
            history.append(record)

    denom = max(n_iters - burnin, 1)
    a = ea / denom if np.any(ea) else a
    b = eb / denom if np.any(eb) else b
    theta = etheta / denom if np.any(etheta) else theta
    alpha = ealpha / denom if np.any(ealpha) else alpha
    s = es / denom if np.any(es) else s
    g = eg / denom if np.any(eg) else g
    predictions = compute_predictions(alpha, s, g) if best_pred is None else best_pred
    pred_theory, pred_experiment = _split_combined_predictions(ds, predictions)

    artifacts = UnifiedArtifacts(
        c=theta,
        alpha=alpha,
        rhat_all=predictions,
        eta_theory=pred_theory,
        eta_experiment=pred_experiment,
    )
    metrics = {
        "model": "fuzzycdf",
        "dataset": ds.name,
        "train_ratio": float(split_cfg.get("train_ratio", 0.8)),
        "val_ratio": float(split_cfg.get("val_ratio", 0.0)),
        "split_seed": int(split_cfg.get("seed", 42)),
        "n_iters": int(n_iters),
        "burnin": int(burnin),
        "question_type_strategy": "all_subjective",
    }
    metrics.update(
        _evaluate_predictions(
            ds=ds,
            pred_theory=pred_theory,
            pred_experiment=pred_experiment,
            splits=splits,
            split_name="test",
        )
    )

    fuzzy_params = {
        "theta": theta,
        "A": a,
        "B": b,
        "S": s,
        "G": g,
    }
    np.savez_compressed(Path(run_dir) / "fuzzy_params.npz", **fuzzy_params)
    meta = {"model": "fuzzycdf", "question_type_strategy": "all_subjective"}
    save_run_bundle(run_dir, cfg=cfg, metrics=metrics, artifacts=artifacts, history=history, meta=meta)
    return TrainResult(metrics=metrics, history=history, artifacts=artifacts, meta=meta)


def train_dina(cfg: Dict[str, Any], run_dir: str | Path) -> TrainResult:
    """
    DINA 需要二值作答；因此对统一输入的 `R.csv`（已归一化到 [0,1]）按 cfg.dina.label_threshold 做 0/1 化。
    """
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
    except ImportError as exc:
        raise ImportError("DINA training requires torch") from exc

    ds = _load_dataset_raw_q(cfg)  # DINA 的 Q 需要是原始 0/1 矩阵
    split_cfg = cfg.get("split", {})
    splits = make_splits(
        ds,
        train_ratio=float(split_cfg.get("train_ratio", 0.8)),
        val_ratio=float(split_cfg.get("val_ratio", 0.0)),
        seed=int(split_cfg.get("seed", 42)),
        split_mode=str(cfg.get("data", {}).get("split_mode", "combined")),
    )

    device_name = str(cfg.get("dina", {}).get("device", "auto"))
    if device_name == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    epochs = int(cfg.get("dina", {}).get("epochs", 300))
    lr = float(cfg.get("dina", {}).get("lr", 0.01))
    log_every = int(cfg.get("dina", {}).get("log_every", 50))
    label_threshold = float(cfg.get("dina", {}).get("label_threshold", 0.6))

    n_students = ds.n_students
    n_items = ds.n_theory + ds.n_experiment
    n_skills = ds.n_skills

    # Q: (n_items, n_skills)
    q_matrix = ds.combined_q().astype(np.float32)
    q_tensor = torch.tensor(q_matrix, dtype=torch.float32, device=device)

    # 二值响应标签: (n_students, n_items)
    combined_r = ds.combined_r().astype(np.float32)
    y = (combined_r >= label_threshold).astype(np.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32, device=device)

    # 训练掩码：只在 train split 上计算损失
    train_mask = np.concatenate([splits.train_theory, splits.train_experiment], axis=1)
    train_mask_tensor = torch.tensor(train_mask, dtype=torch.float32, device=device)
    n_train_obs = float(np.asarray(train_mask, dtype=bool).sum())

    class DINA(nn.Module):
        def __init__(self, n_students: int, n_items: int, n_skills: int, q: torch.Tensor):
            super().__init__()
            self.alpha = nn.Parameter(torch.randn(n_students, n_skills))
            self.slip = nn.Parameter(torch.rand(n_items))
            self.guess = nn.Parameter(torch.rand(n_items))
            self.register_buffer("Q", q)

        def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
            alpha_prob = torch.sigmoid(self.alpha)  # (N, K)
            alpha_expanded = alpha_prob.unsqueeze(1)  # (N, 1, K)
            Q_expanded = self.Q.unsqueeze(0)  # (1, M, K)

            # DINA: eta_ij = prod_k alpha_jk^{q_ik}
            eta = torch.prod(alpha_expanded ** Q_expanded, dim=2)  # (N, M)

            slip = torch.sigmoid(self.slip).unsqueeze(0)  # (1, M)
            guess = torch.sigmoid(self.guess).unsqueeze(0)  # (1, M)

            prob = eta * (1.0 - slip) + (1.0 - eta) * guess  # (N, M)
            return prob, alpha_prob

    model = DINA(n_students=n_students, n_items=n_items, n_skills=n_skills, q=q_tensor).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history: List[Dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        pred, _alpha_prob = model()
        diff2 = (pred - y_tensor) ** 2
        loss = (diff2 * train_mask_tensor).sum() / max(n_train_obs, 1.0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % log_every == 0 or epoch == 1 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                pred_all, alpha_prob = model()
                pred_theory, pred_experiment = _split_combined_predictions(ds, pred_all.detach().cpu().numpy())
                true_theory = (ds.r_theory >= label_threshold).astype(float)
                true_experiment = (ds.r_experiment >= label_threshold).astype(float)
                epoch_metrics = _evaluate_predictions(
                    ds=ds,
                    pred_theory=pred_theory,
                    pred_experiment=pred_experiment,
                    splits=splits,
                    split_name="test",
                    true_theory=true_theory,
                    true_experiment=true_experiment,
                )
                history.append(
                    {
                        "epoch": int(epoch),
                        "train_loss": float(loss.item()),
                        **epoch_metrics,
                    }
                )

    model.eval()
    with torch.no_grad():
        pred_all, alpha_prob = model()
        pred_np = pred_all.detach().cpu().numpy()
        alpha_np = alpha_prob.detach().cpu().numpy()

    pred_theory, pred_experiment = _split_combined_predictions(ds, pred_np)
    true_theory = (ds.r_theory >= label_threshold).astype(float)
    true_experiment = (ds.r_experiment >= label_threshold).astype(float)

    artifacts = UnifiedArtifacts(
        c=np.full((ds.n_students,), np.nan, dtype=float),
        alpha=alpha_np,
        rhat_all=pred_np,
        eta_theory=pred_theory,
        eta_experiment=pred_experiment,
    )

    metrics: Dict[str, Any] = {
        "model": "dina",
        "dataset": ds.name,
        "train_ratio": float(split_cfg.get("train_ratio", 0.8)),
        "val_ratio": float(split_cfg.get("val_ratio", 0.0)),
        "split_seed": int(split_cfg.get("seed", 42)),
        "epochs": int(epochs),
        "label_threshold": float(label_threshold),
    }
    metrics.update(
        _evaluate_predictions(
            ds=ds,
            pred_theory=pred_theory,
            pred_experiment=pred_experiment,
            splits=splits,
            split_name="test",
            true_theory=true_theory,
            true_experiment=true_experiment,
        )
    )

    save_run_bundle(run_dir, cfg=cfg, metrics=metrics, artifacts=artifacts, history=history, meta={"model": "dina"})
    return TrainResult(metrics=metrics, history=history, artifacts=artifacts, meta={"model": "dina"})


def train_irt(cfg: Dict[str, Any], run_dir: str | Path) -> TrainResult:
    """
    IRT 这里按统一输入的 `R.csv`（已归一化到 [0,1]）直接建模为 2PL 的期望作答概率，
    用 MSE 拟合连续分数；与当前统一评估保持一致。
    """
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
    except ImportError as exc:
        raise ImportError("IRT training requires torch") from exc

    ds = load_dataset_from_config(cfg)
    split_cfg = cfg.get("split", {})
    splits = make_splits(
        ds,
        train_ratio=float(split_cfg.get("train_ratio", 0.8)),
        val_ratio=float(split_cfg.get("val_ratio", 0.0)),
        seed=int(split_cfg.get("seed", 42)),
        split_mode=str(cfg.get("data", {}).get("split_mode", "combined")),
    )

    device_name = str(cfg.get("irt", {}).get("device", "auto"))
    if device_name == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    epochs = int(cfg.get("irt", {}).get("epochs", 300))
    lr = float(cfg.get("irt", {}).get("lr", 0.01))
    log_every = int(cfg.get("irt", {}).get("log_every", 50))

    n_students = ds.n_students
    n_items = ds.n_theory + ds.n_experiment
    n_skills = ds.n_skills

    combined_r = ds.combined_r().astype(np.float32)
    train_mask = np.concatenate([splits.train_theory, splits.train_experiment], axis=1)
    train_mask_tensor = torch.tensor(train_mask, dtype=torch.float32, device=device)
    n_train_obs = float(np.asarray(train_mask, dtype=bool).sum())
    y_tensor = torch.tensor(combined_r, dtype=torch.float32, device=device)

    class IRT2PL(nn.Module):
        def __init__(self, n_students: int, n_items: int):
            super().__init__()
            self.theta = nn.Parameter(torch.randn(n_students))
            self.a = nn.Parameter(torch.ones(n_items))
            self.b = nn.Parameter(torch.zeros(n_items))

        def forward(self) -> torch.Tensor:
            theta = self.theta.unsqueeze(1)  # (N, 1)
            a = self.a.unsqueeze(0)  # (1, M)
            b = self.b.unsqueeze(0)  # (1, M)
            logits = a * (theta - b)
            return torch.sigmoid(logits)  # (N, M)

    model = IRT2PL(n_students=n_students, n_items=n_items).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history: List[Dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        pred = model()
        diff2 = (pred - y_tensor) ** 2
        loss = (diff2 * train_mask_tensor).sum() / max(n_train_obs, 1.0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % log_every == 0 or epoch == 1 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                pred_all = model()
                pred_np = pred_all.detach().cpu().numpy()
                pred_theory, pred_experiment = _split_combined_predictions(ds, pred_np)
                epoch_metrics = _evaluate_predictions(
                    ds=ds,
                    pred_theory=pred_theory,
                    pred_experiment=pred_experiment,
                    splits=splits,
                    split_name="test",
                )
                history.append(
                    {
                        "epoch": int(epoch),
                        "train_loss": float(loss.item()),
                        **epoch_metrics,
                    }
                )

    model.eval()
    with torch.no_grad():
        pred_all = model()
        pred_np = pred_all.detach().cpu().numpy()
        theta_np = model.theta.detach().cpu().numpy()

    pred_theory, pred_experiment = _split_combined_predictions(ds, pred_np)
    # IRT 没有天然的“每技能掌握度”；这里做一个工程映射：把同一个 theta 广播到所有技能维度。
    alpha_np = (1.0 / (1.0 + np.exp(-theta_np))).reshape(-1, 1).repeat(n_skills, axis=1)

    artifacts = UnifiedArtifacts(
        c=theta_np.astype(float),
        alpha=alpha_np.astype(float),
        rhat_all=pred_np,
        eta_theory=pred_theory,
        eta_experiment=pred_experiment,
    )

    metrics: Dict[str, Any] = {
        "model": "irt",
        "dataset": ds.name,
        "train_ratio": float(split_cfg.get("train_ratio", 0.8)),
        "val_ratio": float(split_cfg.get("val_ratio", 0.0)),
        "split_seed": int(split_cfg.get("seed", 42)),
        "epochs": int(epochs),
    }
    metrics.update(
        _evaluate_predictions(
            ds=ds,
            pred_theory=pred_theory,
            pred_experiment=pred_experiment,
            splits=splits,
            split_name="test",
        )
    )

    save_run_bundle(run_dir, cfg=cfg, metrics=metrics, artifacts=artifacts, history=history, meta={"model": "irt"})
    return TrainResult(metrics=metrics, history=history, artifacts=artifacts, meta={"model": "irt"})


def train_model(model_name: str, cfg: Dict[str, Any], run_dir: str | Path) -> TrainResult:
    if model_name == "cdf_cse":
        return train_cdf_cse(cfg, run_dir)
    if model_name == "neuralcdm":
        return train_neuralcdm(cfg, run_dir)
    if model_name == "fuzzycdf":
        return train_fuzzycdf(cfg, run_dir)
    if model_name == "dina":
        return train_dina(cfg, run_dir)
    if model_name == "irt":
        return train_irt(cfg, run_dir)
    raise ValueError(f"Unsupported model={model_name}")
