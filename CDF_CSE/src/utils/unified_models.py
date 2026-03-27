from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from src.data.dataset import CDFCSEDataset, load_dataset_from_config
from src.models.cdf_cse import predict


@dataclass(frozen=True)
class UnifiedArtifacts:
    c: np.ndarray
    alpha: np.ndarray
    rhat_all: Optional[np.ndarray]
    eta_theory: Optional[np.ndarray]
    eta_experiment: Optional[np.ndarray]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float)
    x_arr = np.clip(x_arr, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x_arr))


def _split_predictions(ds: CDFCSEDataset, rhat_all: Optional[np.ndarray]) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if rhat_all is None:
        return None, None
    pred_arr = np.asarray(rhat_all, dtype=float)
    if pred_arr.ndim != 2:
        raise ValueError(f"rhat_all must be 2D, got shape={pred_arr.shape}")
    expected_shape = (ds.n_students, ds.n_theory + ds.n_experiment)
    if pred_arr.shape != expected_shape:
        raise ValueError(f"rhat_all must have shape {expected_shape}, got {pred_arr.shape}")
    eta_theory = pred_arr[:, : ds.n_theory]
    eta_experiment = pred_arr[:, ds.n_theory :]
    return eta_theory, eta_experiment


def load_cdf_cse_artifacts(cfg: Dict[str, Any], params_path: str | Path) -> UnifiedArtifacts:
    ds = load_dataset_from_config(cfg)
    params = np.load(Path(params_path))

    c = np.asarray(params["c"], dtype=float).reshape(-1)
    alpha = np.asarray(params["alpha"], dtype=float)

    pred = predict(c=c, alpha=alpha, q_theory=ds.q_theory, q_experiment=ds.q_experiment)
    rhat_all = np.concatenate([pred.eta_theory, pred.eta_experiment], axis=1)
    return UnifiedArtifacts(
        c=c,
        alpha=alpha,
        rhat_all=rhat_all,
        eta_theory=pred.eta_theory,
        eta_experiment=pred.eta_experiment,
    )


def load_fuzzycdf_artifacts(
    cfg: Optional[Dict[str, Any]],
    *,
    fuzzy_dir: Optional[str | Path] = None,
    alpha_path: Optional[str | Path] = None,
    theta_path: Optional[str | Path] = None,
    predictions_path: Optional[str | Path] = None,
) -> UnifiedArtifacts:
    if fuzzy_dir is not None:
        root = Path(fuzzy_dir)
        alpha_path = root / "FuzzyAlpha.txt" if alpha_path is None else Path(alpha_path)
        theta_path = root / "FuzzyTheta.txt" if theta_path is None else Path(theta_path)
        predictions_path = root / "FuzzyX.txt" if predictions_path is None else Path(predictions_path)
    if alpha_path is None:
        raise ValueError("alpha_path is required for model=fuzzycdf")

    alpha = np.asarray(np.loadtxt(Path(alpha_path)), dtype=float)
    if alpha.ndim == 1:
        alpha = alpha.reshape(-1, 1)

    if theta_path is not None and Path(theta_path).exists():
        c = np.asarray(np.loadtxt(Path(theta_path)), dtype=float).reshape(-1)
    else:
        c = np.full((alpha.shape[0],), np.nan, dtype=float)

    rhat_all: Optional[np.ndarray] = None
    if predictions_path is not None and Path(predictions_path).exists():
        rhat_all = np.asarray(np.loadtxt(Path(predictions_path)), dtype=float)

    eta_theory: Optional[np.ndarray] = None
    eta_experiment: Optional[np.ndarray] = None
    if cfg is not None and rhat_all is not None:
        ds = load_dataset_from_config(cfg)
        eta_theory, eta_experiment = _split_predictions(ds, rhat_all)

    return UnifiedArtifacts(
        c=c,
        alpha=alpha,
        rhat_all=rhat_all,
        eta_theory=eta_theory,
        eta_experiment=eta_experiment,
    )


def _load_dataset_with_raw_q(cfg: Dict[str, Any]) -> CDFCSEDataset:
    cfg_copy = copy.deepcopy(cfg)
    cfg_copy.setdefault("data", {})["q_normalize"] = False
    return load_dataset_from_config(cfg_copy)


def _to_numpy_array(value: Any) -> np.ndarray:
    try:
        import torch
    except ImportError as exc:
        raise ImportError("torch is required to load NeuralCDM snapshots") from exc

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _predict_neuralcdm_matrix(alpha: np.ndarray, q_matrix: np.ndarray, state_dict: Dict[str, Any]) -> np.ndarray:
    k_difficulty = _sigmoid(_to_numpy_array(state_dict["k_difficulty.weight"]))
    e_discrimination = _sigmoid(_to_numpy_array(state_dict["e_discrimination.weight"])) * 10.0

    w1 = _to_numpy_array(state_dict["prednet_full1.weight"])
    b1 = _to_numpy_array(state_dict["prednet_full1.bias"])
    w2 = _to_numpy_array(state_dict["prednet_full2.weight"])
    b2 = _to_numpy_array(state_dict["prednet_full2.bias"])
    w3 = _to_numpy_array(state_dict["prednet_full3.weight"])
    b3 = _to_numpy_array(state_dict["prednet_full3.bias"])

    if q_matrix.shape != k_difficulty.shape:
        raise ValueError(f"q_matrix shape {q_matrix.shape} does not match exercise difficulty shape {k_difficulty.shape}")
    if alpha.shape[1] != q_matrix.shape[1]:
        raise ValueError(f"alpha shape {alpha.shape} does not match q_matrix shape {q_matrix.shape}")

    predictions = np.zeros((alpha.shape[0], q_matrix.shape[0]), dtype=float)
    for item_idx in range(q_matrix.shape[0]):
        item_input = float(e_discrimination[item_idx].reshape(-1)[0]) * (
            alpha - k_difficulty[item_idx][None, :]
        ) * q_matrix[item_idx][None, :]
        hidden1 = _sigmoid(item_input @ w1.T + b1[None, :])
        hidden2 = _sigmoid(hidden1 @ w2.T + b2[None, :])
        output = _sigmoid(hidden2 @ w3.T + b3[None, :]).reshape(-1)
        predictions[:, item_idx] = output
    return predictions


def load_neuralcdm_artifacts(cfg: Dict[str, Any], snapshot_path: str | Path) -> UnifiedArtifacts:
    try:
        import torch
    except ImportError as exc:
        raise ImportError("torch is required to load NeuralCDM snapshots") from exc

    state_dict = torch.load(Path(snapshot_path), map_location="cpu")
    student_emb = _to_numpy_array(state_dict["student_emb.weight"])
    alpha = _sigmoid(student_emb)
    c = np.full((alpha.shape[0],), np.nan, dtype=float)

    ds = _load_dataset_with_raw_q(cfg)
    q_matrix = ds.combined_q()
    expected_students = ds.n_students
    expected_items = ds.n_theory + ds.n_experiment

    if alpha.shape[0] != expected_students:
        raise ValueError(f"NeuralCDM student embedding count {alpha.shape[0]} does not match dataset students {expected_students}")
    if q_matrix.shape[0] != expected_items:
        raise ValueError(f"Combined q rows {q_matrix.shape[0]} does not match expected items {expected_items}")

    rhat_all = _predict_neuralcdm_matrix(alpha=alpha, q_matrix=q_matrix, state_dict=state_dict)
    eta_theory, eta_experiment = _split_predictions(ds, rhat_all)
    return UnifiedArtifacts(
        c=c,
        alpha=alpha,
        rhat_all=rhat_all,
        eta_theory=eta_theory,
        eta_experiment=eta_experiment,
    )


def load_rhat_all_artifacts(cfg: Dict[str, Any], params_path: str | Path) -> UnifiedArtifacts:
    """
    直接从 params.npz 读取 rhat_all 并切分为 theory/experiment。
    适用于由根目录统一训练入口导出的模型（如 dina/irt）。
    """
    ds = load_dataset_from_config(cfg)
    params = np.load(Path(params_path))

    if "rhat_all" not in params:
        raise ValueError(f"Missing key 'rhat_all' in params.npz: {params_path}")

    c = np.asarray(params["c"], dtype=float).reshape(-1)
    alpha = np.asarray(params["alpha"], dtype=float)
    rhat_all = np.asarray(params["rhat_all"], dtype=float)

    eta_theory, eta_experiment = _split_predictions(ds, rhat_all)
    return UnifiedArtifacts(
        c=c,
        alpha=alpha,
        rhat_all=rhat_all,
        eta_theory=eta_theory,
        eta_experiment=eta_experiment,
    )


def load_dina_artifacts(cfg: Dict[str, Any], params_path: str | Path) -> UnifiedArtifacts:
    return load_rhat_all_artifacts(cfg=cfg, params_path=params_path)


def load_irt_artifacts(cfg: Dict[str, Any], params_path: str | Path) -> UnifiedArtifacts:
    return load_rhat_all_artifacts(cfg=cfg, params_path=params_path)


def load_prediction_matrix(path: str | Path) -> np.ndarray:
    in_path = Path(path)
    if in_path.suffix == ".npz":
        data = np.load(in_path)
        if "rhat_all" in data:
            return np.asarray(data["rhat_all"], dtype=float)
        if "eta_theory" in data and "eta_experiment" in data:
            return np.concatenate(
                [np.asarray(data["eta_theory"], dtype=float), np.asarray(data["eta_experiment"], dtype=float)],
                axis=1,
            )
        raise ValueError(f"Unsupported npz keys: {list(data.keys())}")
    return np.asarray(np.loadtxt(in_path), dtype=float)
