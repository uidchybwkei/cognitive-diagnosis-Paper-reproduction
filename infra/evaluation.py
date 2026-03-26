from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Literal, Optional

import numpy as np

from infra.artifacts import load_run_artifacts
from infra.common import load_unified_config
from src.data.dataset import load_dataset_from_config, make_splits
from src.utils.metrics import masked_mae, masked_rmse
from src.utils.unified_models import (
    UnifiedArtifacts,
    load_cdf_cse_artifacts,
    load_fuzzycdf_artifacts,
    load_neuralcdm_artifacts,
)


def metrics_dict(prefix: str, y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    return {
        f"{prefix}_mae": masked_mae(y_true, y_pred, mask),
        f"{prefix}_rmse": masked_rmse(y_true, y_pred, mask),
        f"{prefix}_n": float(np.asarray(mask, dtype=bool).sum()),
    }


def load_artifacts_from_args(model_name: str, *, cfg: Dict[str, Any], args) -> UnifiedArtifacts:
    if getattr(args, "run_dir", None) is not None:
        run_dir = Path(args.run_dir)
        params_path = run_dir / "params.npz"
        if params_path.exists():
            return load_run_artifacts(run_dir)

    if model_name == "cdf_cse":
        if getattr(args, "params", None) is None:
            raise ValueError("--params is required for model=cdf_cse when --run_dir is unavailable")
        return load_cdf_cse_artifacts(cfg=cfg, params_path=args.params)
    if model_name == "fuzzycdf":
        return load_fuzzycdf_artifacts(
            cfg=cfg,
            fuzzy_dir=getattr(args, "fuzzy_dir", None),
            alpha_path=getattr(args, "alpha_path", None),
            theta_path=getattr(args, "theta_path", None),
            predictions_path=getattr(args, "predictions_path", None),
        )
    if model_name == "neuralcdm":
        if getattr(args, "snapshot", None) is None:
            raise ValueError("--snapshot is required for model=neuralcdm when --run_dir is unavailable")
        return load_neuralcdm_artifacts(cfg=cfg, snapshot_path=args.snapshot)
    raise ValueError(f"Unsupported model={model_name}")


def evaluate_artifacts(
    cfg: Dict[str, Any],
    artifacts: UnifiedArtifacts,
    *,
    split: Literal["train", "val", "test"] = "test",
    model_name: str,
    neural_threshold: float = 0.5,
) -> Dict[str, Any]:
    ds = load_dataset_from_config(cfg)

    split_cfg = cfg.get("split", {})
    train_ratio = float(split_cfg.get("train_ratio", 0.8))
    val_ratio = float(split_cfg.get("val_ratio", 0.0))
    split_seed = int(split_cfg.get("seed", 42))
    split_mode = str(cfg.get("data", {}).get("split_mode", "combined"))
    splits = make_splits(
        ds,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=split_seed,
        split_mode=split_mode,
    )

    if split == "train":
        mask_theory = splits.train_theory
        mask_experiment = splits.train_experiment
    elif split == "val":
        mask_theory = splits.val_theory
        mask_experiment = splits.val_experiment
    elif split == "test":
        mask_theory = splits.test_theory
        mask_experiment = splits.test_experiment
    else:
        raise ValueError(f"Unsupported split={split}")

    if artifacts.eta_theory is not None and artifacts.eta_experiment is not None:
        pred_theory = np.asarray(artifacts.eta_theory, dtype=float)
        pred_experiment = np.asarray(artifacts.eta_experiment, dtype=float)
    elif artifacts.rhat_all is not None:
        pred_theory = np.asarray(artifacts.rhat_all[:, : ds.n_theory], dtype=float)
        pred_experiment = np.asarray(artifacts.rhat_all[:, ds.n_theory :], dtype=float)
    else:
        raise ValueError("Artifacts do not contain predictions")

    true_theory = np.asarray(ds.r_theory, dtype=float)
    true_experiment = np.asarray(ds.r_experiment, dtype=float)
    if model_name == "neuralcdm":
        true_theory = (true_theory >= float(neural_threshold)).astype(float)
        true_experiment = (true_experiment >= float(neural_threshold)).astype(float)

    metrics: Dict[str, Any] = {
        "model": str(model_name),
        "dataset": ds.name,
        "split": str(split),
        "train_ratio": float(train_ratio),
        "val_ratio": float(val_ratio),
        "split_seed": int(split_seed),
        "split_mode": str(split_mode),
    }
    if model_name == "neuralcdm":
        metrics["label_threshold"] = float(neural_threshold)

    metrics.update(metrics_dict(f"{split}_theory", true_theory, pred_theory, mask_theory))
    metrics.update(metrics_dict(f"{split}_experiment", true_experiment, pred_experiment, mask_experiment))

    combined_true = np.concatenate([true_theory, true_experiment], axis=1)
    combined_pred = np.concatenate([pred_theory, pred_experiment], axis=1)
    combined_mask = np.concatenate([mask_theory, mask_experiment], axis=1)
    metrics.update(metrics_dict(f"{split}_all", combined_true, combined_pred, combined_mask))
    return metrics


def load_config_for_args(args) -> Dict[str, Any]:
    config_path = getattr(args, "config", None)
    run_dir = getattr(args, "run_dir", None)
    if config_path is None and run_dir is not None:
        candidate = Path(run_dir) / "config.yaml"
        if candidate.exists():
            config_path = candidate

    return load_unified_config(
        config_path,
        dataset=getattr(args, "dataset", None),
        train_ratio=getattr(args, "train_ratio", None),
        val_ratio=getattr(args, "val_ratio", None),
        split_seed=getattr(args, "split_seed", None),
        split_mode=getattr(args, "split_mode", None),
    )
