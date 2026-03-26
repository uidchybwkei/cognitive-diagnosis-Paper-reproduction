from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import numpy as np

from src.data.dataset import load_dataset_from_config, make_splits
from src.utils.config import load_yaml
from src.utils.metrics import masked_mae, masked_rmse
from src.utils.unified_models import (
    UnifiedArtifacts,
    load_cdf_cse_artifacts,
    load_fuzzycdf_artifacts,
    load_neuralcdm_artifacts,
)


def _metrics_dict(prefix: str, y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    return {
        f"{prefix}_mae": masked_mae(y_true, y_pred, mask),
        f"{prefix}_rmse": masked_rmse(y_true, y_pred, mask),
        f"{prefix}_n": float(np.asarray(mask, dtype=bool).sum()),
    }


def _resolve_config_path(run_dir: Optional[Path], config_arg: Optional[str]) -> Optional[Path]:
    if config_arg is not None:
        return Path(config_arg)
    if run_dir is None:
        return None
    default_cfg = run_dir / "config.yaml"
    if default_cfg.exists():
        return default_cfg
    return None


def _load_artifacts(model_name: str, args: argparse.Namespace, cfg: Dict[str, Any]) -> UnifiedArtifacts:
    if model_name == "cdf_cse":
        params_path = args.params
        if params_path is None:
            if args.run_dir is None:
                raise ValueError("--params is required for model=cdf_cse unless --run_dir is provided")
            params_path = str(Path(args.run_dir) / "params.npz")
        return load_cdf_cse_artifacts(cfg=cfg, params_path=params_path)

    if model_name == "fuzzycdf":
        return load_fuzzycdf_artifacts(
            cfg=cfg,
            fuzzy_dir=args.fuzzy_dir,
            alpha_path=args.alpha_path,
            theta_path=args.theta_path,
            predictions_path=args.predictions_path,
        )

    if model_name == "neuralcdm":
        if args.snapshot is None:
            raise ValueError("--snapshot is required for model=neuralcdm")
        return load_neuralcdm_artifacts(cfg=cfg, snapshot_path=args.snapshot)

    raise ValueError(f"Unsupported model={model_name}")


def evaluate_artifacts(
    cfg: Dict[str, Any],
    artifacts: UnifiedArtifacts,
    split: Literal["train", "val", "test"] = "test",
    model_name: str = "cdf_cse",
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
        pred_all = np.asarray(artifacts.rhat_all, dtype=float)
        pred_theory = pred_all[:, : ds.n_theory]
        pred_experiment = pred_all[:, ds.n_theory :]
    else:
        raise ValueError("UnifiedArtifacts must contain either eta_theory/eta_experiment or rhat_all")

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

    metrics.update(_metrics_dict(f"{split}_theory", true_theory, pred_theory, mask_theory))
    metrics.update(_metrics_dict(f"{split}_experiment", true_experiment, pred_experiment, mask_experiment))

    combined_true = np.concatenate([true_theory, true_experiment], axis=1)
    combined_pred = np.concatenate([pred_theory, pred_experiment], axis=1)
    combined_mask = np.concatenate([mask_theory, mask_experiment], axis=1)
    metrics.update(_metrics_dict(f"{split}_all", combined_true, combined_pred, combined_mask))
    return metrics


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True, choices=["cdf_cse", "fuzzycdf", "neuralcdm"])
    p.add_argument("--run_dir", type=str, default=None)
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    p.add_argument("--out", type=str, default=None)

    p.add_argument("--params", type=str, default=None)

    p.add_argument("--fuzzy_dir", type=str, default=None)
    p.add_argument("--alpha_path", type=str, default=None)
    p.add_argument("--theta_path", type=str, default=None)
    p.add_argument("--predictions_path", type=str, default=None)

    p.add_argument("--snapshot", type=str, default=None)
    p.add_argument("--neural_threshold", type=float, default=0.5)

    args = p.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir is not None else None
    cfg_path = _resolve_config_path(run_dir=run_dir, config_arg=args.config)
    if cfg_path is None:
        raise ValueError("A config path is required. Use --config or provide --run_dir containing config.yaml")
    cfg = load_yaml(cfg_path)

    artifacts = _load_artifacts(model_name=str(args.model), args=args, cfg=cfg)
    metrics = evaluate_artifacts(
        cfg=cfg,
        artifacts=artifacts,
        split=str(args.split),
        model_name=str(args.model),
        neural_threshold=float(args.neural_threshold),
    )

    if args.out is not None:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
