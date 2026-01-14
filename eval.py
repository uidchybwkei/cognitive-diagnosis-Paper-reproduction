from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import numpy as np

from src.data.dataset import load_dataset_from_config, make_splits
from src.models.cdf_cse import objective_F, predict
from src.utils.config import load_yaml
from src.utils.metrics import masked_mae, masked_rmse


def _metrics_dict(prefix: str, y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    return {
        f"{prefix}_mae": masked_mae(y_true, y_pred, mask),
        f"{prefix}_rmse": masked_rmse(y_true, y_pred, mask),
        f"{prefix}_n": float(np.asarray(mask, dtype=bool).sum()),
    }


def evaluate_run(
    cfg: Dict[str, Any],
    c: np.ndarray,
    alpha: np.ndarray,
    split: Literal["train", "val", "test"] = "test",
    compute_objective: bool = False,
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

    pred = predict(c=c, alpha=alpha, q_theory=ds.q_theory, q_experiment=ds.q_experiment)

    metrics: Dict[str, Any] = {
        "dataset": ds.name,
        "split": str(split),
        "train_ratio": float(train_ratio),
        "val_ratio": float(val_ratio),
        "split_seed": int(split_seed),
        "split_mode": str(split_mode),
    }

    metrics.update(_metrics_dict(f"{split}_theory", ds.r_theory, pred.eta_theory, mask_theory))
    metrics.update(
        _metrics_dict(
            f"{split}_experiment",
            ds.r_experiment,
            pred.eta_experiment,
            mask_experiment,
        )
    )

    combined_true = np.concatenate([ds.r_theory, ds.r_experiment], axis=1)
    combined_pred = np.concatenate([pred.eta_theory, pred.eta_experiment], axis=1)
    combined_mask = np.concatenate([mask_theory, mask_experiment], axis=1)
    metrics.update(_metrics_dict(f"{split}_all", combined_true, combined_pred, combined_mask))

    if compute_objective:
        model_cfg = cfg.get("model", {})
        sigma_r = float(model_cfg.get("sigma_r", 1.0))
        sigma_rp = float(model_cfg.get("sigma_rp", 1.0))
        sigma_alpha = float(model_cfg.get("sigma_alpha", 1.0))
        sigma_c = float(model_cfg.get("sigma_c", 1.0))
        mu_alpha = float(model_cfg.get("mu_alpha", 0.0))
        mu_c = float(model_cfg.get("mu_c", 0.0))

        f_val = objective_F(
            c=c,
            alpha=alpha,
            r_theory=ds.r_theory,
            r_experiment=ds.r_experiment,
            q_theory=ds.q_theory,
            q_experiment=ds.q_experiment,
            mask_theory=mask_theory,
            mask_experiment=mask_experiment,
            sigma_r=sigma_r,
            sigma_rp=sigma_rp,
            sigma_alpha=sigma_alpha,
            sigma_c=sigma_c,
            mu_alpha=mu_alpha,
            mu_c=mu_c,
        )
        metrics[f"{split}_F"] = float(f_val)

    return metrics


def _resolve_config_path(run_dir: Optional[Path], config_arg: Optional[str]) -> Path:
    if config_arg is not None:
        return Path(config_arg)
    if run_dir is None:
        raise ValueError("--config is required when --run_dir is not provided")
    return run_dir / "config.yaml"


def main() -> None:
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--run_dir", type=str, default=None)
    g.add_argument("--params", type=str, default=None)

    p.add_argument("--config", type=str, default=None)
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    p.add_argument("--compute_objective", action="store_true", default=False)

    p.add_argument("--dataset", type=str, default=None)
    p.add_argument("--train_ratio", type=float, default=None)
    p.add_argument("--val_ratio", type=float, default=None)
    p.add_argument("--split_seed", type=int, default=None)
    p.add_argument("--split_mode", type=str, default=None)

    p.add_argument("--out", type=str, default=None)

    args = p.parse_args()

    run_dir: Optional[Path] = Path(args.run_dir) if args.run_dir is not None else None

    if args.params is not None:
        params_path = Path(args.params)
        if run_dir is None:
            run_dir = params_path.parent
    else:
        params_path = Path(args.run_dir) / "params.npz"

    cfg_path = _resolve_config_path(run_dir=run_dir, config_arg=args.config)
    cfg = load_yaml(cfg_path)

    if args.dataset is not None:
        cfg.setdefault("data", {})["dataset_name"] = str(args.dataset)

    if args.train_ratio is not None:
        cfg.setdefault("split", {})["train_ratio"] = float(args.train_ratio)
    if args.val_ratio is not None:
        cfg.setdefault("split", {})["val_ratio"] = float(args.val_ratio)
    if args.split_seed is not None:
        cfg.setdefault("split", {})["seed"] = int(args.split_seed)
    if args.split_mode is not None:
        cfg.setdefault("data", {})["split_mode"] = str(args.split_mode)

    params = np.load(params_path)
    c = np.asarray(params["c"], dtype=float)
    alpha = np.asarray(params["alpha"], dtype=float)

    metrics = evaluate_run(
        cfg=cfg,
        c=c,
        alpha=alpha,
        split=str(args.split),
        compute_objective=bool(args.compute_objective),
    )

    if args.out is not None:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
