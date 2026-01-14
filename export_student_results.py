from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from src.data.dataset import load_dataset_from_config
from src.models.cdf_cse import predict
from src.utils.config import load_yaml


def _str2bool(v: str) -> bool:
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y"}:
        return True
    if s in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got: {v}")


def _resolve_config_path(run_dir: Optional[Path], params_path: Path, config_arg: Optional[str]) -> Path:
    if config_arg is not None:
        return Path(config_arg)
    if run_dir is not None:
        return run_dir / "config.yaml"
    return params_path.parent / "config.yaml"


def _apply_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    if args.dataset is not None:
        cfg.setdefault("data", {})["dataset_name"] = str(args.dataset)

    if args.train_ratio is not None:
        cfg.setdefault("split", {})["train_ratio"] = float(args.train_ratio)

    if args.split_seed is not None:
        cfg.setdefault("split", {})["seed"] = int(args.split_seed)

    if args.split_mode is not None:
        cfg.setdefault("data", {})["split_mode"] = str(args.split_mode)

    return cfg


def _default_out_dir(run_dir: Optional[Path]) -> Path:
    if run_dir is not None:
        return run_dir / "export"
    ts = time.strftime("%Y%m%d_%H%M%S")
    return Path("outputs_export") / ts


def _format_float(x: float) -> str:
    return f"{float(x):.6f}"


def _write_students_csv(path: Path, c: np.ndarray, alpha: np.ndarray) -> None:
    n = int(alpha.shape[0])
    k = int(alpha.shape[1])

    header = ["student_id", "c"]
    header.extend([f"alpha_k{i}" for i in range(k)])
    header.extend(["alpha_mean", "alpha_sum", "alpha_topk"])

    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)

        for j in range(n):
            a = alpha[j]
            a_mean = float(np.mean(a))
            a_sum = float(np.sum(a))

            top_n = min(3, k)
            topk_idx = np.argsort(a)[::-1][:top_n]
            topk = "|".join([f"k{int(i)}" for i in topk_idx])

            row = [
                str(j),
                _format_float(c[j]),
            ]
            row.extend([_format_float(x) for x in a.tolist()])
            row.extend([_format_float(a_mean), _format_float(a_sum), topk])
            w.writerow(row)


def main() -> None:
    p = argparse.ArgumentParser()

    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--run_dir", type=str, default=None)
    g.add_argument("--params", type=str, default=None)

    p.add_argument("--config", type=str, default=None)
    p.add_argument("--out_dir", type=str, default=None)

    p.add_argument("--dataset", type=str, default=None)
    p.add_argument("--train_ratio", type=float, default=None)
    p.add_argument("--split_seed", type=int, default=None)
    p.add_argument("--split_mode", type=str, default=None)

    p.add_argument("--save_predictions", type=_str2bool, nargs="?", const=True, default=True)
    p.add_argument("--save_students_csv", type=_str2bool, nargs="?", const=True, default=True)

    args = p.parse_args()

    run_dir: Optional[Path] = Path(args.run_dir) if args.run_dir is not None else None

    if run_dir is not None:
        params_path = run_dir / "params.npz"
    else:
        params_path = Path(str(args.params))

    cfg_path = _resolve_config_path(run_dir=run_dir, params_path=params_path, config_arg=args.config)
    cfg = load_yaml(cfg_path)
    cfg = _apply_overrides(cfg, args)

    ds = load_dataset_from_config(cfg)

    params = np.load(params_path)
    if "c" not in params or "alpha" not in params:
        raise ValueError(f"params.npz must contain keys 'c' and 'alpha', got keys={list(params.keys())}")

    c = np.asarray(params["c"], dtype=float).reshape(-1)
    alpha = np.asarray(params["alpha"], dtype=float)

    if alpha.ndim != 2:
        raise ValueError(f"alpha must be 2D of shape (N,K), got shape={alpha.shape}")

    if int(c.shape[0]) != ds.n_students:
        raise ValueError(f"c has shape {c.shape} but dataset has n_students={ds.n_students}")
    if int(alpha.shape[0]) != ds.n_students:
        raise ValueError(f"alpha has shape {alpha.shape} but dataset has n_students={ds.n_students}")
    if int(alpha.shape[1]) != ds.n_skills:
        raise ValueError(f"alpha has shape {alpha.shape} but dataset has n_skills={ds.n_skills}")

    pred = predict(c=c, alpha=alpha, q_theory=ds.q_theory, q_experiment=ds.q_experiment)
    eta_theory = pred.eta_theory
    eta_experiment = pred.eta_experiment
    rhat_all = np.concatenate([eta_theory, eta_experiment], axis=1)

    out_dir = Path(args.out_dir) if args.out_dir is not None else _default_out_dir(run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if bool(args.save_students_csv):
        _write_students_csv(out_dir / "students.csv", c=c, alpha=alpha)

    if bool(args.save_predictions):
        np.savez_compressed(
            out_dir / "predictions.npz",
            eta_theory=eta_theory,
            eta_experiment=eta_experiment,
            rhat_all=rhat_all,
            c=c,
            alpha=alpha,
        )

    print(str(out_dir))


if __name__ == "__main__":
    main()
