from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

from src.utils.config import load_yaml
from src.utils.students import save_predictions_npz, write_students_csv
from src.utils.unified_models import (
    UnifiedArtifacts,
    load_cdf_cse_artifacts,
    load_fuzzycdf_artifacts,
    load_neuralcdm_artifacts,
)


def _resolve_config_path(run_dir: Optional[Path], config_arg: Optional[str]) -> Optional[Path]:
    if config_arg is not None:
        return Path(config_arg)
    if run_dir is None:
        return None
    default_cfg = run_dir / "config.yaml"
    if default_cfg.exists():
        return default_cfg
    return None


def _default_out_dir(model_name: str, run_dir: Optional[Path]) -> Path:
    if run_dir is not None:
        return run_dir / "export"
    ts = time.strftime("%Y%m%d_%H%M%S")
    return Path("outputs_export") / model_name / ts


def _load_artifacts(args: argparse.Namespace, cfg: Optional[Dict[str, Any]]) -> UnifiedArtifacts:
    model_name = str(args.model)
    if model_name == "cdf_cse":
        if args.params is None:
            raise ValueError("--params is required for model=cdf_cse unless --run_dir is provided")
        if cfg is None:
            raise ValueError("config is required for model=cdf_cse")
        return load_cdf_cse_artifacts(cfg=cfg, params_path=args.params)

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
        if cfg is None:
            raise ValueError("config is required for model=neuralcdm")
        return load_neuralcdm_artifacts(cfg=cfg, snapshot_path=args.snapshot)

    raise ValueError(f"Unsupported model={model_name}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True, choices=["cdf_cse", "fuzzycdf", "neuralcdm"])
    p.add_argument("--run_dir", type=str, default=None)
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--out_dir", type=str, default=None)

    p.add_argument("--params", type=str, default=None)

    p.add_argument("--fuzzy_dir", type=str, default=None)
    p.add_argument("--alpha_path", type=str, default=None)
    p.add_argument("--theta_path", type=str, default=None)
    p.add_argument("--predictions_path", type=str, default=None)

    p.add_argument("--snapshot", type=str, default=None)

    args = p.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir is not None else None

    if args.model == "cdf_cse" and args.params is None and run_dir is not None:
        args.params = str(run_dir / "params.npz")

    cfg_path = _resolve_config_path(run_dir=run_dir, config_arg=args.config)
    cfg = load_yaml(cfg_path) if cfg_path is not None else None
    artifacts = _load_artifacts(args=args, cfg=cfg)

    out_dir = Path(args.out_dir) if args.out_dir is not None else _default_out_dir(args.model, run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    write_students_csv(out_dir / "students.csv", c=artifacts.c, alpha=artifacts.alpha)
    save_predictions_npz(
        out_dir / "predictions.npz",
        eta_theory=artifacts.eta_theory,
        eta_experiment=artifacts.eta_experiment,
        c=artifacts.c,
        alpha=artifacts.alpha,
        extra_arrays={"rhat_all": artifacts.rhat_all} if artifacts.rhat_all is not None else None,
    )

    with (out_dir / "export_meta.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model": args.model,
                "run_dir": str(run_dir) if run_dir is not None else None,
                "config": str(cfg_path) if cfg_path is not None else None,
            },
            f,
            indent=2,
        )

    print(str(out_dir))


if __name__ == "__main__":
    main()
