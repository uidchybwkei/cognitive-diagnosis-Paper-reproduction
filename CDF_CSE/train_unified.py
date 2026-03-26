from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict

import yaml

from train import train as train_cdf_cse
from src.utils.config import load_yaml
from src.utils.students import save_predictions_npz, write_students_csv


def _make_run_dir(cfg: Dict[str, Any], model_name: str, dataset_name: str) -> Path:
    out_cfg = cfg.get("outputs", {})
    root = Path(str(out_cfg.get("root", "outputs")))
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = root / model_name / dataset_name / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True, choices=["cdf_cse", "fuzzycdf", "neuralcdm"])
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--dataset", type=str, default=None)
    p.add_argument("--train_ratio", type=float, default=None)
    args = p.parse_args()

    cfg = load_yaml(args.config)
    if args.dataset is not None:
        cfg.setdefault("data", {})["dataset_name"] = args.dataset
    if args.train_ratio is not None:
        cfg.setdefault("split", {})["train_ratio"] = float(args.train_ratio)

    model_name = str(args.model)
    dataset_name = str(cfg.get("data", {}).get("dataset_name"))

    if model_name != "cdf_cse":
        raise NotImplementedError(
            "train_unified.py currently supports model=cdf_cse only. "
            "For fuzzycdf and neuralcdm, train them externally first, then use "
            "export_unified_results.py and eval_unified.py to normalize outputs."
        )

    run_dir = _make_run_dir(cfg=cfg, model_name=model_name, dataset_name=dataset_name)
    result = train_cdf_cse(cfg)

    with (run_dir / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(result["final_metrics"], f, indent=2)

    out_cfg = cfg.get("outputs", {})
    if bool(out_cfg.get("save_history", True)):
        with (run_dir / "history.json").open("w", encoding="utf-8") as f:
            json.dump(result["history"], f, indent=2)

    if bool(out_cfg.get("save_params", True)):
        import numpy as np

        np.savez_compressed(
            run_dir / "params.npz",
            c=result["c"],
            alpha=result["alpha"],
        )

    export_dir = run_dir / "export"
    write_students_csv(export_dir / "students.csv", c=result["c"], alpha=result["alpha"])
    save_predictions_npz(
        export_dir / "predictions.npz",
        eta_theory=result["pred"].eta_theory,
        eta_experiment=result["pred"].eta_experiment,
        c=result["c"],
        alpha=result["alpha"],
    )

    print(str(run_dir))
    print(json.dumps(result["final_metrics"], indent=2))


if __name__ == "__main__":
    main()
