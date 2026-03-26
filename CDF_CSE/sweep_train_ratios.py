from __future__ import annotations

import argparse
import copy
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

import train as train_module
from src.utils.config import load_yaml


def _timestamp_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def _as_list(x: Any) -> List[float]:
    if x is None:
        return []
    if isinstance(x, list):
        return [float(v) for v in x]
    return [float(x)]


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if len(rows) == 0:
        return

    keys: List[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                keys.append(str(k))
                seen.add(str(k))

    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--dataset", type=str, default=None)
    p.add_argument("--train_ratios", type=float, nargs="+", default=None)
    p.add_argument("--val_ratio", type=float, default=None)
    p.add_argument("--split_seed", type=int, default=None)
    p.add_argument("--split_mode", type=str, default=None)
    p.add_argument("--out_dir", type=str, default=None)

    args = p.parse_args()

    cfg = load_yaml(args.config)

    if args.dataset is not None:
        cfg.setdefault("data", {})["dataset_name"] = str(args.dataset)

    ds_name = str(cfg.get("data", {}).get("dataset_name"))

    if args.val_ratio is not None:
        cfg.setdefault("split", {})["val_ratio"] = float(args.val_ratio)

    if args.split_seed is not None:
        cfg.setdefault("split", {})["seed"] = int(args.split_seed)

    if args.split_mode is not None:
        cfg.setdefault("data", {})["split_mode"] = str(args.split_mode)

    train_ratios: List[float]
    if args.train_ratios is not None:
        train_ratios = [float(x) for x in args.train_ratios]
    else:
        split_cfg = cfg.get("split", {})
        train_ratios = _as_list(split_cfg.get("train_ratios_sparsity"))
        if len(train_ratios) == 0:
            train_ratios = [float(split_cfg.get("train_ratio", 0.8))]

    if len(train_ratios) == 0:
        raise ValueError("No train_ratios provided and split.train_ratios_sparsity is empty")

    out_root: Path
    if args.out_dir is not None:
        out_root = Path(args.out_dir)
    else:
        out_cfg = cfg.get("outputs", {})
        out_root = Path(str(out_cfg.get("root", "outputs")))

    sweep_dir = out_root / ds_name / f"sweep_{_timestamp_id()}"
    sweep_dir.mkdir(parents=True, exist_ok=False)

    with (sweep_dir / "base_config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    summary: List[Dict[str, Any]] = []

    for tr in train_ratios:
        cfg_run = copy.deepcopy(cfg)
        cfg_run.setdefault("split", {})["train_ratio"] = float(tr)

        ratio_tag = f"{float(tr):g}"
        run_dir = sweep_dir / f"ratio_{ratio_tag}"
        run_dir.mkdir(parents=True, exist_ok=False)

        result = train_module.train(cfg_run)

        with (run_dir / "config.yaml").open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg_run, f, sort_keys=False)

        with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(result["final_metrics"], f, indent=2)

        out_cfg = cfg_run.get("outputs", {})
        if bool(out_cfg.get("save_history", True)):
            with (run_dir / "history.json").open("w", encoding="utf-8") as f:
                json.dump(result["history"], f, indent=2)

        if bool(out_cfg.get("save_params", True)):
            np.savez_compressed(run_dir / "params.npz", c=result["c"], alpha=result["alpha"])

        row = dict(result["final_metrics"])
        row["run_dir"] = str(run_dir)
        summary.append(row)

    with (sweep_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    _write_csv(sweep_dir / "summary.csv", summary)

    print(json.dumps({"sweep_dir": str(sweep_dir), "n_runs": len(summary)}, indent=2))


if __name__ == "__main__":
    main()
