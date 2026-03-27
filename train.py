from __future__ import annotations

import argparse
import json

from infra.common import load_unified_config, make_run_dir
from infra.trainers import train_model


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True, choices=["cdf_cse", "fuzzycdf", "neuralcdm", "dina", "irt"])
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--dataset", type=str, default=None)
    p.add_argument("--train_ratio", type=float, default=None)
    p.add_argument("--val_ratio", type=float, default=None)
    p.add_argument("--split_seed", type=int, default=None)
    p.add_argument("--split_mode", type=str, default=None)
    args = p.parse_args()

    cfg = load_unified_config(
        args.config,
        dataset=args.dataset,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        split_seed=args.split_seed,
        split_mode=args.split_mode,
    )
    dataset_name = str(cfg.get("data", {}).get("dataset_name"))
    run_dir = make_run_dir(model_name=str(args.model), dataset_name=dataset_name)
    result = train_model(str(args.model), cfg=cfg, run_dir=run_dir)

    print(str(run_dir))
    print(json.dumps(result.metrics, indent=2))


if __name__ == "__main__":
    main()
