from __future__ import annotations

import argparse
import json
from pathlib import Path

from infra.evaluation import evaluate_artifacts, load_artifacts_from_args, load_config_for_args


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True, choices=["cdf_cse", "fuzzycdf", "neuralcdm", "dina", "irt"])
    p.add_argument("--run_dir", type=str, default=None)
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--dataset", type=str, default=None)
    p.add_argument("--train_ratio", type=float, default=None)
    p.add_argument("--val_ratio", type=float, default=None)
    p.add_argument("--split_seed", type=int, default=None)
    p.add_argument("--split_mode", type=str, default=None)
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

    cfg = load_config_for_args(args)
    artifacts = load_artifacts_from_args(str(args.model), cfg=cfg, args=args)
    metrics = evaluate_artifacts(
        cfg=cfg,
        artifacts=artifacts,
        split=str(args.split),
        model_name=str(args.model),
        neural_threshold=float(args.neural_threshold),
    )

    out_path = None
    if args.out is not None:
        out_path = Path(args.out)
    elif args.run_dir is not None:
        out_path = Path(args.run_dir) / f"eval_{args.split}.json"

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
