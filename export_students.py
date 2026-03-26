from __future__ import annotations

import argparse
import json
from pathlib import Path

from infra.artifacts import load_run_artifacts, save_export_at_dir
from infra.evaluation import load_artifacts_from_args, load_config_for_args


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True, choices=["cdf_cse", "fuzzycdf", "neuralcdm"])
    p.add_argument("--run_dir", type=str, default=None)
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--dataset", type=str, default=None)
    p.add_argument("--train_ratio", type=float, default=None)
    p.add_argument("--val_ratio", type=float, default=None)
    p.add_argument("--split_seed", type=int, default=None)
    p.add_argument("--split_mode", type=str, default=None)
    p.add_argument("--out_dir", type=str, default=None)

    p.add_argument("--params", type=str, default=None)
    p.add_argument("--fuzzy_dir", type=str, default=None)
    p.add_argument("--alpha_path", type=str, default=None)
    p.add_argument("--theta_path", type=str, default=None)
    p.add_argument("--predictions_path", type=str, default=None)
    p.add_argument("--snapshot", type=str, default=None)
    args = p.parse_args()

    if args.run_dir is not None and (Path(args.run_dir) / "params.npz").exists():
        artifacts = load_run_artifacts(args.run_dir)
    else:
        cfg = load_config_for_args(args)
        artifacts = load_artifacts_from_args(str(args.model), cfg=cfg, args=args)

    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
    elif args.run_dir is not None:
        out_dir = Path(args.run_dir) / "export"
    else:
        raise ValueError("Either --run_dir or --out_dir must be provided")

    out_dir.mkdir(parents=True, exist_ok=True)
    save_export_at_dir(out_dir, artifacts)

    meta_path = out_dir / "export_meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump({"model": args.model, "source_run_dir": args.run_dir}, f, indent=2)

    print(str(out_dir))


if __name__ == "__main__":
    main()
