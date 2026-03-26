from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from infra.common import save_config_yaml
from src.utils.students import save_predictions_npz, write_students_csv
from src.utils.unified_models import UnifiedArtifacts


def save_history_json(run_dir: str | Path, history: Any) -> None:
    out_path = Path(run_dir) / "history.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def save_metrics_json(run_dir: str | Path, metrics: Dict[str, Any]) -> None:
    out_path = Path(run_dir) / "metrics.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def save_unified_params(run_dir: str | Path, artifacts: UnifiedArtifacts) -> None:
    out_path = Path(run_dir) / "params.npz"
    arrays: Dict[str, np.ndarray] = {
        "c": np.asarray(artifacts.c, dtype=float),
        "alpha": np.asarray(artifacts.alpha, dtype=float),
    }
    if artifacts.rhat_all is not None:
        arrays["rhat_all"] = np.asarray(artifacts.rhat_all, dtype=float)
    np.savez_compressed(out_path, **arrays)


def save_export_at_dir(out_dir: str | Path, artifacts: UnifiedArtifacts) -> None:
    export_dir = Path(out_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    write_students_csv(export_dir / "students.csv", c=artifacts.c, alpha=artifacts.alpha)
    save_predictions_npz(
        export_dir / "predictions.npz",
        eta_theory=artifacts.eta_theory,
        eta_experiment=artifacts.eta_experiment,
        c=artifacts.c,
        alpha=artifacts.alpha,
        extra_arrays={"rhat_all": artifacts.rhat_all} if artifacts.rhat_all is not None else None,
    )


def save_export_dir(run_dir: str | Path, artifacts: UnifiedArtifacts) -> None:
    export_dir = Path(run_dir) / "export"
    save_export_at_dir(export_dir, artifacts)


def save_run_bundle(
    run_dir: str | Path,
    *,
    cfg: Dict[str, Any],
    metrics: Dict[str, Any],
    artifacts: UnifiedArtifacts,
    history: Optional[Any] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    save_config_yaml(run_dir, cfg)
    save_metrics_json(run_dir, metrics)
    if history is not None:
        save_history_json(run_dir, history)
    save_unified_params(run_dir, artifacts)
    save_export_dir(run_dir, artifacts)
    if meta is not None:
        out_path = Path(run_dir) / "run_meta.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)


def load_run_artifacts(run_dir: str | Path) -> UnifiedArtifacts:
    run_path = Path(run_dir)
    params_path = run_path / "params.npz"
    if not params_path.exists():
        raise FileNotFoundError(f"Missing params.npz in run_dir: {run_path}")

    params = np.load(params_path)
    c = np.asarray(params["c"], dtype=float).reshape(-1)
    alpha = np.asarray(params["alpha"], dtype=float)
    rhat_all = np.asarray(params["rhat_all"], dtype=float) if "rhat_all" in params else None

    eta_theory = None
    eta_experiment = None
    pred_path = run_path / "export" / "predictions.npz"
    if pred_path.exists():
        pred = np.load(pred_path)
        if "eta_theory" in pred:
            eta_theory = np.asarray(pred["eta_theory"], dtype=float)
        if "eta_experiment" in pred:
            eta_experiment = np.asarray(pred["eta_experiment"], dtype=float)
        if rhat_all is None and "rhat_all" in pred:
            rhat_all = np.asarray(pred["rhat_all"], dtype=float)

    return UnifiedArtifacts(
        c=c,
        alpha=alpha,
        rhat_all=rhat_all,
        eta_theory=eta_theory,
        eta_experiment=eta_experiment,
    )
