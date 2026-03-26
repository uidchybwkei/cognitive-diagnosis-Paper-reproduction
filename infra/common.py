from __future__ import annotations

import copy
import importlib.util
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CDF_CSE_ROOT = REPO_ROOT / "CDF_CSE"
FUZZYCDF_ROOT = REPO_ROOT / "FuzzyCDF"
NEURALCDM_ROOT = REPO_ROOT / "NeuralCDM"
RUNS_ROOT = REPO_ROOT / "runs"
DEFAULT_CONFIG_PATH = CDF_CSE_ROOT / "configs" / "default.yaml"

if str(CDF_CSE_ROOT) not in sys.path:
    sys.path.insert(0, str(CDF_CSE_ROOT))

from src.utils.config import load_yaml  # noqa: E402


def load_python_module(module_name: str, file_path: str | Path):
    spec = importlib.util.spec_from_file_location(module_name, str(Path(file_path)))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_dataset_root(dataset_root: str | Path) -> Path:
    root = Path(dataset_root)
    if root.is_absolute():
        return root
    return (CDF_CSE_ROOT / root).resolve()


def load_unified_config(
    config_path: Optional[str | Path] = None,
    *,
    dataset: Optional[str] = None,
    train_ratio: Optional[float] = None,
    val_ratio: Optional[float] = None,
    split_seed: Optional[int] = None,
    split_mode: Optional[str] = None,
) -> Dict[str, Any]:
    cfg_path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    cfg = load_yaml(cfg_path)
    cfg = copy.deepcopy(cfg)

    data_cfg = cfg.setdefault("data", {})
    split_cfg = cfg.setdefault("split", {})
    outputs_cfg = cfg.setdefault("outputs", {})

    data_cfg["dataset_root"] = str(resolve_dataset_root(data_cfg.get("dataset_root", "dataset")))
    outputs_cfg["root"] = str(RUNS_ROOT)

    if dataset is not None:
        data_cfg["dataset_name"] = str(dataset)
    if train_ratio is not None:
        split_cfg["train_ratio"] = float(train_ratio)
    if val_ratio is not None:
        split_cfg["val_ratio"] = float(val_ratio)
    if split_seed is not None:
        split_cfg["seed"] = int(split_seed)
    if split_mode is not None:
        data_cfg["split_mode"] = str(split_mode)

    return cfg


def make_run_dir(model_name: str, dataset_name: str) -> Path:
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_ROOT / model_name / dataset_name / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_config_yaml(run_dir: str | Path, cfg: Dict[str, Any]) -> None:
    out_path = Path(run_dir) / "config.yaml"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
