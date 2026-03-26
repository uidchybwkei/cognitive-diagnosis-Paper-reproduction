from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import yaml

from src.data.dataset import load_dataset_from_config, make_splits
from src.models.cdf_cse import objective_F, predict, update_alpha, update_c
from src.utils.config import load_yaml
from src.utils.metrics import masked_mae, masked_rmse


def _init_params(m: int, k: int, cfg: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    初始化模型参数 c 和 alpha
    
    输入:
        m: 学生数量 (int)
        k: 技能数量 (int)
        cfg: 配置字典，包含以下键:
            - training.seed: 随机种子
            - training.init_method: 初始化方法 ("prior_sample" 或 "mean")
            - model.mu_c: c 的均值
            - model.mu_alpha: alpha 的均值
            - model.sigma_c: c 的精度参数 (必须 > 0)
            - model.sigma_alpha: alpha 的精度参数 (必须 > 0)
    
    输出:
        c: 学生能力参数数组，形状 (m,)
        alpha: 学生-技能矩阵，形状 (m, k)
    """
    train_cfg = cfg.get("training", {})
    model_cfg = cfg.get("model", {})

    seed = int(train_cfg.get("seed", 42))
    rng = np.random.default_rng(seed)

    mu_c = float(model_cfg.get("mu_c", 0.0))
    mu_alpha = float(model_cfg.get("mu_alpha", 0.0))
    sigma_c = float(model_cfg.get("sigma_c", 1.0))
    sigma_alpha = float(model_cfg.get("sigma_alpha", 1.0))

    if sigma_c <= 0 or sigma_alpha <= 0:
        raise ValueError("sigma_c and sigma_alpha must be > 0")

    method = str(train_cfg.get("init_method", "prior_sample"))

    # 论文中 Eq.3.3-5 假设参数服从如下先验分布
    if method == "prior_sample":
        # c_j \sim \mathcal{N}(\mu_c, \sigma_c^{-1} I)
        c = rng.normal(loc=mu_c, scale=1.0 / np.sqrt(sigma_c), size=(m,))
        # \alpha_{jk} \sim \mathcal{N}(\mu_\alpha, \sigma_\alpha^{-1} I)
        alpha = rng.normal(loc=mu_alpha, scale=1.0 / np.sqrt(sigma_alpha), size=(m, k))
        return c, alpha

    if method == "mean":
        c = np.full((m,), mu_c, dtype=float)
        alpha = np.full((m, k), mu_alpha, dtype=float)
        return c, alpha

    raise ValueError(f"Unsupported init_method={method}")


def _make_run_dir(cfg: Dict[str, Any], dataset_name: str) -> Path:
    """
    创建运行输出目录
    
    输入:
        cfg: 配置字典，包含 outputs.root 键指定输出根目录
        dataset_name: 数据集名称 (str)
    
    输出:
        run_dir: 创建的运行目录路径对象 (Path)
    """
    out_cfg = cfg.get("outputs", {})
    root = Path(str(out_cfg.get("root", "outputs")))
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = root / dataset_name / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _metrics_dict(prefix: str, y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """
    计算并返回指标字典
    
    输入:
        prefix: 指标名称前缀 (str)
        y_true: 真实值数组
        y_pred: 预测值数组
        mask: 布尔掩码数组
    
    输出:
        包含以下键的字典:
            - {prefix}_mae: 平均绝对误差
            - {prefix}_rmse: 均方根误差
            - {prefix}_n: 有效样本数量
    """
    return {
        f"{prefix}_mae": masked_mae(y_true, y_pred, mask),
        f"{prefix}_rmse": masked_rmse(y_true, y_pred, mask),
        f"{prefix}_n": float(np.asarray(mask, dtype=bool).sum()),
    }


def train(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    训练认知诊断模型
    
    输入:
        cfg: 完整配置字典，包含:
            - data: 数据配置 (dataset_name, split_mode 等)
            - split: 数据分割配置 (train_ratio, val_ratio, seed)
            - model: 模型参数 (mu_c, mu_alpha, sigma_c, sigma_alpha, sigma_r, sigma_rp)
            - training: 训练配置 (n_iters, r1, r2, log_every, clip_params 等)
    
    输出:
        包含以下键的字典:
            - final_metrics: 最终评估指标 (包含各种 MAE, RMSE, 样本数等)
            - history: 训练历史记录列表 (每轮迭代的指标)
            - c: 最终的学生能力参数数组
            - alpha: 最终的学生-技能矩阵
            - pred: 最终预测结果对象
    """
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

    m = ds.n_students
    k = ds.n_skills

    c, alpha = _init_params(m=m, k=k, cfg=cfg)

    model_cfg = cfg.get("model", {})
    sigma_r = float(model_cfg.get("sigma_r", 1.0))
    sigma_rp = float(model_cfg.get("sigma_rp", 1.0))
    sigma_alpha = float(model_cfg.get("sigma_alpha", 1.0))
    sigma_c = float(model_cfg.get("sigma_c", 1.0))
    mu_alpha = float(model_cfg.get("mu_alpha", 0.0))
    mu_c = float(model_cfg.get("mu_c", 0.0))

    train_cfg = cfg.get("training", {})
    n_iters = int(train_cfg.get("n_iters", 1000))
    r1 = float(train_cfg.get("r1", 0.01))
    r2 = float(train_cfg.get("r2", 0.01))
    log_every = int(train_cfg.get("log_every", 50))

    clip_params = bool(train_cfg.get("clip_params", False))
    clip_min = float(train_cfg.get("clip_min", 0.0))
    clip_max = float(train_cfg.get("clip_max", 1.0))

    history = []

    t0 = time.time()
    for it in range(1, n_iters + 1):
        c = update_c(
            c=c,
            alpha=alpha,
            r_experiment=ds.r_experiment,
            q_experiment=ds.q_experiment,
            mask_experiment=splits.train_experiment,
            sigma_rp=sigma_rp,
            sigma_c=sigma_c,
            mu_c=mu_c,
            lr=r1,
        )

        if clip_params:
            c = np.clip(c, clip_min, clip_max)

        alpha = update_alpha(
            c=c,
            alpha=alpha,
            r_theory=ds.r_theory,
            r_experiment=ds.r_experiment,
            q_theory=ds.q_theory,
            q_experiment=ds.q_experiment,
            mask_theory=splits.train_theory,
            mask_experiment=splits.train_experiment,
            sigma_r=sigma_r,
            sigma_rp=sigma_rp,
            sigma_alpha=sigma_alpha,
            mu_alpha=mu_alpha,
            lr=r2,
        )

        if clip_params:
            alpha = np.clip(alpha, clip_min, clip_max)

        do_log = (it == 1) or (it == n_iters) or (log_every > 0 and it % log_every == 0)
        if do_log:
            f_train = objective_F(
                c=c,
                alpha=alpha,
                r_theory=ds.r_theory,
                r_experiment=ds.r_experiment,
                q_theory=ds.q_theory,
                q_experiment=ds.q_experiment,
                mask_theory=splits.train_theory,
                mask_experiment=splits.train_experiment,
                sigma_r=sigma_r,
                sigma_rp=sigma_rp,
                sigma_alpha=sigma_alpha,
                sigma_c=sigma_c,
                mu_alpha=mu_alpha,
                mu_c=mu_c,
            )

            pred = predict(c=c, alpha=alpha, q_theory=ds.q_theory, q_experiment=ds.q_experiment)

            metrics = {
                "iter": int(it),
                "train_F": float(f_train),
            }
            metrics.update(_metrics_dict("test_theory", ds.r_theory, pred.eta_theory, splits.test_theory))
            metrics.update(
                _metrics_dict(
                    "test_experiment",
                    ds.r_experiment,
                    pred.eta_experiment,
                    splits.test_experiment,
                )
            )
            history.append(metrics)

    elapsed = float(time.time() - t0)

    pred = predict(c=c, alpha=alpha, q_theory=ds.q_theory, q_experiment=ds.q_experiment)

    final_metrics = {
        "dataset": ds.name,
        "train_ratio": float(train_ratio),
        "val_ratio": float(val_ratio),
        "split_seed": int(split_seed),
        "training_seed": int(train_cfg.get("seed", 42)),
        "n_iters": int(n_iters),
        "elapsed_sec": elapsed,
    }

    final_metrics.update(_metrics_dict("test_theory", ds.r_theory, pred.eta_theory, splits.test_theory))
    final_metrics.update(
        _metrics_dict(
            "test_experiment",
            ds.r_experiment,
            pred.eta_experiment,
            splits.test_experiment,
        )
    )

    combined_true = np.concatenate([ds.r_theory, ds.r_experiment], axis=1)
    combined_pred = np.concatenate([pred.eta_theory, pred.eta_experiment], axis=1)
    combined_mask = np.concatenate([splits.test_theory, splits.test_experiment], axis=1)
    final_metrics.update(_metrics_dict("test_all", combined_true, combined_pred, combined_mask))

    result = {
        "final_metrics": final_metrics,
        "history": history,
        "c": c,
        "alpha": alpha,
        "pred": pred,
    }

    return result


def main() -> None:
    """
    主函数：解析命令行参数，加载配置，训练模型，保存结果
    
    命令行参数:
        --config: 配置文件路径 (默认: "configs/default.yaml")
        --dataset: 数据集名称 (可选，覆盖配置文件中的设置)
        --train_ratio: 训练集比例 (可选，覆盖配置文件中的设置)
    
    输出:
        无返回值，但会:
            - 创建输出目录并保存配置文件
            - 保存最终指标到 metrics.json
            - 保存训练历史到 history.json
            - 保存模型参数到 params.npz
            - 打印最终指标到控制台
    """
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--dataset", type=str, default=None)
    p.add_argument("--train_ratio", type=float, default=None)
    args = p.parse_args()

    cfg = load_yaml(args.config)

    if args.dataset is not None:
        cfg.setdefault("data", {})["dataset_name"] = args.dataset

    if args.train_ratio is not None:
        cfg.setdefault("split", {})["train_ratio"] = float(args.train_ratio)

    ds_name = str(cfg.get("data", {}).get("dataset_name"))

    run_dir = _make_run_dir(cfg, dataset_name=ds_name)

    result = train(cfg)

    with (run_dir / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(result["final_metrics"], f, indent=2)

    out_cfg = cfg.get("outputs", {})
    if bool(out_cfg.get("save_history", True)):
        with (run_dir / "history.json").open("w", encoding="utf-8") as f:
            json.dump(result["history"], f, indent=2)

    if bool(out_cfg.get("save_params", True)):
        np.savez_compressed(
            run_dir / "params.npz",
            c=result["c"],
            alpha=result["alpha"],
        )

    print(json.dumps(result["final_metrics"], indent=2))


if __name__ == "__main__":
    main()
