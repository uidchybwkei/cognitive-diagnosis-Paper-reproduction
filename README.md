# CDF-CSE（论文复现）

此仓库实现了并实验了在 `MD/paper.md` 中描述的 **CDF-CSE** 模型。

核心思想是联合建模：

- 学生的每个技能的**理论掌握程度**：$\alpha_{jk}$
- 学生的**整体编程能力**：$c_j$
- 学生的每个技能的**实验掌握程度**：$\beta_{jk} = c_j \alpha_{jk}$

然后预测学生在理论和实验问题上的分数。

## 快速开始

安装依赖：

```bash
pip install -r requirements.txt
```

训练（写入 `outputs/<dataset>/<timestamp>/`）：

```bash
python train.py --config configs/default.yaml --dataset dataStructure --train_ratio 0.8
```

评估保存的运行目录：

```bash
python eval.py --run_dir outputs/dataStructure/<timestamp> --split test
```

导出学生参数 + 完整预测矩阵：

```bash
python export_student_results.py --run_dir outputs/dataStructure/<timestamp>
```

导出命令打印输出目录路径并生成：

- `students.csv`
- `predictions.npz`

## 模型定义

### 输入

- $R \in \mathbb{R}^{M \times N_i}$: 理论分数
- $R' \in \mathbb{R}^{M \times N_e}$: 实验（编程）分数
- $Q \in \mathbb{R}^{N_i \times K}$: 理论问题-技能矩阵
- $Q' \in \mathbb{R}^{N_e \times K}$: 实验问题-技能矩阵

此实现将提供的 `q.csv` 文件解释为 **行 = 问题** 和 **列 = 技能**。

#### Q 归一化（Eq.3.1-1）

每个问题行通过其行和归一化（如果非零）：

- $q_{ik} \leftarrow q_{ik} / \sum_l q_{il}$
- $q'_{ek} \leftarrow q'_{ek} / \sum_l q'_{el}$

实现于：`src/data/dataset.py::_normalize_q_rows`。

### 潜在变量

- $c_j$: 整体编程能力（向量 $c \in \mathbb{R}^M$）
- $\alpha_{jk}$: 理论掌握（矩阵 $\alpha \in \mathbb{R}^{M \times K}$）
- $\beta_{jk}$: 实验掌握（未明确存储）

#### 理论与实验关系（Eq.3.2-1）

$\beta_{jk} = c_j \alpha_{jk}$

通过使用 `(c[:, None] * alpha)` 隐式实现。

### 预测

#### 理论预测性能（Eq.3.2-2）

对于学生 `j` 和理论问题 `i`：

$\eta_{ji} = \sum_k \alpha_{jk} q_{ik}$

矩阵形式（`Q` 存储为 `(N_i×K)`）：

$\eta = \alpha Q^T$

#### 实验预测性能（Eq.3.2-3）

对于学生 `j` 和实验问题 `e`：

$\eta'_{je} = \sum_k \beta_{jk} q'_{ek} = \sum_k (c_j \alpha_{jk}) q'_{ek}$

矩阵形式：

$\eta' = (c \odot \alpha) Q'^T$

实现于：`src/models/cdf_cse.py::predict`。

### 似然（Eq.3.2-4）

论文使用具有**精度**参数的高斯似然：

- $R_{ji} \sim \mathcal{N}(\eta_{ji}, \sigma_R^{-1} I)$
- $R'_{je} \sim \mathcal{N}(\eta'_{je}, \sigma_{R'}^{-1} I)$

在代码中，`sigma_r` 和 `sigma_rp` 被视为精度（乘以平方误差）。

### 先验（Eq.3.2-5）

高斯先验（也具有精度参数）：

- $c_j \sim \mathcal{N}(\mu_c, \sigma_c^{-1} I)$
- $\alpha_{jk} \sim \mathcal{N}(\mu_\alpha, \sigma_\alpha^{-1} I)$

在 `configs/default.yaml` 的 `model:` 下配置。

### 目标：负对数后验（Eq.3.3-5）

训练目标 $F(c, \alpha)$ 是以下的总和：

- 理论分数上的平方重建误差（由 $\sigma_R$ 加权）
- 实验分数上的平方重建误差（由 $\sigma_{R'}$ 加权）
- $\alpha$ 上的 L2 先验惩罚（由 $\sigma_\alpha$ 加权）
- $c$ 上的 L2 先验惩罚（由 $\sigma_c$ 加权）

实现于：`src/models/cdf_cse.py::objective_F`。

### 优化（Eq.3.3-6 ~ Eq.3.3-9）

训练使用简单的交替梯度下降：

1. 使用 `update_c` 更新 $c$ （梯度 $\text{grad}_c$）
2. 使用 `update_alpha` 更新 $\alpha$ （梯度 $\text{grad}_\alpha$）

实现于：

- `src/models/cdf_cse.py::update_c`, `grad_c`
- `src/models/cdf_cse.py::update_alpha`, `grad_alpha`

## 项目结构

- `train.py`

  - 运行训练并在 `outputs/` 下写入带时间戳的运行目录。
  - 保存：
    - `config.yaml` （解析的配置）
    - `metrics.json` （最终测试指标）
    - `history.json` （可选的每迭代指标）
    - `params.npz` （学习的 $c$ 和 $\alpha$）
- `eval.py`

  - 加载 `params.npz` 并在训练/验证/测试分割上评估 MAE/RMSE。
  - 支持从 CLI 覆盖数据集/分割参数。
- `export_student_results.py`

  - 加载 `params.npz` + 数据集配置，通过 `predict()` 计算预测，并导出：
    - `students.csv` （每学生一行：$c$，$\alpha$ 向量，和摘要列）
    - `predictions.npz` （完整预测矩阵：$\eta_{\text{theory}}$，$\eta_{\text{experiment}}$，$\hat{r}_{\text{all}}$，加上 $c$/$\alpha$）
- `configs/default.yaml`

  - 数据集加载、分割、模型超参数和训练超参数的默认配置。
- `src/models/cdf_cse.py`

  - 核心模型实现：
    - `predict`
    - `objective_F`
    - 梯度和更新规则
- `src/data/dataset.py`

  - 数据集加载和预处理：
    - 分数归一化到 `[0,1]`
    - 缺失值掩码
    - Q-行归一化
    - 训练/验证/测试条目级分割
- `src/utils/`

  - 小工具（YAML 加载、指标）。
- `docs/ASSUMPTIONS.md`

  - 记录由于论文省略细节或提供的数据集不一致而做出的选择。

## 实验结果

### Data Structure数据集实验结果

在dataStructure数据集上使用80%训练比例的实验结果：

```json
{
  "dataset": "dataStructure",
  "train_ratio": 0.8,
  "val_ratio": 0.0,
  "split_seed": 42,
  "training_seed": 42,
  "n_iters": 1000,
  "elapsed_sec": 0.07590198516845703,
  "test_theory_mae": 0.11133681194155481,
  "test_theory_rmse": 0.17458728386170572,
  "test_theory_n": 1086.0,
  "test_experiment_mae": 0.16277593069657567,
  "test_experiment_rmse": 0.23121613331021126,
  "test_experiment_n": 189.0,
  "test_all_mae": 0.11896190483935791,
  "test_all_rmse": 0.18408474548099737,
  "test_all_n": 1275.0
}
```

**关键指标总结：**


| 指标           | 理论部分 | 实验部分 | 全部   |
| -------------- | -------- | -------- | ------ |
| **MAE**        | 0.1113   | 0.1628   | 0.1190 |
| **RMSE**       | 0.1746   | 0.2312   | 0.1841 |
| **测试样本数** | 1086     | 189      | 1275   |

### 模型性能对比

下图展示了不同模型在dataStructure数据集上的性能对比，包括CDF-CSE、FuzzyCDF、IRT、PMF-5D、PMF-10D和PMF-KD等模型在理论、实验和混合数据上的MAE和RMSE表现：

![1768365182098](images/README/1768365182098.png)

**观察结果：**

- 实验结果基本符合论文中的描述
- CDF-CSE模型在理论和实验部分都表现出较好的预测性能
- 实验部分的预测误差普遍高于理论部分，这与论文中的观察一致
- 随着测试比例的变化，各模型的性能趋势与论文报告相符
