# 中文逐行讲解：论文对照 CDF-CSE（项目复现说明）

本文件将用**全中文**、尽可能“逐行/逐段”的方式，解释本项目如何复现论文（`MD/paper.md` / PDF）中的 **CDF-CSE** 模型，并对照论文的章节与公式。

- 覆盖文件：
  - `src/data/dataset.py`
  - `src/models/cdf_cse.py`
  - `src/utils/metrics.py`
  - `train.py` / `eval.py`
  - `configs/default.yaml`
  - `docs/ASSUMPTIONS.md`
  - `tests/test_dataset.py` / `tests/test_cdf_cse.py`

> 说明：论文在 `paper.md` 中未提供统一的“(1)(2)(3)”公式编号，因此本项目在 `docs/paper_map.md` 中给关键公式做了自定义编号（如 `Eq.3.3-5`）。本文也沿用该编号，并在需要时同时注明“论文第几节”。

---

## 目录（建议先读）

- **1. 论文 ↔ 代码：总览映射**
- **2. 全局符号、shape 与 axis 约定（全文通用）**
- **3. 数据集加载与预处理：`src/data/dataset.py`**
- **4. 模型实现（CDF-CSE）：`src/models/cdf_cse.py`**
- **5. 指标实现：`src/utils/metrics.py`**
- **6. 训练脚本：`train.py`**
- **7. 评估脚本：`eval.py`**
- **8. 配置文件：`configs/default.yaml`**
- **9. 重要假设与论文缺失细节补全：`docs/ASSUMPTIONS.md`**
- **10. 单元测试在验证什么：`tests/*.py`**
- **11. 读完本文你应该掌握什么？**

---

## 1. 论文 ↔ 代码：总览映射

本项目实现的模型对应论文：

- `MD/paper.md`：第 3 章（CDF-CSE 建模与优化）
- 关键公式清单：`docs/paper_map.md`（本项目整理的公式编号）

### 1.1 论文核心变量

论文第 3.1 节（Problem definition）给出数据与符号，本项目的变量命名与论文几乎同构：

- **学生整体编程能力**：\(c_j\)

  - 代码变量：`c`，shape `(M,)`
- **学生技能理论能力**：\(\alpha_{jk}\)

  - 代码变量：`alpha`，shape `(M,K)`
- **学生技能实验能力**：\(\beta_{jk}\)

  - 论文公式：`Eq.3.2-1`，\(\beta_{jk}=c_j\alpha_{jk}\)
  - 代码实现：`(c[:, None] * alpha)`（见 `src/models/cdf_cse.py: predict`）
- **理论题预测表现/掌握度**：\(\eta_{ji}\)

  - 论文公式：`Eq.3.2-2`
  - 代码变量：`eta_theory`，shape `(M,N_i)`
- **实验题预测表现/掌握度**：\(\eta'_{je}\)

  - 论文公式：`Eq.3.2-3`
  - 代码变量：`eta_experiment`，shape `(M,N_e)`

### 1.2 论文的训练目标（MAP）对应代码的 `objective_F`

论文第 3.3 节定义负对数后验（省略常数项）：

- `Eq.3.3-5`：\(F(c,\alpha)\)

代码中对应：

- `src/models/cdf_cse.py: objective_F(...)`

并且本项目为了处理缺失值与训练/测试划分，会用 `mask` 把求和范围限制在“当前集合的观测条目”上（详见第 3 节和第 4 节）。

### 1.3 论文的交替优化（alternating optimization）对应代码的训练循环

论文第 3.3 节提出交替优化（重复两步直到收敛）：

- Step 1：固定 \(\alpha\)，更新 \(c\)（`Eq.3.3-6 ~ Eq.3.3-7`）
- Step 2：固定 \(c\)，更新 \(\alpha\)（`Eq.3.3-8 ~ Eq.3.3-9`）

代码中对应：

- `train.py` 的主循环（每个迭代 `it`）
  - 先调用 `update_c(...)`
  - 再调用 `update_alpha(...)`

其中：

- `update_c = c - r1 * grad_c`
- `update_alpha = alpha - r2 * grad_alpha`

这里的 `r1/r2` 即论文的 step length（步长）。

---

## 2. 全局符号、shape 与 axis 约定（全文通用）

### 2.1 维度符号（paper 与代码的对应关系）

论文采用：\(M, N_i, N_e, K\)。你在提需求时写的 `N/I/E/K` 可以这样理解：

- \(N\) 就是 \(M\)：学生数
- \(I\) ≈ \(N_i\)：理论题数量
- \(E\) ≈ \(N_e\)：实验题数量
- \(K\)：技能/知识点数量

后文默认用论文记号 \(M, N_i, N_e, K\)

### 2.2 数据矩阵（R/R′/Q/Q′）在本项目中的 shape

本项目把原始 CSV 分成“理论部分”和“实验部分”，对应论文第 3.1 节：

- `r_theory ∈ R^{M×N_i}`：理论题得分矩阵（论文：\(R\)）
- `r_experiment ∈ R^{M×N_e}`：实验题得分矩阵（论文：\(R'\)）
- `q_theory ∈ R^{N_i×K}`：理论题 Q 矩阵（论文：\(Q\)）
- `q_experiment ∈ R^{N_e×K}`：实验题 Q′ 矩阵（论文：\(Q'\)）

### 2.3 mask（掩码）在本项目中的含义

数据里存在缺失分数（默认 `-1`）。本项目用布尔矩阵 `mask` 表示“该位置是否观测到”：

- `mask_theory ∈ {False,True}^{M×N_i}`
- `mask_experiment ∈ {False,True}^{M×N_e}`

另外，训练/测试划分并不是“按学生/按题目划分”，而是**按观测到的条目（entry-level）随机划分**：

- 训练集 mask：`splits.train_*`
- 测试集 mask：`splits.test_*`

这对应 `docs/ASSUMPTIONS.md` 的“训练/测试划分协议”。

### 2.4 模型参数与预测输出的 shape

- `c ∈ R^{M}`：每个学生一个标量（整体编程能力）
- `alpha ∈ R^{M×K}`：每个学生对每个技能的理论能力

预测输出（由 `predict(...)` 产生）：

- `eta_theory = alpha @ q_theory.T ∈ R^{M×N_i}`
- `eta_experiment = (c[:,None] * alpha) @ q_experiment.T ∈ R^{M×N_e}`

### 2.5 axis 方向（你看代码时要随时对照）

对不同类型矩阵，axis 的含义固定如下：

- 对 `R` 类矩阵（`r_*`、`eta_*`、`mask_*`）

  - `axis=0`：学生维度 \(M\)
  - `axis=1`：题目维度（\(N_i\) 或 \(N_e\)）
- 对 `Q` 类矩阵（`q_*`）

  - `axis=0`：题目维度（\(N_i\) 或 \(N_e\)）
  - `axis=1`：技能维度 \(K\)
- 对 `alpha (M,K)`

  - `axis=0`：学生
  - `axis=1`：技能

---

## 3. 数据集加载与预处理：`src/data/dataset.py`

本节对应论文：

- 第 3.1 节（Problem definition：R/R′/Q/Q′ 的定义）
- `Eq.3.1-1`（Q 矩阵按行归一化）

本节会按**代码出现顺序**逐段解释该文件中的主要函数/类，并在每处给出：

- **输入**（参数 + shape）
- **输出**（返回值 + shape）
- **内部步骤**（尽量逐行解释关键语句）
- **与论文公式对应**（章节 + `Eq.*` 编号）
- **与 `docs/ASSUMPTIONS.md` 的关系**（哪些地方是论文未说明的补全）

---

### 3.1 这个文件解决的“问题”是什么？

论文第 3.1 节假设我们已知：

- 理论题得分矩阵 \(R\)
- 实验题得分矩阵 \(R'\)
- 理论题 Q 矩阵 \(Q\)
- 实验题 Q 矩阵 \(Q'\)

但真实数据文件（`dataset/*/*.csv`）并不直接以四个矩阵分别存储，而是：

- **`R.csv`**：把理论题与实验题的列拼在一起存（总列数 \(N=N_i+N_e\)）
- **`q.csv`**：把理论题与实验题的行拼在一起存（总行数 \(N=N_i+N_e\)）

因此 `dataset.py` 的职责就是：

- **读入 CSV**
- **把 R/q 拆回论文定义的 R/R′ 与 Q/Q′**
- **把缺失值（-1）变成 mask（掩码）**
- **把原始分数 [0,10] 归一化到 [0,1]**（论文期望 [0,1]）
- **对 Q 做按行归一化**（论文 `Eq.3.1-1`）
- **生成训练/验证/测试的 entry-level mask**（论文未写明的细节，本项目给出一套明确实现）

---

### 3.2 `CDFCSEDataset`：数据结构（把 R/R′/Q/Q′ + mask 打包）

代码位置：`src/data/dataset.py: 11-48`

#### 3.2.1 字段（Fields）逐个解释

`@dataclass(frozen=True)` 的含义：

- `dataclass`：自动生成 `__init__` 等方法，让这个类更像“结构体”
- `frozen=True`：实例创建后不可修改（immutable），避免训练过程中被误改

字段列表：

- `name: str`

  - 数据集名（如 `dataStructure` / `networkSecurity`）
- `r_theory: np.ndarray`

  - shape `(M, N_i)`
  - 理论题得分矩阵（已归一化到 [0,1]，缺失位置填 0，但会被 mask 排除）
- `r_experiment: np.ndarray`

  - shape `(M, N_e)`
  - 实验题得分矩阵（同上）
- `q_theory: np.ndarray`

  - shape `(N_i, K)`
  - 理论题 Q 矩阵（默认按行归一化，每行和为 1）
- `q_experiment: np.ndarray`

  - shape `(N_e, K)`
  - 实验题 Q′ 矩阵（默认按行归一化，每行和为 1）
- `mask_theory: np.ndarray`

  - dtype `bool`，shape `(M, N_i)`
  - True 表示该学生-该理论题的分数是“观测到的”（不是 -1）
- `mask_experiment: np.ndarray`

  - dtype `bool`，shape `(M, N_e)`
- `problem_ids_theory: np.ndarray`

  - shape `(N_i,)`
  - 理论题 ID（来自 `q.csv` 的索引）
- `problem_ids_experiment: np.ndarray`

  - shape `(N_e,)`
- `skill_ids: np.ndarray`

  - shape `(K,)`
  - 技能 ID（来自 `q.csv` 的列名）

#### 3.2.2 便捷属性（Properties）

代码位置：`dataset.py: 24-38`

这些属性本质是“从矩阵 shape 推导维度”，避免写死：

- `n_students = r_theory.shape[0] = M`
- `n_theory = r_theory.shape[1] = N_i`
- `n_experiment = r_experiment.shape[1] = N_e`
- `n_skills = q_theory.shape[1] = K`

为什么 `n_skills` 从 `q_theory` 得到而不是 `q_experiment`？

- 两者都应当是 `K` 列；任选其一即可。
- 如果数据异常导致两者 K 不一致，后续模型也会因为 shape check 报错。

#### 3.2.3 三个“拼接函数”：`combined_*`

代码位置：`dataset.py: 40-47`

- `combined_r()`：`np.concatenate([r_theory, r_experiment], axis=1)`

  - 输出 shape `(M, N_i+N_e)`
  - 含义：把理论题与实验题当成“同一种题”（论文第 4.3 节情形①）
- `combined_mask()`：同样在 `axis=1` 拼接
- `combined_q()`：`np.concatenate([q_theory, q_experiment], axis=0)`

  - 输出 shape `(N_i+N_e, K)`
  - 含义：把“题目维度”拼回总题目

---

### 3.3 `SplitMasks`：训练/验证/测试的 6 个 mask

代码位置：`dataset.py: 50-58`

这个 `dataclass` 只做一件事：保存 `make_splits(...)` 的输出。

- `train_theory: (M,N_i)`
- `train_experiment: (M,N_e)`
- `val_theory: (M,N_i)`
- `val_experiment: (M,N_e)`
- `test_theory: (M,N_i)`
- `test_experiment: (M,N_e)`

注意：这些 mask 的 True 总数代表“该集合里有多少条观测分数（entry）”。

---

### 3.4 `_normalize_q_rows(q)`：实现论文 `Eq.3.1-1`（Q 按行归一化）

代码位置：`dataset.py: 60-65`

#### 3.4.1 输入 / 输出

- 输入：`q`，shape `(N, K)`（这里的 `N` 可以是 `N_i` 或 `N_e` 或总题数）
- 输出：`out`，shape `(N, K)`

#### 3.4.2 逐行解释

核心代码：

```python
row_sum = q.sum(axis=1, keepdims=True)
out = np.zeros_like(q, dtype=float)
nonzero = row_sum.squeeze(axis=1) > 0
out[nonzero] = q[nonzero] / row_sum[nonzero]
return out
```

- `row_sum = q.sum(axis=1, keepdims=True)`

  - 对每一道题（每一行）求和：\(\sum_l q_{il}\)
  - `keepdims=True` 让 `row_sum` 维持 `(N,1)`，用于广播除法
- `out = np.zeros_like(q, dtype=float)`

  - 输出先初始化为全 0
- `nonzero = row_sum.squeeze(axis=1) > 0`

  - 找出行和大于 0 的题目
- `out[nonzero] = q[nonzero] / row_sum[nonzero]`

  - 对这些题做归一化：每行除以该行的和
  - 对应论文 `Eq.3.1-1`：
    - \(q_{ik} \leftarrow q_{ik} / \sum_l q_{il}\)

对“全零行”会发生什么？

- `nonzero` 为 False，所以该行保持全 0
- 这会让该题在模型里“完全不依赖任何技能”，通常不符合论文语义
- 因此更推荐在归一化前，用 `_ensure_expected_k(..., q_zero_row_strategy="unknown_skill")` 修复全零行（见 3.5）

---

### 3.5 `_ensure_expected_k(...)`：处理技能列数不一致与 Q 全零行（论文缺失细节补全）

代码位置：`dataset.py: 68-97`

该函数对应 `docs/ASSUMPTIONS.md` 中两条非常关键的“数据集异常处理”假设：

- **NetworkSecurity 数据集技能数不一致**：论文 Table 1 说 `K=7`，但提供的 `q.csv` 只有 6 列
- **NetworkSecurity 存在 Q 全零行**：部分题目一行全 0，论文未说明

#### 3.5.1 输入 / 输出

- 输入：
  - `q: pd.DataFrame`，shape `(N, K_raw)`
  - `expected_k: int`（期望技能数，来自 `configs/default.yaml` 的 `expected_k`）
  - `q_zero_row_strategy: "unknown_skill" | "error"`
- 输出：`q: pd.DataFrame`，shape `(N, expected_k)`

#### 3.5.2 分两步理解这段代码

**第 1 步：保证列数 = expected_k**

- 如果 `q.shape[1] > expected_k`：直接报错

  - 这表示数据文件包含“比配置更多的技能列”，属于严重不一致
- 如果 `q.shape[1] < expected_k`：追加全 0 列直到满足

```python
for col_idx in range(q.shape[1], expected_k):
    q[str(col_idx)] = 0
```

注意这里列名用的是字符串 `"0"`, `"1"` ...：

- 因为 CSV 读出来后列名本来就是字符串

**第 2 步：处理全零行（row_sum==0）**

```python
row_sum = q.sum(axis=1)
zero_rows = row_sum == 0
...
unknown_col = str(expected_k - 1)
q.loc[zero_rows, unknown_col] = 1
```

- `zero_rows` 是一个布尔向量（长度 N），标记哪些题目“没有任何技能”
- 当策略是 `unknown_skill` 时：把这些题目的“最后一列技能”置为 1

这等价于：

- 给这些题目引入一个“未知技能”，使其至少依赖一个技能
- 后续再做 `_normalize_q_rows` 时，每行和为 1，不会出现除 0

---

### 3.6 `load_real_dataset(...)`：从 CSV 生成 (R/R′/Q/Q′/mask)

代码位置：`dataset.py: 100-164`

#### 3.6.1 输入（Inputs）

- 路径与数据集信息：

  - `dataset_root`（默认 `dataset`）
  - `dataset_name`（如 `dataStructure`）
- 数据集规格（来自配置 `dataset_specs`，对应论文 Table 1）：

  - `n_theory = N_i`
  - `n_experiment = N_e`
  - `expected_k = K`
- 数据处理超参数：

  - `score_scale`：分数缩放（默认 10.0）
  - `missing_value`：缺失值标记（默认 -1.0）
  - `q_normalize`：是否对 Q 行归一化（默认 True，对应 `Eq.3.1-1`）
  - `q_zero_row_strategy`：全零行策略（默认 `unknown_skill`）

#### 3.6.2 输出（Outputs）

- 返回 `CDFCSEDataset`，其中 `r_*` 与 `q_*` 都已按理论/实验切分。

#### 3.6.3 关键步骤逐段解释（尽量贴近逐行）

**(1) 拼出 CSV 路径**

```python
ds_dir = Path(dataset_root) / dataset_name
r_path = ds_dir / "R.csv"
q_path = ds_dir / "q.csv"
```

- 这体现了数据目录约定：每个数据集有自己的文件夹，内部固定 `R.csv` 与 `q.csv`。

**(2) 读 CSV**

```python
r_df = pd.read_csv(r_path, index_col=0)
q_df = pd.read_csv(q_path, index_col=0)
```

- `index_col=0`：把第一列当 index
  - `R.csv`：通常 index 是学生 ID
  - `q.csv`：通常 index 是题目 ID

**(3) 校验题目数量一致性**

```python
n_total = n_theory + n_experiment
if r_df.shape[1] != n_total: ...
if q_df.shape[0] != n_total: ...
```

- `R.csv`：列数必须等于总题数
- `q.csv`：行数必须等于总题数

这一步在工程上非常重要：

- 如果 `R` 与 `Q` 的题目数量对不上，后面所有矩阵乘法都会错位。

**(4) 处理技能列数与全零行**

```python
q_df = _ensure_expected_k(q_df, expected_k=expected_k, q_zero_row_strategy=q_zero_row_strategy)
```

- 这是“论文缺失细节补全”的核心落点之一（对应 `docs/ASSUMPTIONS.md`）。

**(5) 保存题目/技能 ID（用于可解释性）**

```python
problem_ids = np.asarray(q_df.index)
skill_ids = np.asarray(q_df.columns)
```

- 这些 ID 在训练过程中不参与任何计算，但对你之后理解某个技能/题目的含义非常有价值。

**(6) 从 R 构造 mask，并做分数归一化**

```python
r = r_df.to_numpy(dtype=float)
mask = r != float(missing_value)
r_norm = np.zeros_like(r, dtype=float)
r_norm[mask] = r[mask] / float(score_scale)
```

逐行解释：

- `r` shape `(M, N_i+N_e)`
- `mask` shape `(M, N_i+N_e)`，True 表示不是缺失值
- `r_norm`：先全 0
- 只对观测到的位置做归一化：
  - 数据文件范围 `[0,10]`
  - 论文期望 `[0,1]`
  - 因此除以 `score_scale=10.0`

这对应 `docs/ASSUMPTIONS.md` 的“数据归一化”条目。

**(7) Q 转 numpy，并可选做 `Eq.3.1-1` 行归一化**

```python
q = q_df.to_numpy(dtype=float)
if q_normalize:
    q = _normalize_q_rows(q)
```

- 如果归一化开启：每题的技能权重和为 1

**(8) 按 n_theory 切分理论/实验**

```python
r_theory = r_norm[:, :n_theory]
r_experiment = r_norm[:, n_theory:n_total]
mask_theory = mask[:, :n_theory]
mask_experiment = mask[:, n_theory:n_total]

q_theory = q[:n_theory, :]
q_experiment = q[n_theory:n_total, :]
```

这段切分隐含了一个重要约定：

- `R.csv` 的前 `N_i` 列是理论题，后 `N_e` 列是实验题
- `q.csv` 的前 `N_i` 行是理论题，后 `N_e` 行是实验题

它本质上是“文件层面的 R/Q 与论文符号 R/R′、Q/Q′ 的转换桥梁”。

---

### 3.7 `_split_observed_entries(...)`：entry-level 随机划分（只在观测条目上切）

代码位置：`dataset.py: 167-201`

#### 3.7.1 输入 / 输出

- 输入：

  - `observed_mask: (M,N)`：True 表示观测到
  - `train_ratio`：训练占比（必须在 `(0,1)`）
  - `val_ratio`：验证占比（默认 0.0）
  - `seed`：随机种子
- 输出：`train_mask, val_mask, test_mask`，shape 均为 `(M,N)`

#### 3.7.2 为什么要“按条目”划分？

论文第 4.3 节描述的是“训练数据稀疏度”（training data ratio 从 80% 到 20%），更像是：

- 在一个学生-题目的评分矩阵里，随机保留一部分观测条目用于训练，其余用于测试

所以本项目的划分策略是：

- 先拿到所有观测到的位置 `obs_idx = argwhere(mask)`
- 随机打乱
- 前 `train_ratio` 的位置属于训练
- 剩余属于测试（以及可选验证）

#### 3.7.3 逐段解释

```python
obs_idx = np.argwhere(observed_mask)
n_obs = int(obs_idx.shape[0])
rng = np.random.default_rng(int(seed))
rng.shuffle(obs_idx)
```

- `obs_idx` shape `(n_obs, 2)`：每一行是一个 `(student_index, problem_index)`
- `shuffle`：打乱顺序，实现随机划分

```python
n_train = int(np.floor(train_ratio * n_obs))
n_val = int(np.floor(val_ratio * n_obs))
```

- 这里用 `floor`：保证是整数，并且 `train+val+test` 不会超过 `n_obs`

```python
train_idx = obs_idx[:n_train]
val_idx = obs_idx[n_train : n_train + n_val]
test_idx = obs_idx[n_train + n_val :]
```

```python
train_mask = np.zeros_like(observed_mask, dtype=bool)
...
train_mask[train_idx[:, 0], train_idx[:, 1]] = True
```

- 把索引列表“散射”回二维 mask

---

### 3.8 `make_splits(...)`：两种 split_mode（combined / per_matrix）

代码位置：`dataset.py: 203-249`

#### 3.8.1 输入 / 输出

- 输入：
  - `dataset: CDFCSEDataset`
  - `train_ratio` / `val_ratio` / `seed`
  - `split_mode: "combined" | "per_matrix"`
- 输出：`SplitMasks`

#### 3.8.2 split_mode = `combined`（默认，推荐）

核心逻辑：

```python
combined_obs = dataset.combined_mask()
train_all, val_all, test_all = _split_observed_entries(combined_obs, ...)

train_theory = train_all[:, :dataset.n_theory]
train_experiment = train_all[:, dataset.n_theory:]
...
```

解释：

- 把理论题与实验题的观测条目视为同一池子，在总矩阵上做一次随机划分。
- 然后再切回理论/实验。

这与论文第 4.3 节的设置①更贴近：

- “Treat both kinds of questions as the same kind of question”

#### 3.8.3 split_mode = `per_matrix`

核心逻辑：

- 理论题的 mask 单独划分一次
- 实验题的 mask 单独划分一次（并使用 `seed+1` 避免完全相同的随机序列）

```python
train_experiment, val_experiment, test_experiment = _split_observed_entries(
    dataset.mask_experiment,
    train_ratio=train_ratio,
    val_ratio=val_ratio,
    seed=seed + 1,
)
```

适用场景：

- 你希望“理论题的训练比例”和“实验题的训练比例”分别独立控制时。

---

### 3.9 `load_dataset_from_config(cfg)`：把 YAML 配置翻译成一次 `load_real_dataset` 调用

代码位置：`dataset.py: 252-270`

它做的事情非常直接：

1. 从 `cfg["data"]` 里拿到 `dataset_name`
2. 从 `cfg["data"]["dataset_specs"][dataset_name]` 里拿到 `n_theory/n_experiment/expected_k`
3. 再把 `score_scale/missing_value/q_normalize/q_zero_row_strategy` 等参数传给 `load_real_dataset`

这个函数的存在让 `train.py/eval.py` 不需要手写数据集规格，保证配置是“单一事实来源”（single source of truth）。

---

## 4. 模型实现（CDF-CSE）：`src/models/cdf_cse.py`

本节对应论文：

> 建议你边看本文边打开 `src/models/cdf_cse.py`，因为这个文件基本就是把论文第 3.2-3.3 节“翻译”成 numpy 代码。

- 第 3.2 节（模型定义：\(\beta=c\alpha\)，\(\eta\) 与 \(\eta'\)）
- 第 3.3 节（MAP 目标函数 \(F\) 与梯度下降 + 交替优化）

接下来会逐函数解释：`predict`、`objective_F`、`grad_c/grad_alpha`、`update_*` 以及它们如何组成 `train.py` 的交替优化过程。

---

### 4.0 文件整体结构（先建立心智模型）

你可以把 `cdf_cse.py` 理解成 4 层：

- **(A) 输出结构**：`CDFCSEResult`
- **(B) shape 安全检查**：`_check_shapes`
- **(C) 模型数学本体**：
  - `predict`：根据 \(c,\alpha,Q,Q'\) 计算 \(\eta,\eta'\)
  - `objective_F`：计算 \(F(c,\alpha)\)
  - `grad_c` / `grad_alpha`：计算梯度
- **(D) 优化一步封装**：`update_c` / `update_alpha`

`train.py` 做的事情，就是不断调用 (D)，并在需要时用 (C) 计算日志与指标。

---

### 4.1 `CDFCSEResult`：预测输出的“结构体”

代码位置：`src/models/cdf_cse.py: 9-13`

```python
@dataclass(frozen=True)
class CDFCSEResult:
    eta_theory: np.ndarray
    eta_experiment: np.ndarray
```

逐行解释：

- `eta_theory`

  - shape `(M, N_i)`
  - 含义：\(\eta\)，理论题的预测分数/掌握度（论文 `Eq.3.2-2`）
- `eta_experiment`

  - shape `(M, N_e)`
  - 含义：\(\eta'\)，实验题的预测分数/掌握度（论文 `Eq.3.2-3`）

为什么要用 dataclass 而不是直接返回 tuple？

- 命名字段能显著降低“把 (theory, experiment) 顺序写反”的风险。
- `frozen=True` 保证结果对象不可变，有利于调试（不会被下游不小心改掉）。

---

### 4.2 `_check_shapes(...)`：把 shape 错误在第一时间炸出来（非常推荐你读懂）

代码位置：`cdf_cse.py: 15-73`

这个函数不参与任何数学计算，只做一件事：**严格检查每个输入矩阵的维度一致性**。

为什么很重要？

- 在 CDF-CSE 里，矩阵形状稍微错一点，numpy 可能会广播（broadcast）出“看似能算但其实错”的结果。
- `_check_shapes` 让错误尽早以异常形式暴露，而不是悄悄算错。

#### 4.2.1 输入 / 输出

- 输入：

  - `c: (M,)`
  - `alpha: (M,K)`
  - `q_theory: (N_i,K)`
  - `q_experiment: (N_e,K)`
  - `r_theory: (M,N_i)`
  - `r_experiment: (M,N_e)`
  - `mask_theory: (M,N_i)`
  - `mask_experiment: (M,N_e)`
- 输出：`(m, k, n_theory, n_experiment)`

  - 它其实就是把关键维度提取出来返回，供上层逻辑可能复用

#### 4.2.2 逐段解释（对应到代码块）

- `if c.ndim != 1` / `if alpha.ndim != 2`

  - 强制 `c` 必须是一维向量、`alpha` 必须是二维矩阵
  - 避免出现 `c` 是 `(M,1)` 或 `(1,M)` 等导致的广播歧义
- `m = alpha.shape[0]` / `k = alpha.shape[1]`

  - 把学生数与技能数从 `alpha` 推导出来
- `if c.shape[0] != m`

  - 强制 `c` 与 `alpha` 的学生维一致
- `n_theory = q_theory.shape[0]` / `n_experiment = q_experiment.shape[0]`

  - 把理论题数与实验题数从 Q 矩阵推导出来
- `if q_theory.shape[1] != k` / `if q_experiment.shape[1] != k`

  - 强制两个 Q 矩阵都必须是 `K` 列
- `if r_theory.shape != (m, n_theory)`

  - 强制理论题得分矩阵的 shape 完全匹配
- `if r_experiment.shape != (m, n_experiment)`

  - 同理
- mask shape 检查：

  - `mask_theory` 必须与 `r_theory` 同 shape
  - `mask_experiment` 必须与 `r_experiment` 同 shape

---

### 4.3 `predict(...)`：实现 \(\eta\) 与 \(\eta'\)（论文 `Eq.3.2-1/2/3`）

代码位置：`cdf_cse.py: 75-88`

核心两行：

```python
eta_theory = alpha @ q_theory.T
eta_experiment = (c[:, None] * alpha) @ q_experiment.T
```

#### 4.3.1 输入（Inputs）

- `c: (M,)`
- `alpha: (M,K)`
- `q_theory: (N_i,K)`
- `q_experiment: (N_e,K)`

#### 4.3.2 输出（Outputs）

- 返回 `CDFCSEResult`：
  - `eta_theory: (M,N_i)`
  - `eta_experiment: (M,N_e)`

#### 4.3.3 逐行解释 + 与论文公式对照

**(1) `eta_theory = alpha @ q_theory.T`**

- shape 推导：
  - `alpha (M,K)`
  - `q_theory.T (K,N_i)`
  - 乘积得到 `(M,N_i)`

对单个元素展开：

- `eta_theory[j,i] = sum_k alpha[j,k] * q_theory[i,k]`

对应论文：

- `Eq.3.2-2`：\(\eta_{ji}=\sum_k \alpha_{jk} q_{ik}\)

这里再次强调 `Q` 的方向约定（对应 `docs/ASSUMPTIONS.md`）：

- `q_theory` 的行是题目 `i`，列是技能 `k`
- 因此矩阵写法自然是 `alpha @ q_theory.T`

**(2) `eta_experiment = (c[:, None] * alpha) @ q_experiment.T`**

分两层看：

- `c[:, None]`：把 `c` 从 `(M,)` 扩成 `(M,1)`

  - 目的：与 `(M,K)` 的 `alpha` 广播相乘
- `(c[:, None] * alpha)`：得到 `(M,K)`

  - 对每个学生 j：把该学生的整行 `alpha[j,:]` 乘上标量 `c[j]`
  - 这就是论文 `Eq.3.2-1`：\(\beta_{jk}=c_j\alpha_{jk}\)

然后再乘：

- `@ q_experiment.T`（`(K,N_e)`）得到 `(M,N_e)`

对单个元素展开：

- `eta_experiment[j,e] = sum_k (c[j] * alpha[j,k]) * q_experiment[e,k]`

对应论文：

- `Eq.3.2-3`：\(\eta'_{je}=\sum_k \beta_{jk} q'_{ek}\)

---

### 4.4 `objective_F(...)`：负对数后验 \(F(c,\alpha)\)（论文 `Eq.3.3-5`）

代码位置：`cdf_cse.py: 90-133`

这一节会解释目标函数里每一项如何对应论文，并重点解释 mask（掩码）在目标函数中的作用。

#### 4.4.1 输入（Inputs）/ 输出（Outputs）

- 输入：

  - `c: (M,)`
  - `alpha: (M,K)`
  - `r_theory: (M,N_i)`
  - `r_experiment: (M,N_e)`
  - `q_theory: (N_i,K)`
  - `q_experiment: (N_e,K)`
  - `mask_theory: (M,N_i)`
  - `mask_experiment: (M,N_e)`
  - 超参数（标量）：`sigma_r, sigma_rp, sigma_alpha, sigma_c, mu_alpha, mu_c`
- 输出：

  - `float`：目标函数值 \(F(c,\alpha)\)

#### 4.4.2 这段代码在做什么？（先给一句“总解释”）

论文 `Eq.3.3-5` 的目标函数是一个“平方误差 + 高斯先验正则”的组合：

- 理论题：\(\frac{\sigma_R}{2}(R-\eta)^2\)
- 实验题：\(\frac{\sigma_{R'}}{2}(R'-\eta')^2\)
- \(\alpha\) 先验：\(\frac{\sigma_\alpha}{2}(\alpha-\mu_\alpha)^2\)
- \(c\) 先验：\(\frac{\sigma_c}{2}(c-\mu_c)^2\)

代码 `objective_F` 就是把这四项按矩阵形式计算出来并求和。

#### 4.4.3 逐段解释（按代码顺序）

**(1) 形状检查：`_check_shapes(...)`**

```python
_check_shapes(...)
```

- 这一步不产生数值结果，但能确保后面的矩阵乘法不会因为广播而“悄悄算错”。

**(2) 计算预测：`pred = predict(...)`**

```python
pred = predict(c=c, alpha=alpha, q_theory=q_theory, q_experiment=q_experiment)
```

- 得到：
  - `pred.eta_theory: (M,N_i)`
  - `pred.eta_experiment: (M,N_e)`

与论文对照：

- `pred.eta_theory` 对应 \(\eta\)（`Eq.3.2-2`）
- `pred.eta_experiment` 对应 \(\eta'\)（`Eq.3.2-3`）

**(3) 误差（residual）并应用 mask**

```python
diff_theory = (r_theory - pred.eta_theory) * mask_theory
diff_experiment = (r_experiment - pred.eta_experiment) * mask_experiment
```

关键点：为什么要乘 `mask_*`？

- 在 `dataset.py` 中，缺失值（-1）会被转换成 `mask=False`。
- 在 `make_splits(...)` 中，训练/测试也是通过 mask 区分。
- 因此在目标函数里乘 mask，相当于把求和范围限定在“当前集合的观测条目”上。

更形式化地说：

- 如果 `mask_theory[j,i] == False`，则 `diff_theory[j,i] = 0`，该项对 `sum(diff^2)` 没贡献。

**(4) 两个数据项损失：理论题 + 实验题**

```python
loss_theory = 0.5 * sigma_r * sum(diff_theory^2)
loss_experiment = 0.5 * sigma_rp * sum(diff_experiment^2)
```

对应论文 `Eq.3.3-5` 的前两项：

- \(\sum_{j,i} \frac{\sigma_R}{2}(R_{ji}-\eta_{ji})^2\)
- \(\sum_{j,e} \frac{\sigma_{R'}}{2}(R'_{je}-\eta'_{je})^2\)

注意：

- 论文从概率模型（`Eq.3.2-4` 的高斯）推导出这两个平方项。
- `sigma_r`/`sigma_rp` 在概率意义上是“精度（precision）”，越大表示噪声越小，误差惩罚越重。

**(5) 两个先验项：对 \(\alpha\) 与 \(c\) 的 L2 正则**

```python
prior_alpha = 0.5 * sigma_alpha * sum((alpha - mu_alpha)^2)
prior_c = 0.5 * sigma_c * sum((c - mu_c)^2)
```

与论文对照：

- 先验分布：`Eq.3.2-5`
  - \(c \sim \mathcal{N}(\mu_c, \sigma_c^{-1}I)\)
  - \(\alpha \sim \mathcal{N}(\mu_\alpha, \sigma_\alpha^{-1}I)\)

高斯先验的负对数（省略常数）就是一个平方项：\(\frac{\sigma}{2}(x-\mu)^2\)。

**(6) 返回总和**

```python
return loss_theory + loss_experiment + prior_alpha + prior_c
```

这正是论文 `Eq.3.3-5`（省略常数项）的实现。

---

### 4.5 `grad_c(...)`：对 \(c\) 的梯度 \(\nabla_c F\)（论文 `Eq.3.3-7`）

代码位置：`cdf_cse.py: 135-167`

#### 4.5.1 输入 / 输出

- 输入：

  - `c: (M,)`
  - `alpha: (M,K)`
  - `r_experiment: (M,N_e)`
  - `q_experiment: (N_e,K)`
  - `mask_experiment: (M,N_e)`
  - 超参数：`sigma_rp, sigma_c, mu_c`
- 输出：

  - `g: (M,)`，即每个学生一个梯度值 \(\partial F/\partial c_j\)

注意：为什么 `grad_c` 不需要 `r_theory/q_theory/mask_theory`？

- 因为理论题的预测 \(\eta\) 不含 \(c\)，只有实验题的 \(\eta'\) 才含 \(c\)。

#### 4.5.2 逐段解释（按代码顺序）

**(1) 计算 \(\sum_k \alpha_{jk} q'_{ek}\)**

```python
alpha_q = alpha @ q_experiment.T
```

- `alpha (M,K)` @ `q_experiment.T (K,N_e)` -> `alpha_q (M,N_e)`

对单个元素展开：

- `alpha_q[j,e] = sum_k alpha[j,k] * q_experiment[e,k]`

这就是论文 `Eq.3.3-7` 中括号里的那项：\(\sum_k \alpha_{jk} q'_{ke}\)（记号差异仅来自转置）。

**(2) 计算实验题预测 \(\eta'\)**

```python
eta_experiment = c[:, None] * alpha_q
```

- `c[:,None]` shape `(M,1)`
- 广播后得到 `(M,N_e)`

对单个元素：

- `eta_experiment[j,e] = c[j] * alpha_q[j,e]`

**(3) 残差（预测-真实）并应用 mask**

```python
err = (eta_experiment - r_experiment) * mask_experiment
```

- 这里用的是 `(预测-真实)`，而 `objective_F` 用的是 `(真实-预测)`。
- 两者在平方损失里等价，但在求导时，更自然的形式是 `(预测-真实)`。

**(4) 梯度汇总（对每个学生按实验题求和）**

```python
g = sigma_rp * np.sum(err * alpha_q, axis=1) + sigma_c * (c - mu_c)
```

解释：

- `err * alpha_q`：对应每道实验题对 \(c_j\) 的梯度贡献
- `sum(..., axis=1)`：对实验题维度 \(e\) 求和，得到每个学生的一个标量梯度
- `+ sigma_c * (c - mu_c)`：加上先验项的梯度

与论文 `Eq.3.3-7` 对照：

- 论文写的是 \(-\sigma_{R'}\sum_e (R'-\eta')(\cdots) + \sigma_c(c-\mu_c)\)
- 代码写的是 \(+\sigma_{R'}\sum_e (\eta'-R')(\cdots) + \sigma_c(c-\mu_c)\)
- 两者完全等价（只是把负号并入括号）。

---

### 4.6 `grad_alpha(...)`：对 \(\alpha\) 的梯度 \(\nabla_\alpha F\)（论文 `Eq.3.3-9`）

代码位置：`cdf_cse.py: 170-219`

#### 4.6.1 输入 / 输出

- 输入：

  - `c: (M,)`
  - `alpha: (M,K)`
  - `r_theory: (M,N_i)` / `r_experiment: (M,N_e)`
  - `q_theory: (N_i,K)` / `q_experiment: (N_e,K)`
  - `mask_theory: (M,N_i)` / `mask_experiment: (M,N_e)`
  - 超参数：`sigma_r, sigma_rp, sigma_alpha, mu_alpha`
- 输出：

  - `g: (M,K)`，即每个学生-技能一个梯度值 \(\partial F/\partial \alpha_{jk}\)

#### 4.6.2 梯度的“三部分结构”（先看数学）

论文 `Eq.3.3-9` 的结构是：

- 理论题误差项（与 \(\sigma_R\) 相关）
- 实验题误差项（与 \(\sigma_{R'}\) 相关，并多一个 \(c_j\)）
- 先验项（与 \(\sigma_\alpha\) 相关）

代码也严格按这三块分别计算：`g_theory + g_experiment + g_prior`。

#### 4.6.3 逐段解释（按代码顺序）

**(1) 理论题部分：`g_theory`**

```python
eta_theory = alpha @ q_theory.T
err_theory = (eta_theory - r_theory) * mask_theory
g_theory = sigma_r * (err_theory @ q_theory)
```

shape 推导：

- `err_theory (M,N_i)`
- `q_theory (N_i,K)`
- 乘积 `(M,N_i) @ (N_i,K) -> (M,K)`

对单个元素展开：

- `g_theory[j,k] = sigma_r * sum_i (eta_theory[j,i] - r_theory[j,i]) * q_theory[i,k]`

对应论文 `Eq.3.3-9` 第一项（符号差异同 4.5）。

**(2) 实验题部分：`g_experiment`**

```python
alpha_q = alpha @ q_experiment.T
eta_experiment = c[:, None] * alpha_q
err_experiment = (eta_experiment - r_experiment) * mask_experiment
g_experiment = sigma_rp * ((err_experiment @ q_experiment) * c[:, None])
```

关键点：最后为什么要乘 `c[:,None]`？

- 因为 \(\eta'_{je}=c_j\sum_k \alpha_{jk} q'_{ek}\)
- 对 \(\alpha_{jk}\) 求导会带出一个系数 \(c_j\)

shape 推导：

- `err_experiment (M,N_e)`
- `q_experiment (N_e,K)`
- `err_experiment @ q_experiment -> (M,K)`
- 再 `* c[:,None]` 保持 `(M,K)`

对单个元素展开：

- `g_experiment[j,k] = sigma_rp * c[j] * sum_e (eta_experiment[j,e] - r_experiment[j,e]) * q_experiment[e,k]`

这对应论文 `Eq.3.3-9` 第二项。

**(3) 先验部分：`g_prior`**

```python
g_prior = sigma_alpha * (alpha - mu_alpha)
```

对单个元素：

- `g_prior[j,k] = sigma_alpha * (alpha[j,k] - mu_alpha)`

对应论文 `Eq.3.3-9` 第三项。

**(4) 返回总梯度**

```python
return g_theory + g_experiment + g_prior
```

---

### 4.7 `update_c/update_alpha`：一步梯度下降（对应论文 Step 1/Step 2）

代码位置：

- `update_c`: `cdf_cse.py: 222-244`
- `update_alpha`: `cdf_cse.py: 246-275`

这两个函数非常“薄”，本质上就是：

- 先算梯度
- 再按 `new = old - lr * grad` 更新

#### 4.7.1 `update_c`（论文 `Eq.3.3-6`）

```python
g = grad_c(...)
return c - lr * g
```

- 论文 `Eq.3.3-6`：\(c^{new}=c^{old}-r_1 g(c)\)
- 代码里把 `r_1` 命名为 `lr`，在 `train.py` 中传入的就是配置里的 `r1`

#### 4.7.2 `update_alpha`（论文 Step 2 + `docs/ASSUMPTIONS.md`）

```python
g = grad_alpha(...)
return alpha - lr * g
```

论文在 Step 2（`Eq.3.3-8`）处存在一个常见排版/记号歧义：

- 论文写：\(\alpha^{new}=\alpha^{old}-r_2 \frac{\partial}{\partial\alpha} g(\alpha)\)
- 同一段又定义：\(g(\alpha)=\frac{\partial F}{\partial \alpha}\)

如果严格照字面理解，会变成“用二阶导/海森矩阵更新”，但论文后面的 `Eq.3.3-9` 给出的其实就是一阶梯度形式。

因此本项目采用 `docs/ASSUMPTIONS.md` 中说明的解释：

- Step 2 仍然是标准梯度下降：\(\alpha^{new}=\alpha^{old}-r_2 g(\alpha)\)

---

### 4.8 交替优化（Alternating Optimization）在代码里是如何体现的？

论文第 3.3 节强调：

- 固定 \(\alpha\) 时，各个学生的 \(c_j\) 相互独立，可并行优化
- 固定 \(c\) 时，各个学生的 \(\alpha_j\) 相互独立，也可并行优化

代码的实现方式是“向量化并行”（numpy 一次算完所有学生/技能），对应到 `train.py` 的每次迭代：

```text
for it in 1..n_iters:
  c     = c     - r1 * grad_c(c, alpha, R', Q', mask_train_experiment)
  alpha = alpha - r2 * grad_alpha(c, alpha, R, R', Q, Q', mask_train_theory, mask_train_experiment)
```

几点你需要特别注意：

- `update_c` **只使用实验题训练 mask**（因为 \(c\) 只影响实验题预测 \(\eta'\)）
- `update_alpha` 同时使用理论题与实验题的训练 mask（因为 \(\alpha\) 同时影响 \(\eta\) 与 \(\eta'\)）
- `train.py` 支持把 `c/alpha` 裁剪（clip）到 `[0,1]`
  - 这不是论文明确给出的步骤，但与论文“诊断目标值域为 [0,1]”的描述一致（详见第 9 节）

---

## 5. 指标实现：`src/utils/metrics.py`

本节对应论文：

- 第 4.3 节提到的 MAE/RMSE 指标（论文未给公式，项目在 `docs/ASSUMPTIONS.md` 补全为标准定义）

接下来会逐行解释 `masked_mae` 与 `masked_rmse`。

---

### 5.1 为什么这里必须是 masked（掩码）指标？

如果没有 mask，本项目会遇到两类“不能直接算 MAE/RMSE”的情况：

- **缺失值（missing value）**：原始 `R.csv` 中用 `-1` 表示缺失成绩。

  - 这部分条目在真实世界里“没有观测”，不能当成 0 分算进误差。
- **训练/测试划分**：本项目是按“观测条目（entry）”随机划分训练集与测试集。

  - 评估时必须只在 `test_mask==True` 的条目上计算误差。

因此 `masked_*` 的核心语义是：

- **只在 mask 为 True 的条目上计算平均误差**。

这也正对应 `docs/ASSUMPTIONS.md` 的两条：

- “缺失值处理”：缺失条目要排除
- “评价指标公式”：只在已观测（非缺失）的测试条目上计算

---

### 5.2 `masked_mae(y_true, y_pred, mask)`：掩码 MAE

代码位置：`src/utils/metrics.py: 6-18`

#### 5.2.1 输入 / 输出

- 输入：
  - `y_true: np.ndarray`
  - `y_pred: np.ndarray`
  - `mask: np.ndarray`

三者必须 **shape 完全一致**。在本项目中常见用法是：

- `y_true = ds.r_theory`，`y_pred = pred.eta_theory`，`mask = splits.test_theory`
- `y_true = ds.r_experiment`，`y_pred = pred.eta_experiment`，`mask = splits.test_experiment`
- 输出：

  - `float`：MAE（如果 mask 全 False，则返回 `nan`）

#### 5.2.2 与论文的对应

论文第 4.3 节说使用 MAE 衡量预测误差，但没给公式。本项目按标准定义补全：

- \(\mathrm{MAE}=\mathrm{mean}(|y-\hat{y}|)\)

并且仅在 `mask=True` 的测试条目上取 mean。

#### 5.2.3 逐行解释

```python
y_true = np.asarray(y_true, dtype=float)
y_pred = np.asarray(y_pred, dtype=float)
mask = np.asarray(mask, dtype=bool)
```

- 强制类型：
  - `y_true/y_pred` 用 float 计算
  - `mask` 用 bool，避免出现 0/1 浮点导致的“乘法当 mask”的歧义

```python
if y_true.shape != y_pred.shape or y_true.shape != mask.shape:
    raise ValueError(...)
```

- 形状校验：保证每个位置 `(j,i/e)` 都能一一对应

```python
if int(mask.sum()) == 0:
    return float("nan")
```

- 如果一个 split 没有任何条目（比如 `val_ratio=0` 时的验证集），返回 NaN。
- 这样比返回 0 更安全：0 会误导你以为模型“误差很小”。

```python
err = np.abs(y_true[mask] - y_pred[mask])
return float(np.mean(err))
```

- `y_true[mask]` 会把二维矩阵压平成一维向量，只保留 True 的位置。
- 误差向量 `err` 的长度等于 `mask.sum()`。
- 最后对误差向量求均值，就是 masked MAE。

---

### 5.3 `masked_rmse(y_true, y_pred, mask)`：掩码 RMSE

代码位置：`src/utils/metrics.py: 21-33`

#### 5.3.1 输入 / 输出

- 输入：同 `masked_mae`
- 输出：`float`（若 `mask.sum()==0`，返回 NaN）

#### 5.3.2 与论文的对应

同样，论文未给 RMSE 公式，本项目采用标准定义：

- \(\mathrm{RMSE}=\sqrt{\mathrm{mean}((y-\hat{y})^2)}\)

并只在 `mask=True` 的条目上取 mean。

#### 5.3.3 逐行解释

前半段类型转换与 shape 检查完全一致，这里只看核心计算：

```python
err2 = (y_true[mask] - y_pred[mask]) ** 2
return float(np.sqrt(np.mean(err2)))
```

- 先计算平方误差向量 `err2`
- 再取均值
- 最后开方得到 RMSE

---

### 5.4 指标在训练/评估中的调用方式：`_metrics_dict`

你会在两个文件里看到同名函数 `_metrics_dict`：

- `train.py: _metrics_dict(...)`
- `eval.py: _metrics_dict(...)`

它们都做同一件事：把 MAE/RMSE 以及样本数 `n`（mask 中 True 的个数）打包成一个 dict。

核心返回结构：

```python
{
  f"{prefix}_mae": masked_mae(...),
  f"{prefix}_rmse": masked_rmse(...),
  f"{prefix}_n": float(mask.sum()),
}
```

为什么要把 `n` 也打印出来？

- 因为这个项目的训练/测试划分是 entry-level，`train_ratio` 不同会导致测试条目数量变化。
- 比较不同实验（比如 0.2 vs 0.8 稀疏度）时，知道 `n` 能帮你判断统计稳定性。

---

## 6. 训练脚本：`train.py`

本节将把训练流程按“逐段 + shape + 论文对应”解释清楚。

---

### 6.1 `main()`：命令行入口与输出文件组织

代码位置：`train.py: 219-263`

#### 6.1.1 命令行参数（CLI args）

```text
--config      默认 configs/default.yaml
--dataset     可选，覆盖 data.dataset_name
--train_ratio 可选，覆盖 split.train_ratio
```

对应代码：

- `p.add_argument("--config", ...)`
- `p.add_argument("--dataset", ...)`
- `p.add_argument("--train_ratio", ...)`

#### 6.1.2 `load_yaml`：读取配置

```python
cfg = load_yaml(args.config)
```

- `load_yaml` 实现在 `src/utils/config.py`，内部用 `yaml.safe_load`。
- 如果 YAML 为空返回 `{}`。

#### 6.1.3 覆盖配置（dataset / train_ratio）

```python
if args.dataset is not None:
    cfg.setdefault("data", {})["dataset_name"] = args.dataset

if args.train_ratio is not None:
    cfg.setdefault("split", {})["train_ratio"] = float(args.train_ratio)
```

这段逻辑说明：

- 配置文件是默认值
- 命令行参数是“优先级更高的覆盖层”

#### 6.1.4 创建输出目录 `_make_run_dir`

```python
run_dir = _make_run_dir(cfg, dataset_name=ds_name)
```

`_make_run_dir` 的命名规则：

- 根目录：`outputs.root`（默认 `outputs/`）
- 子目录：`outputs/<dataset_name>/<YYYYmmdd_HHMMSS>/`

其中时间戳由 `time.strftime("%Y%m%d_%H%M%S")` 生成。

#### 6.1.5 运行训练并落盘

```python
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
```

输出文件含义：

- `config.yaml`
  - 本次 run 实际使用的配置（包含命令行覆盖后的结果）
- `metrics.json`
  - 最终测试指标（MAE/RMSE + n）
- `history.json`
  - 训练过程日志（每隔 `log_every` 或首/末迭代记录一次）
- `params.npz`
  - 保存 `c` 与 `alpha`，供 `eval.py` 直接加载评估

---

### 6.2 `_init_params(m, k, cfg)`：初始化 \(c\) 与 \(\alpha\)

代码位置：`train.py: 18-45`

#### 6.2.1 输入 / 输出

- 输入：

  - `m`：学生数 \(M\)
  - `k`：技能数 \(K\)
  - `cfg`：配置 dict
- 输出：

  - `c: (M,)`
  - `alpha: (M,K)`

#### 6.2.2 与论文公式的对应

论文 `Eq.3.2-5` 给出先验：

- \(c_j \sim \mathcal{N}(\mu_c, \sigma_c^{-1}I)\)
- \(\alpha_{jk} \sim \mathcal{N}(\mu_\alpha, \sigma_\alpha^{-1}I)\)

注意：这里的 \(\sigma\) 是“精度（precision）”，所以方差是 \(1/\sigma\)，标准差是 \(1/\sqrt{\sigma}\)。

代码正是按这个标准差进行采样：

```python
c = rng.normal(loc=mu_c, scale=1.0 / np.sqrt(sigma_c), size=(m,))
alpha = rng.normal(loc=mu_alpha, scale=1.0 / np.sqrt(sigma_alpha), size=(m, k))
```

#### 6.2.3 初始化策略（`init_method`）

支持两种：

- `prior_sample`（默认）

  - 从先验分布采样
  - 好处：随机初始化，避免所有学生完全相同；与 MAP 目标一致
- `mean`

  - 直接用先验均值初始化：
    - `c = mu_c`
    - `alpha = mu_alpha`
  - 好处：可复现实验、更稳定；坏处：初始时所有学生/技能完全一致，可能需要更多迭代才分化

#### 6.2.4 参数合法性检查

```python
if sigma_c <= 0 or sigma_alpha <= 0:
    raise ValueError(...)
```

- 因为标准差是 `1/sqrt(sigma)`，所以 `sigma` 必须 > 0。

---

### 6.3 `_make_run_dir(cfg, dataset_name)`：输出目录创建

代码位置：`train.py: 48-54`

```python
out_cfg = cfg.get("outputs", {})
root = Path(str(out_cfg.get("root", "outputs")))
run_id = time.strftime("%Y%m%d_%H%M%S")
run_dir = root / dataset_name / run_id
run_dir.mkdir(parents=True, exist_ok=False)
```

关键点：

- `exist_ok=False`：如果同名目录已存在会直接报错，避免覆盖旧实验。

---

### 6.4 `_metrics_dict(prefix, y_true, y_pred, mask)`：指标打包

代码位置：`train.py: 57-63`

这部分在第 5 节已经解释过，这里只强调它在训练日志里被用来生成：

- `test_theory_mae/rmse/n`
- `test_experiment_mae/rmse/n`

---

### 6.5 `train(cfg)`：训练主流程（交替优化的代码化实现）

代码位置：`train.py: 65-216`

你可以把它分成 6 个阶段理解。

#### 6.5.1 阶段 A：加载数据集

```python
ds = load_dataset_from_config(cfg)
```

得到 `ds`（见第 3 节）：

- `ds.r_theory: (M,N_i)`
- `ds.r_experiment: (M,N_e)`
- `ds.q_theory: (N_i,K)`
- `ds.q_experiment: (N_e,K)`
- `ds.mask_theory/mask_experiment`

#### 6.5.2 阶段 B：划分训练/验证/测试 mask

```python
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
```

得到 `splits`（见第 3 节）：

- `splits.train_theory/train_experiment`
- `splits.val_*`
- `splits.test_*`

这些 mask 决定：

- **训练时 objective 与梯度只在 train mask 的 True 条目上计算**
- **评估时 MAE/RMSE 只在 test mask 的 True 条目上计算**

#### 6.5.3 阶段 C：初始化参数与读取超参数

```python
m = ds.n_students
k = ds.n_skills
c, alpha = _init_params(m, k, cfg)
```

此时：

- `c: (M,)`
- `alpha: (M,K)`

然后读取论文里的各类 \(\sigma\)、\(\mu\) 超参数：

```python
sigma_r, sigma_rp, sigma_alpha, sigma_c, mu_alpha, mu_c
```

这些分别对应论文：

- `Eq.3.2-4`（观测噪声超参数）：`sigma_r`, `sigma_rp`
- `Eq.3.2-5`（先验超参数）：`mu_alpha`, `mu_c`, `sigma_alpha`, `sigma_c`

再读取优化超参数（论文未给出，见第 9 节）：

- `n_iters`
- `r1`（更新 `c` 的步长，论文 Step 1 的 \(r_1\)）
- `r2`（更新 `alpha` 的步长，论文 Step 2 的 \(r_2\)）

#### 6.5.4 阶段 D：交替优化主循环（核心）

代码结构：

```python
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
```

把它翻译成论文语言：

- Step 1：固定 \(\alpha\)，用实验题训练集更新 \(c\)
- Step 2：固定 \(c\)，用理论+实验训练集更新 \(\alpha\)

这正对应论文第 3.3 节的 alternating optimization。

特别注意这两行 mask：

- `update_c(... mask_experiment=splits.train_experiment ...)`

  - 表示 \(c\) 的梯度只来自实验题训练条目
- `update_alpha(... mask_theory=splits.train_theory, mask_experiment=splits.train_experiment ...)`

  - 表示 \(\alpha\) 的梯度来自：理论题训练条目 + 实验题训练条目

#### 6.5.5 阶段 E：日志记录（objective + test 指标）

日志触发条件：

```python
do_log = (it == 1) or (it == n_iters) or (log_every > 0 and it % log_every == 0)
```

当触发时，会做三件事：

**(1) 计算训练集上的目标函数值 `train_F`**

```python
f_train = objective_F(..., mask_theory=splits.train_theory, mask_experiment=splits.train_experiment, ...)
```

- 注意这里传入的是 train mask，所以 `train_F` 是训练集的负对数后验。

**(2) 用当前参数预测全部题目分数**

```python
pred = predict(c=c, alpha=alpha, q_theory=ds.q_theory, q_experiment=ds.q_experiment)
```

- `pred.eta_theory (M,N_i)`
- `pred.eta_experiment (M,N_e)`

**(3) 在测试集 mask 上计算 MAE/RMSE**

```python
metrics.update(_metrics_dict("test_theory", ds.r_theory, pred.eta_theory, splits.test_theory))
metrics.update(_metrics_dict("test_experiment", ds.r_experiment, pred.eta_experiment, splits.test_experiment))
```

这里相当于：

- 每隔一段迭代在测试集上“偷看”一下当前性能（方便画曲线/观察收敛）。

#### 6.5.6 阶段 F：训练结束后的最终指标（theory/experiment/all）

训练结束后，会再算一次预测 `pred`，并构造 `final_metrics`。

除了分别算理论与实验外，还会额外算一个 **test_all**：

```python
combined_true = np.concatenate([ds.r_theory, ds.r_experiment], axis=1)
combined_pred = np.concatenate([pred.eta_theory, pred.eta_experiment], axis=1)
combined_mask = np.concatenate([splits.test_theory, splits.test_experiment], axis=1)
```

shape：

- `combined_*` 都是 `(M, N_i+N_e)`

然后对 `combined_*` 调用 `masked_mae/rmse`。

这与论文第 4.3 节情形①（把两类题当一类）是同一评价视角。

---

### 6.6 “参数取值范围 [0,1]” 在训练代码里的体现：`clip_params`

训练循环里有一段可选裁剪：

```python
if clip_params:
    c = np.clip(c, clip_min, clip_max)
...
if clip_params:
    alpha = np.clip(alpha, clip_min, clip_max)
```

这对应论文里“诊断目标值在 [0,1]”的描述，但论文没有明确说优化过程是否要投影/裁剪。

注意一个现实细节：

- `docs/ASSUMPTIONS.md` 写“默认不裁剪”，但 `configs/default.yaml` 当前设置 `clip_params: true`。
- 因此你实际用默认配置跑训练时，会开启裁剪。

本文档以**代码与配置的当前行为**为准：

- 如果你想严格遵循“无约束高斯”优化，把 `clip_params` 设为 `false` 即可。

---

## 7. 评估脚本：`eval.py`

本节将解释：

- 如何从 `params.npz` 载入 `c/alpha`
- 如何复用 `make_splits` 选择 train/val/test mask
- 如何计算并输出指标（以及可选输出目标函数值）

---

### 7.1 eval 的定位：它做的不是“训练”，而是“复现同一 split 上的评估”

`eval.py` 的核心思想是：

- 给定一组已训练好的参数 `c/alpha`
- 用与训练一致的数据集与划分协议（`make_splits`）
- 在指定 split（train/val/test）上计算：
  - masked MAE/RMSE
  - 可选：目标函数值 \(F(c,\alpha)\)

因此评估结果是否可信，取决于两点：

- **你加载的 `params.npz` 是否来自你想评估的那次训练**
- **评估时的配置（尤其 split_seed / split_mode / train_ratio）是否与训练一致**

为解决第二点，`eval.py` 支持直接传 `--run_dir`，它会优先读取 `run_dir/config.yaml`，从而保证与训练一致。

---

### 7.2 `_metrics_dict(...)`：与 `train.py` 同构的指标打包

代码位置：`eval.py: 16-21`

它与 `train.py` 的 `_metrics_dict` 完全同构：

- 调 `masked_mae/masked_rmse`
- 同时输出 `n = mask.sum()`

评估脚本用它生成：

- `test_theory_mae/rmse/n`
- `test_experiment_mae/rmse/n`
- `test_all_mae/rmse/n`（把理论与实验在 `axis=1` 拼起来再算）

---

### 7.3 `evaluate_run(cfg, c, alpha, split, compute_objective)`：评估核心函数

代码位置：`eval.py: 24-112`

#### 7.3.1 输入 / 输出

- 输入：

  - `cfg: Dict[str,Any]`：配置 dict
  - `c: (M,)`
  - `alpha: (M,K)`
  - `split: "train" | "val" | "test"`（默认 `test`）
  - `compute_objective: bool`（默认 False）
- 输出：`Dict[str,Any]`

  - 包含 dataset/split 信息
  - 包含 theory/experiment/all 三套 MAE/RMSE/n
  - 可选包含 `{split}_F`

#### 7.3.2 内部步骤（按代码顺序）

**(1) 加载数据集**

```python
ds = load_dataset_from_config(cfg)
```

得到：

- `ds.r_theory (M,N_i)`
- `ds.r_experiment (M,N_e)`
- `ds.q_theory (N_i,K)`
- `ds.q_experiment (N_e,K)`

**(2) 读取 split 配置，并重做一次划分**

```python
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
```

注意：这里是“重做一次划分”，并不是从训练结果里加载 mask。

- 如果 `cfg` 与训练时完全一致（尤其 `split_seed`），则划分结果与训练时一致。
- 如果不一致，则你得到的评估结果与训练日志里的测试指标将不可比。

**(3) 根据 `split` 选择对应的 mask**

```python
if split == "train":
    mask_theory = splits.train_theory
    mask_experiment = splits.train_experiment
elif split == "val":
    ...
elif split == "test":
    mask_theory = splits.test_theory
    mask_experiment = splits.test_experiment
```

这一步把“评估在哪个集合上”变成两个布尔矩阵：

- `mask_theory (M,N_i)`
- `mask_experiment (M,N_e)`

**(4) 预测分数：调用 `predict`**

```python
pred = predict(c=c, alpha=alpha, q_theory=ds.q_theory, q_experiment=ds.q_experiment)
```

- `pred.eta_theory (M,N_i)`
- `pred.eta_experiment (M,N_e)`

对应论文 `Eq.3.2-2/3`。

**(5) 分别计算 theory/experiment/all 的 masked MAE/RMSE**

```python
metrics.update(_metrics_dict(f"{split}_theory", ds.r_theory, pred.eta_theory, mask_theory))
metrics.update(_metrics_dict(f"{split}_experiment", ds.r_experiment, pred.eta_experiment, mask_experiment))
```

以及 combined（all）：

```python
combined_true = np.concatenate([ds.r_theory, ds.r_experiment], axis=1)
combined_pred = np.concatenate([pred.eta_theory, pred.eta_experiment], axis=1)
combined_mask = np.concatenate([mask_theory, mask_experiment], axis=1)
metrics.update(_metrics_dict(f"{split}_all", combined_true, combined_pred, combined_mask))
```

shape 复核：

- `combined_*` 都是 `(M, N_i+N_e)`

**(6) 可选：计算目标函数值 \(F\)**

如果 `compute_objective=True`：

```python
f_val = objective_F(
    c=c,
    alpha=alpha,
    r_theory=ds.r_theory,
    r_experiment=ds.r_experiment,
    q_theory=ds.q_theory,
    q_experiment=ds.q_experiment,
    mask_theory=mask_theory,
    mask_experiment=mask_experiment,
    sigma_r=..., sigma_rp=..., sigma_alpha=..., sigma_c=..., mu_alpha=..., mu_c=...
)
metrics[f"{split}_F"] = float(f_val)
```

这里的 `mask_*` 是你选择的 split（train/val/test），所以：

- `test_F` 表示目标函数只在测试条目上计算误差项（但先验项仍然全量加入）。

---

### 7.4 `_resolve_config_path(run_dir, config_arg)`：优先使用 run_dir/config.yaml

代码位置：`eval.py: 115-121`

规则非常简单：

- 如果你显式传了 `--config`，就用它
- 否则如果你传了 `--run_dir`，就默认用 `run_dir/config.yaml`
- 都没有则报错

这就是为什么推荐评估时用 `--run_dir`：

- 不容易因为配置不一致导致评估 split 不一致。

---

### 7.5 `main()`：命令行参数、加载参数、输出指标

代码位置：`eval.py: 123-190`

#### 7.5.1 `--run_dir` vs `--params`

`eval.py` 要求二者必选其一（互斥组）：

- `--run_dir <dir>`

  - 默认认为参数文件在 `<dir>/params.npz`
  - 默认 config 在 `<dir>/config.yaml`
- `--params <path/to/params.npz>`

  - 允许你直接指定参数文件
  - 如果同时没给 `--run_dir`，代码会把 `run_dir` 推断为 `params_path.parent`

#### 7.5.2 可覆盖的 split 配置

除 `--split` 之外，还允许你临时覆盖：

- `--dataset`
- `--train_ratio`
- `--val_ratio`
- `--split_seed`
- `--split_mode`

它们会写回 `cfg`，然后再调用 `evaluate_run(...)`。

#### 7.5.3 输出

- 默认打印 JSON 到 stdout
- 如果传了 `--out`，则把 JSON 保存到指定路径

---

## 8. 配置文件：`configs/default.yaml`

本节将逐项解释默认配置如何控制：数据集、划分协议、模型超参数、训练超参数与输出。

`configs/default.yaml` 是整个项目的“控制面板”。训练（`train.py`）与评估（`eval.py`）都会读取它，并把其中的字段映射到：

- 数据加载：`load_dataset_from_config`
- 划分协议：`make_splits`
- 模型超参数：`objective_F / grad_*`
- 训练超参数：交替优化循环（迭代次数、步长、是否裁剪）
- 输出路径：`_make_run_dir`

---

### 8.1 YAML 总体结构

默认配置的顶层 key：

```yaml
data: ...
split: ...
model: ...
training: ...
outputs: ...
```

你可以把它对应到训练过程的“时间顺序”：

1. 先用 `data` 确定要加载哪个数据集、如何处理 Q、缺失值等
2. 再用 `split` 确定训练/测试划分
3. 再用 `model` 决定 \(\sigma\)、\(\mu\) 等概率模型超参数
4. 再用 `training` 决定优化策略（迭代次数、步长、裁剪等）
5. 最后用 `outputs` 决定输出目录与保存内容

---

### 8.2 `data`：数据根目录、数据集选择、Q 处理、数据集规格

配置片段（节选）：

```yaml
data:
  dataset_root: dataset
  dataset_name: dataStructure
  score_scale: 10.0
  missing_value: -1.0
  q_normalize: true
  q_zero_row_strategy: unknown_skill
  split_mode: combined

  dataset_specs:
    dataStructure:
      n_theory: 58
      n_experiment: 10
      expected_k: 19
    networkSecurity:
      n_theory: 10
      n_experiment: 8
      expected_k: 7
```

#### 8.2.1 `dataset_root`

- 含义：数据集根目录
- 默认：`dataset`
- 代码使用位置：`src/data/dataset.py: load_dataset_from_config -> load_real_dataset`

#### 8.2.2 `dataset_name`

- 含义：选择哪个数据集子目录
- 默认：`dataStructure`
- 可选：`networkSecurity`
- 代码使用位置：同上

注意：本仓库 `dataset/` 下只提供了两个真实数据集（与论文 Table 1 的前两行对应）。论文中的 Synthetic 数据集在本项目里**未提供生成/加载脚本**。

#### 8.2.3 `score_scale`（分数归一化比例）

- 含义：把原始分数缩放到 `[0,1]`
- 默认：`10.0`
- 代码使用位置：`load_real_dataset` 内
  - `r_norm[mask] = r[mask] / score_scale`

对应 `docs/ASSUMPTIONS.md`：“数据归一化”。

#### 8.2.4 `missing_value`（缺失值标记）

- 含义：在 `R.csv` 中用什么数代表缺失
- 默认：`-1.0`
- 代码使用位置：`load_real_dataset` 内
  - `mask = r != missing_value`

对应 `docs/ASSUMPTIONS.md`：“缺失值处理”。

#### 8.2.5 `q_normalize`（是否对 Q 按行归一化）

- 含义：是否对每道题目的 Q 行做归一化，使每行和为 1
- 默认：`true`
- 代码使用位置：`load_real_dataset` 内
  - `q = _normalize_q_rows(q)`

对应论文 `Eq.3.1-1`。

#### 8.2.6 `q_zero_row_strategy`（Q 全零行处理策略）

- 含义：当 `q.csv` 存在某些题目一行全为 0 时怎么办
- 默认：`unknown_skill`
- 代码使用位置：`_ensure_expected_k(..., q_zero_row_strategy=...)`

策略含义：

- `unknown_skill`
  - 把全零行题目分配给“最后一列技能”（未知技能）
- `error`
  - 直接报错（更严格，但可能导致数据无法加载）

对应 `docs/ASSUMPTIONS.md`：“NetworkSecurity 中 Q 矩阵的全零行问题”。

#### 8.2.7 `split_mode`

- 含义：训练/测试划分时是否把理论+实验合在一起划分
- 默认：`combined`
- 代码使用位置：
  - `train.py` 与 `eval.py` 读取后传给 `make_splits(..., split_mode=...)`

可选值：

- `combined`
  - 在 `(M, N_i+N_e)` 的总矩阵上做一次 entry-level 随机划分
- `per_matrix`
  - 理论题与实验题分别划分

#### 8.2.8 `dataset_specs`：把论文 Table 1 的 (N_i, N_e, K) 固化为配置

每个数据集对应一个规格（spec）：

- `n_theory`：\(N_i\)
- `n_experiment`：\(N_e\)
- `expected_k`：\(K\)

这些数字与论文 Table 1 对应：

- Data Structure：`N_i=58, N_e=10, K=19`
- Network Security：`N_i=10, N_e=8, K=7`

其中 `expected_k` 对 NetworkSecurity 很关键：

- 论文写 `K=7`
- 但数据文件 `q.csv` 只有 6 列
- 所以必须用 `expected_k=7` 强制补齐一列，并用于“未知技能”兜底

---

### 8.3 `split`：划分协议（seed / train_ratio / val_ratio / 稀疏度 sweep）

配置片段：

```yaml
split:
  seed: 42
  train_ratio: 0.8
  val_ratio: 0.0
  train_ratios_sparsity: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
```

#### 8.3.1 `seed`

- 含义：划分 train/val/test 时的随机种子
- 代码使用位置：`make_splits(..., seed=split_seed)`

复现性要点：

- 只要 `seed`、`train_ratio`、`val_ratio`、`split_mode` 和数据本身不变，同一份代码每次划分结果一致。

#### 8.3.2 `train_ratio`

- 含义：训练集占“观测条目”的比例（不是按学生/题目）
- 默认：`0.8`

对应 `docs/ASSUMPTIONS.md`：“训练 / 测试划分协议”。

#### 8.3.3 `val_ratio`

- 含义：验证集占“观测条目”的比例
- 默认：`0.0`

论文没有显式引入验证集，因此默认关闭。

#### 8.3.4 `train_ratios_sparsity`

- 含义：一组训练比例，用于做“稀疏度变化”的 sweep 实验
- 代码使用位置：不是 `train.py` 本身，而是 `sweep_train_ratios.py`
  - 该脚本会循环把 `split.train_ratio` 改成列表里的值，并重复调用 `train(cfg)`

这个字段对应论文第 4.3 节提到的：

- training data ratio 从 80% 下降到 20%（以及更稀疏时优势更明显）

---

### 8.4 `model`：概率模型超参数（\(\sigma\) 与 \(\mu\)）

配置片段：

```yaml
model:
  sigma_r: 1.0
  sigma_rp: 1.0
  sigma_alpha: 1.0
  sigma_c: 1.0
  mu_alpha: 0.8
  mu_c: 0.8
```

这些参数直接进入 `objective_F / grad_*`，对应论文：

- `Eq.3.2-4`：观测噪声的精度 \(\sigma_R,\sigma_{R'}\)
- `Eq.3.2-5`：先验的均值与精度 \(\mu_\alpha,\mu_c,\sigma_\alpha,\sigma_c\)

#### 8.4.1 `sigma_r` 与 `sigma_rp`

- `sigma_r`：理论题误差项的权重（精度）
- `sigma_rp`：实验题误差项的权重（精度）

在 `objective_F` 中分别乘在：

- `loss_theory = 0.5 * sigma_r * sum((R - eta)^2)`
- `loss_experiment = 0.5 * sigma_rp * sum((R' - eta')^2)`

直观理解：

- 值越大：越“相信观测分数”，误差惩罚越重
- 值越小：越“认为噪声大”，误差惩罚越轻

#### 8.4.2 `sigma_alpha` 与 `sigma_c`

- 控制先验正则强度：
  - \(\sigma\) 越大，参数越被拉向 \(\mu\)

对应 `objective_F`：

- `prior_alpha = 0.5 * sigma_alpha * sum((alpha - mu_alpha)^2)`
- `prior_c = 0.5 * sigma_c * sum((c - mu_c)^2)`

#### 8.4.3 `mu_alpha` 与 `mu_c`

- 含义：高斯先验的均值
- 默认：`0.8`

这意味着：

- 在缺少数据（训练条目稀疏）时，模型倾向于把 `c/alpha` 往 0.8 附近收缩。

---

### 8.5 `training`：优化超参数（迭代次数、步长、裁剪、初始化）

配置片段：

```yaml
training:
  seed: 42
  n_iters: 1000
  r1: 0.01
  r2: 0.01
  log_every: 50
  clip_params: true
  clip_min: 0.0
  clip_max: 1.0
  init_method: prior_sample
```

#### 8.5.1 `seed`

- 含义：控制参数初始化的随机种子
- 代码使用位置：`_init_params` 里 `rng = np.random.default_rng(seed)`

注意：这与 `split.seed` 是两套独立的种子：

- `split.seed` 控制数据划分
- `training.seed` 控制参数初始化

#### 8.5.2 `n_iters`

- 含义：交替优化的迭代次数（本项目用固定迭代，不做收敛判据）
- 代码使用位置：`for it in range(1, n_iters+1)`

论文没有给出收敛判据细节，因此这里属于“缺失细节补全”（见第 9 节）。

#### 8.5.3 `r1` 与 `r2`

- `r1`：更新 `c` 的步长（论文 Step 1 的 \(r_1\)）
- `r2`：更新 `alpha` 的步长（论文 Step 2 的 \(r_2\)）

代码使用位置：

- `update_c(..., lr=r1)`
- `update_alpha(..., lr=r2)`

#### 8.5.4 `log_every`

- 含义：每隔多少次迭代记录一次日志
- 代码使用位置：
  - `do_log = it==1 or it==n_iters or it%log_every==0`

#### 8.5.5 `clip_params/clip_min/clip_max`

- 含义：是否把 `c` 与 `alpha` 裁剪到区间 `[clip_min, clip_max]`
- 默认：开启裁剪到 `[0,1]`

这与论文“诊断目标值域为 [0,1]”的描述一致，但论文未明确该操作是否在优化过程中进行。

同时要注意第 9 节会指出：`docs/ASSUMPTIONS.md` 的文字与默认配置在这一点上存在不一致（文档说默认不裁剪，但配置默认裁剪）。

#### 8.5.6 `init_method`

- 含义：初始化方式
- 代码使用位置：`_init_params`

可选：

- `prior_sample`：从先验采样
- `mean`：直接用先验均值

---

### 8.6 `outputs`：输出目录与保存开关

配置片段：

```yaml
outputs:
  root: outputs
  save_params: true
  save_history: true
```

- `root`：输出根目录
- `save_params`：是否保存 `params.npz`（包含 `c` 与 `alpha`）
- `save_history`：是否保存 `history.json`（训练过程日志）

代码使用位置：`train.py: main()`。

---

## 9. 重要假设与论文缺失细节补全：`docs/ASSUMPTIONS.md`

本节严格以 `docs/ASSUMPTIONS.md` 的条目顺序为主线，逐条解释：

- **论文哪里没说清/有歧义**
- **本项目采用了什么实现策略**
- **具体代码落点**（函数、变量、shape）
- **该假设对复现结果可能产生的影响**

---

### 9.1 数据归一化（score_scale=10.0）

#### 9.1.1 论文与数据的不一致

- 论文第 3.1 节声明：\(R_{ji}, R'_{je} \in [0,1]\)
- 但提供的 `dataset/*/R.csv` 实际范围是 `[0,10]`，并用 `-1` 表示缺失

#### 9.1.2 本项目的处理

- 对所有非缺失条目执行：`score / score_scale`
- 默认 `score_scale=10.0`

#### 9.1.3 代码落点

- `src/data/dataset.py: load_real_dataset`
  - `mask = r != missing_value`
  - `r_norm[mask] = r[mask] / score_scale`

shape 复核：

- `r` / `r_norm` / `mask`：`(M, N_i+N_e)`

#### 9.1.4 对结果的影响

如果你不做归一化，`R` 的量级从 1 变成 10，会导致：

- 目标函数里的平方误差项约放大 `10^2 = 100` 倍
- 相对来说先验项（`sigma_alpha/sigma_c`）的正则效果会“显得更弱”
- 学习率（`r1/r2`）的合适范围也会改变（更容易梯度爆炸）

所以归一化不仅是“为了对齐论文”，也是为了让优化超参数更可控。

---

### 9.2 缺失值处理（missing_value=-1）

#### 9.2.1 为什么需要

论文在公式推导里默认 \(R\) 是完整矩阵（每个学生每题都有分数），但真实数据存在缺失（未提交/未测/未记录）。

如果把缺失当成 0：

- 会把“未观测”误当成“得分很低”，从而系统性拉低预测。

#### 9.2.2 本项目的处理

- 通过 `mask` 显式表示“哪些条目被观测到”
- 在以下阶段都排除缺失条目：
  - 数据划分（只对观测条目划分 train/test）
  - 目标函数（只对 mask=True 的条目累积误差）
  - 指标（只对 mask=True 的条目算 MAE/RMSE）

#### 9.2.3 代码落点

- `src/data/dataset.py`

  - `mask = r != missing_value`
  - `make_splits` / `_split_observed_entries`
- `src/models/cdf_cse.py: objective_F`

  - `diff_* = (r_* - eta_*) * mask_*`
- `src/utils/metrics.py`

  - `y_true[mask]` / `y_pred[mask]`

#### 9.2.4 对结果的影响

这种实现方式的关键性质是：

- 缺失条目对训练和评估都“完全不可见”

这会让模型更像“矩阵补全/推荐系统”视角：

- 只用已观测评分拟合潜变量，再预测未观测评分。

---

### 9.3 Q 矩阵归一化（按题目行归一化）

#### 9.3.1 对应论文

论文 `Eq.3.1-1`：

\[
q_{ik} \leftarrow \frac{q_{ik}}{\sum_{l=1}^K q_{il}},\qquad
q'_{ek} \leftarrow \frac{q'_{ek}}{\sum_{l=1}^K q'_{el}}
\]

#### 9.3.2 本项目的处理

- 只对“行和>0”的行执行归一化
- 行和为 0 的行会保持 0（但默认会在更早一步用 `unknown_skill` 修复全零行）

#### 9.3.3 代码落点

- `src/data/dataset.py: _normalize_q_rows`

#### 9.3.4 对结果的影响

归一化后的含义更像“技能权重分配”：

- 如果一道题依赖 2 个技能，则每个技能权重 0.5
- 这样 \(\eta\) / \(\eta'\) 就是各技能能力的“加权平均/加权求和”

如果不归一化：

- 多技能题的 \(\eta\) 会天然更大（因为累加更多项），与论文设定不一致。

---

### 9.4 Q 矩阵方向约定（q.csv 被解释为 (N,K)）

#### 9.4.1 论文的歧义点

论文文字定义用 `q_{ik}`（题目 i、技能 k），但公式中又写 `q_{ki}`（疑似转置）。

#### 9.4.2 本项目的约定

- `q.csv` 解释为 shape `(N,K)`：
  - 行=题目
  - 列=技能

因此预测写成矩阵形式：

- \(\eta = \alpha Q^T\)
- \(\eta' = (c \odot \alpha) Q'^T\)

#### 9.4.3 代码落点

- `src/models/cdf_cse.py: predict`
  - `eta_theory = alpha @ q_theory.T`
  - `eta_experiment = (c[:, None] * alpha) @ q_experiment.T`

#### 9.4.4 对结果的影响

只要你保持一致的 shape 约定，数学上没有问题；但如果你把 Q 读成 `(K,N)`，则：

- 乘法方向会反过来
- 梯度公式也要整体重推

所以该约定属于“实现层面必须固定的一条基准”。

---

### 9.5 NetworkSecurity：技能数量不一致（expected_k=7 + “未知技能”）

#### 9.5.1 为什么需要

论文 Table 1 指出 NetworkSecurity 的技能数 \(K=7\)，但数据文件 `dataset/networkSecurity/q.csv` 只有 6 列。

#### 9.5.2 本项目的处理

- 用 `expected_k=7` 强制把 Q 的列补齐到 7
- 新增的最后一列被解释为“未知技能”（unknown skill）

#### 9.5.3 代码落点

- `src/data/dataset.py: _ensure_expected_k`
  - 当 `q.shape[1] < expected_k` 时追加全 0 列

#### 9.5.4 对结果的影响

- 这一步保证了模型参数 `alpha` 的第二维 \(K\) 与论文一致。
- 同时也为“Q 全零行”的修复提供了一个安全落点（见 9.6）。

---

### 9.6 NetworkSecurity：Q 全零行（分配到未知技能）

#### 9.6.1 为什么需要

`dataset/networkSecurity/q.csv` 中有若干题目行全 0（`docs/ASSUMPTIONS.md` 记录了索引：2、7、14、17）。

如果直接做 `Eq.3.1-1` 归一化：

- 行和为 0，无法归一化
- 或者保留全 0，会导致该题“完全不依赖技能”，与论文语义冲突

#### 9.6.2 本项目的处理

当 `q_zero_row_strategy="unknown_skill"`：

- 把这些全零行的“未知技能列”置为 1

效果：

- 行和变成 1
- 归一化后仍然是 one-hot（只有未知技能为 1）

#### 9.6.3 代码落点

- `src/data/dataset.py: _ensure_expected_k`
  - `zero_rows = q.sum(axis=1) == 0`
  - `unknown_col = str(expected_k - 1)`
  - `q.loc[zero_rows, unknown_col] = 1`

#### 9.6.4 对结果的影响

这相当于把这些题目解释为：

- 它们考察的是“其他无法归类到既有技能的能力”

因此：

- 对这些题目的拟合会主要推动 `alpha[:, unknown_skill]` 的学习。

---

### 9.7 训练/测试划分协议（entry-level 随机划分）

#### 9.7.1 为什么需要

论文只说“训练集包含 10%~80% 的成绩数据，其余用于测试”，但没有说明：

- 是按学生划分？按题目划分？还是按评分条目划分？

#### 9.7.2 本项目的解释

- “成绩数据”解释为：**已观测到的评分条目（mask=True 的 entries）**
- 在这些 entries 上随机打乱并切分 train/val/test

#### 9.7.3 代码落点

- `src/data/dataset.py: _split_observed_entries`

  - `obs_idx = np.argwhere(observed_mask)`
  - `shuffle` 后按比例切分
- `src/data/dataset.py: make_splits`

  - `split_mode=combined`：在 `(M,N_i+N_e)` 上统一切
  - `split_mode=per_matrix`：理论与实验分别切

#### 9.7.4 对结果的影响

entry-level 划分与论文“稀疏度”实验的直觉最吻合：

- 训练比例越小，矩阵越稀疏，模型更像“冷启动/早期教学阶段”

---

### 9.8 验证集设置（val_ratio=0.0）

#### 9.8.1 为什么需要

论文未显式引入验证集（只提训练/测试），因此默认不使用验证集。

#### 9.8.2 本项目的实现

- `configs/default.yaml`：`split.val_ratio = 0.0`
- `make_splits` 支持启用验证集
- 当某个 mask 的 True 个数为 0 时：
  - `masked_mae/rmse` 返回 `nan`

---

### 9.9 参数取值范围：论文说在 [0,1]，但模型是高斯（取值域为实数）

#### 9.9.1 论文的矛盾点

- 论文说诊断目标 \(c_j,\alpha_{jk}\in[0,1]\)
- 但似然与先验都是高斯分布（自然定义域是 \(\mathbb{R}\)）
- 论文没有说明是否使用截断高斯/投影/裁剪/链接函数（如 sigmoid）

#### 9.9.2 本项目的做法

- 模型层（`cdf_cse.py`）不强制范围约束
- 训练层（`train.py`）提供可配置的裁剪：
  - `clip_params: true/false`
  - `clip_min/clip_max`

#### 9.9.3 需要你注意的一点（文档与默认配置存在不一致）

- `docs/ASSUMPTIONS.md` 写“默认不裁剪”
- 但 `configs/default.yaml` 当前为 `clip_params: true`

所以在**当前仓库默认行为**下：

- 训练每步都会把 `c` 与 `alpha` 裁剪到 `[0,1]`

如果你希望严格按“无约束高斯 MAP”跑：

- 把 `clip_params` 设为 `false`

#### 9.9.4 对结果的影响

- 开启裁剪：

  - 更符合“诊断值域”的直觉
  - 可能提升稳定性（防止数值爆炸）
  - 但也可能引入“投影梯度”效应，使最优点不同于纯高斯 MAP
- 关闭裁剪：

  - 更忠实于高斯先验/似然的数学形式
  - 但可能出现 `c/alpha` 超出 [0,1]

---

### 9.10 缺失的优化超参数（r1/r2、迭代次数、先验参数等）

论文未给出：

- 步长 \(r_1,r_2\)
- 迭代次数或收敛判据
- \(\mu_c,\mu_\alpha,\sigma_c,\sigma_\alpha\) 的具体数值

本项目做法：

- 把这些都放进 `configs/default.yaml`，并在 `train.py` 中读取

代码落点：

- `train.py`

  - `train_cfg = cfg.get("training", {})`
  - `n_iters = int(train_cfg.get("n_iters", 1000))`
  - `r1 = float(train_cfg.get("r1", 0.01))`
  - `r2 = float(train_cfg.get("r2", 0.01))`
- `cdf_cse.py`

  - `objective_F/grad_*` 使用 `mu_*`、`sigma_*`

---

### 9.11 α 更新步骤（Eq.3.3-8）的歧义处理

论文写：

- `Eq.3.3-8`：\(\alpha^{new}=\alpha^{old}-r_2\,\frac{\partial}{\partial\alpha}g(\alpha)\)
- 但又定义 \(g(\alpha)=\frac{\partial F}{\partial\alpha}\)

按字面会变成“用二阶导更新”，但论文给出的 `Eq.3.3-9` 其实就是一阶梯度表达式。

本项目解释：

- Step 2 是标准梯度下降：\(\alpha^{new}=\alpha^{old}-r_2\,g(\alpha)\)

代码落点：

- `src/models/cdf_cse.py: update_alpha`
  - `alpha_new = alpha - lr * grad_alpha(...)`

---

### 9.12 评价指标公式（MAE/RMSE）

论文只描述使用 MAE/RMSE，但未给公式；本项目采用标准定义，并只在测试 mask 的条目上计算。

代码落点：

- `src/utils/metrics.py: masked_mae/masked_rmse`

---

## 10. 单元测试在验证什么：`tests/test_dataset.py`、`tests/test_cdf_cse.py`

单元测试的作用不是“复现实验结果”，而是保证：

- 数据处理的 shape/归一化/异常处理与我们的假设一致
- 模型的梯度推导正确（否则训练更新方向会错）

---

### 10.1 `tests/test_dataset.py`：验证数据加载与划分协议

#### 10.1.1 `test_load_data_structure_shapes_and_ranges`

它验证 `load_real_dataset(...)` 在 `dataStructure` 数据集上满足：

- **shape 与论文 Table 1 一致**
  - `M=96, N_i=58, N_e=10, K=19`
- **分数归一化后范围在 [0,1]**（只检查 mask=True 的观测条目）
- **Q 行归一化后每行和为 1**

对应到代码就是在测试里做了这些断言：

- `ds.r_theory.shape == (96,58)`
- `obs_theory.min() >= 0` 且 `obs_theory.max() <= 1`
- `np.allclose(ds.q_theory.sum(axis=1), 1.0)`

这套测试直接覆盖了第 3 节与第 9.1/9.3 的核心假设。

#### 10.1.2 `test_load_network_security_unknown_skill_and_normalized_q`

它验证 NetworkSecurity 的两条“异常补全”是否生效：

- `expected_k=7` 生效：`ds.n_skills == 7`
- `q_zero_row_strategy="unknown_skill"` 生效：
  - `unknown_skill_col = ds.combined_q()[:, -1]`
  - `unknown_skill_col.sum() > 0`（说明确实有题目被分配到了未知技能）

同时也检查：

- `ds.combined_q().sum(axis=1) == 1.0`（每题行和为 1）

对应第 9.5 与第 9.6。

#### 10.1.3 `test_make_splits_covers_observed_entries_combined_mode`

它验证 `make_splits(..., split_mode="combined")` 的两个关键性质：

- **覆盖性（coverage）**：train/val/test 三个集合加起来，True 的总数应当等于总观测条目数
- **互斥性（no overlap）**：任意两个集合的 mask 交集都应当为空

测试里做的事情是：

- `total_obs = ds.combined_mask().sum()`
- `covered = train.sum() + val.sum() + test.sum()`
- `covered == total_obs`
- 各种 `(A & B).sum() == 0`

对应第 3.7/3.8 与第 9.7。

---

### 10.2 `tests/test_cdf_cse.py`：验证梯度公式是否正确（对训练成败至关重要）

这个测试的核心思想是：

- 用一个很小的随机问题（小 M/K/N_i/N_e）
- 用数值差分（finite differences）近似 \(\frac{\partial F}{\partial c}\) 与 \(\frac{\partial F}{\partial \alpha}\)
- 与解析梯度 `grad_c/grad_alpha` 的输出对比

#### 10.2.1 为什么要做梯度校验？

因为训练更新是：

- `c = c - r1 * grad_c(...)`
- `alpha = alpha - r2 * grad_alpha(...)`

如果梯度推导实现错了：

- 你可能仍然能“跑完训练”，但参数会朝错误方向更新，指标不会收敛或会收敛到错误解。

#### 10.2.2 `test_gradients_match_finite_differences_small_random`

测试步骤概括：

- 随机生成：

  - `c (M,)`、`alpha (M,K)`
  - `Q/Q'`（按行归一化）
  - `R/R'`
  - `mask_theory/mask_experiment`
- 计算解析梯度：

  - `g_c = grad_c(...)`
  - `g_a = grad_alpha(...)`
- 定义一个闭包函数 `F(c_vec, a_mat)` 调用 `objective_F(...)`
- 用中心差分近似偏导：

  - \(\frac{\partial F}{\partial x} \approx \frac{F(x+\epsilon)-F(x-\epsilon)}{2\epsilon}\)
  - `eps = 1e-6`
- 对 `c` 的前 3 个维度做对比
- 对 `alpha` 的 3 个选定坐标做对比

通过阈值：

- `np.isclose(..., rtol=1e-4, atol=1e-4)`

含义：

- 数值梯度受 `eps` 与浮点误差影响，不可能完全相等，因此使用相对/绝对误差容忍。

这个测试基本等价于在说：

- `cdf_cse.py` 里写的 `grad_c/grad_alpha` 与 `objective_F` 是自洽的（一阶导正确）。

---

## 11. 读完本文你应该掌握什么？

- **你能把论文第 3 章每个核心公式，定位到代码中的哪一行/哪一段实现。**
- **你能独立画出关键矩阵的 shape（M/N_i/N_e/K），并在读代码时不再被 axis 搞混。**
- **你能解释训练循环为什么是“交替优化”，以及每一步更新到底在最小化什么。**
- **你能理解项目为了补齐论文缺失细节（数据异常、Q 方向、clip、超参数）做了哪些工程约定。**
