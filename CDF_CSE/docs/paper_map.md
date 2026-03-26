# docs/paper_map.md

本文档仅依据 `MD/paper.md`（论文全文）整理，用于后续“逐条对照实现”。

## 0. 复现目标（来自 paper.md 第 3.1 节 Problem definition）

给定：

- 学生在理论题上的得分矩阵 `R`
- 学生在实验（编程）题上的得分矩阵 `R'`
- 理论题的 Q 矩阵 `Q`
- 实验题的 Q 矩阵 `Q'`

目标（paper.md 原文三点）：

- (Ⅰ) 诊断学生 `j` 的整体编程能力 `c_j`
- (Ⅱ) 诊断学生 `j` 对技能 `k` 的理论掌握 `α_{jk}` 与实验掌握 `β_{jk}`
- (Ⅲ) 预测学生 `j` 在新理论题 `i` 或新实验题 `e` 上的表现：`η_{ji}` 与 `η'_{je}`

论文声明上述诊断目标“均取值于 `[0,1]`”。

## 1. 符号表 / 变量含义 / 维度

### 1.1 索引范围（paper.md 第 3.1 节）

- 学生索引：`j = 1,2,…,M`
- 理论题索引：`i = 1,2,…,N_i`
- 实验题索引：`e = 1,2,…,N_e`
- 技能/知识概念索引：`k = 1,2,…,K`
- 归一化求和索引：`l = 1,2,…,K`

### 1.2 观测数据（paper.md 第 3.1 节）

- `M`
  - 含义：学生数量
- `K`
  - 含义：技能/知识概念数量
- `N_i`
  - 含义：理论题数量
- `N_e`
  - 含义：实验题（编程实验）数量

- `R ∈ R^{M×N_i}`
  - 元素：`R_{ji}`，学生 `j` 在理论题 `i` 上的得分
  - 取值：论文写明 `R_{ji} ∈ [0,1]`

- `R' ∈ R^{M×N_e}`
  - 元素：`R'_{je}`，学生 `j` 在实验题 `e` 上的得分
  - 取值：论文写明 `R'_{je} ∈ [0,1]`

- `Q`
  - 含义：理论题-技能指示矩阵
  - 维度：论文文字描述为 `N_i` 行、`K` 列
  - 元素：`q_{ik}` 表示理论题 `i` 是否考察技能 `k`（`1/0`）

- `Q'`
  - 含义：实验题-技能指示矩阵
  - 维度：论文文字描述为 `N_e` 行、`K` 列
  - 元素：`q'_{ek}` 表示实验题 `e` 是否需要技能 `k`（`1/0`）

### 1.3 潜变量 / 诊断结果（paper.md 第 3.1-3.2 节）

- `c_j`
  - 含义：学生 `j` 的整体编程能力（高阶潜变量）

- `α_{jk}`
  - 含义：学生 `j` 使用技能 `k` 做理论题的能力（theoretical mastery）

- `β_{jk}`
  - 含义：学生 `j` 使用技能 `k` 做实验（写代码）的能力（experimental mastery）

- `η_{ji}`
  - 含义：学生 `j` 对理论题 `i` 的掌握度/预测表现（paper.md：predicted performance）

- `η'_{je}`
  - 含义：学生 `j` 对实验题 `e` 的掌握度/预测表现

### 1.4 超参数（paper.md 第 3.2-3.3 节）

- `σ_R`, `σ_{R'}`
  - 含义：观测噪声的超参数（论文中用于高斯分布的精度 `σ^{-1}`）
- `σ_α`, `σ_c`
  - 含义：先验分布的超参数（精度）
- `μ_α`, `μ_c`
  - 含义：先验分布的均值
- `r_1`, `r_2`
  - 含义：梯度下降步长（step length）
- `I`
  - 含义：单位矩阵（identity matrix）

## 2. 数据集（paper.md 第 4 节，Table 1）

**Table 1. Overview of datasets**：

| Data Set | # Student | # Skill | # Problem (Theoretical) | # Problem (Experimental) |
| --- | ---: | ---: | ---: | ---: |
| “Data Structure” | 96 | 19 | 58 | 10 |
| “Network Security” | 194 | 7 | 10 | 8 |
| Synthetic | 1000 | 20 | 200 | 50 |

## 3. 训练目标 / 概率图模型（paper.md 第 3.2-3.3 节）

- 论文以概率图模型描述生成过程（Figure 1）。
- 训练目标：最小化负对数后验 `F(c, α)`（paper.md 第 3.3 节）。

## 4. 公式清单（实现必须逐条对照）

为便于代码中引用，以下公式使用“自定义编号”，格式为 `Eq.[section]-[order]`，并保留 paper.md 原始公式表达。

### 4.1 归一化（paper.md 第 3.1 节）

**Eq.3.1-1**

$$
q_{ik}=\frac{q_{ik}}{\sum_{l=1}^{K} q_{il}},\qquad
q'_{ek}=\frac{q'_{ek}}{\sum_{l=1}^{K} q'_{el}}.
$$

### 4.2 技能理论-实验关系（paper.md 第 3.2 节）

**Eq.3.2-1**

$$
\beta_{jk}=c_j\alpha_{jk}.
$$

### 4.3 题目掌握度（paper.md 第 3.2 节，Problem mastery）

**Eq.3.2-2**

$$
\eta_{ji}=\sum_{k=1}^{K}\alpha_{jk} q_{ki}
$$

**Eq.3.2-3**

$$
\eta'_{je}=\sum_{k=1}^{K}\beta_{jk} q'_{ke}.
$$

### 4.4 观测分布（paper.md 第 3.2 节，Actual score）

**Eq.3.2-4**

$$
R_{ji}\sim \mathcal{N}(\eta_{ji},\sigma_R^{-1}I),\qquad
R'_{je}\sim \mathcal{N}(\eta'_{je},\sigma_{R'}^{-1}I).
$$

### 4.5 先验（paper.md 第 3.2 节，prior distribution）

**Eq.3.2-5**

$$
c_j \sim \mathcal{N}(\mu_c,\sigma_c^{-1}I),\qquad
\alpha_{jk} \sim \mathcal{N}(\mu_\alpha,\sigma_\alpha^{-1}I),
$$

### 4.6 后验分解（paper.md 第 3.3 节）

**Eq.3.3-1**

$$
P(c,\alpha \mid R,R') \propto P(R\mid \alpha)P(R'\mid c,\alpha)P(c)P(\alpha).
$$

### 4.7 各项概率分布（paper.md 第 3.3 节）

**Eq.3.3-2**

$$
P(R_{ji}\mid \alpha_j)=\mathcal{N}\!\left(\sum_{k=1}^{K}\alpha_{jk} q_{ki},\sigma_R^{-1}I\right),
$$

**Eq.3.3-3**

$$
P(R'_{je}\mid c_j,\alpha_j)=\mathcal{N}\!\left(\sum_{k=1}^{K}c_j\alpha_{jk} q'_{ke},\sigma_{R'}^{-1}I\right),
$$

**Eq.3.3-4**

$$
P(c_j)=\mathcal{N}(\mu_c,\sigma_c^{-1}I),\qquad
P(\alpha_{jk})=\mathcal{N}(\mu_\alpha,\sigma_\alpha^{-1}I).
$$

### 4.8 负对数后验目标函数（paper.md 第 3.3 节）

**Eq.3.3-5**

$$
\begin{aligned}
F(c,\alpha)=&\sum_{j=1}^{M}\sum_{i=1}^{N_i}\frac{\sigma_R}{2}\left(R_{ji}-\eta_{ji}\right)^2
+\sum_{j=1}^{M}\sum_{e=1}^{N_e}\frac{\sigma_{R'}}{2}\left(R'_{je}-\eta'_{je}\right)^2 \\
&+\frac{\sigma_\alpha}{2}\sum_{j=1}^{M}\sum_{k=1}^{K}\left(\alpha_{jk}-\mu_\alpha\right)^2
+\frac{\sigma_c}{2}\sum_{j=1}^{M}\left(c_j-\mu_c\right)^2 .
\end{aligned}
$$

### 4.9 交替优化（paper.md 第 3.3 节）

**Eq.3.3-6**（Step 1：w.r.t. `c`）

$$
c^{\mathrm{new}}=c^{\mathrm{old}}-r_1 g(c),
$$

**Eq.3.3-7**（Step 1 的梯度表达式）

$$
g(c)=-\sigma_{R'}\sum_{e=1}^{N_e}\left(R'_{je}-\eta'_{je}\right)\left(\sum_{k=1}^{K}\alpha_{jk} q'_{ke}\right)
+\sigma_c(c_j-\mu_c).
$$

**Eq.3.3-8**（Step 2：w.r.t. `α`，paper.md 原文如下）

$$
\alpha^{\mathrm{new}}=\alpha^{\mathrm{old}}-r_2\frac{\partial}{\partial \alpha}g(\alpha),
$$

**Eq.3.3-9**（paper.md 给出的 `g(α)`）

$$
g(\alpha)=-\sigma_R\sum_{i=1}^{N_i} q_{ki}\left(R_{ji}-\eta_{ji}\right)
-\sigma_{R'}\sum_{e=1}^{N_e} c_j q'_{ke}\left(R'_{je}-\eta'_{je}\right)
+\sigma_\alpha\left(\alpha_{jk}-\mu_\alpha\right).
$$

## 5. 评价指标（paper.md 第 4.3 节）

论文用于衡量“预测得分与真实得分的误差”的指标：

- MAE
- RMSE

> paper.md 未给出 MAE/RMSE 的数学定义（公式），后续实现时若需要具体公式将作为“缺失细节”写入 `ASSUMPTIONS.md` 并做可配置。

## 6. 实验协议（paper.md 第 4.3-4.5 节，按原文可得信息）

- 训练在两种情形下进行：
  - ① Treat both kinds of questions as the same kind of question
  - ② Divide two kinds of problems into two data sets and train them separately
- 训练数据稀疏度变化：training data ratio 从 80% 下降到 20%
- “teaching process” 实验（paper.md 第 4.5 节）：
  - 固定训练集数据量为 80%，其余为测试集
  - 按时间顺序逐步增加用于训练的问题数量

## 7. 需要复现/生成的论文图表（paper.md 内出现的图/表）

- Figure 1：CDF-CSE 的概率图模型
- Table 1：数据集概览（见上）
- Figure 2：在 “data structure” 数据集上的各模型性能
- Figure 3：在 “network security” 数据集上的各模型性能
- Figure 4：在 synthetic 数据集上的各模型性能
- Figure 5：teaching process 场景下的性能曲线

> 注：Figures 2-5 的坐标轴/图例/曲线对应的精确定义在 paper.md 的图片文件中；若 paper.md 未提供可机器解析的具体数值，后续将以“重新计算并生成同标题/同指标/同横轴定义”的方式复现，并在 `ASSUMPTIONS.md` 记录任何不可避免的差异。

## 8. 发现的符号/描述不一致与缺失点（将进入 ASSUMPTIONS.md）

仅根据 paper.md 可观察到以下潜在不一致/缺失（此处先记录，后续实现时会再次核对并写入 `ASSUMPTIONS.md`）：

- `Q` 的元素符号：文字定义为 `q_{ik}`（`N_i × K`），但多处公式使用 `q_{ki}`（疑似转置）。
- Step 2 更新式（Eq.3.3-8）写为 `- r2 * (∂/∂α) g(α)`，但紧接着又定义 `g(α)=∂F/∂α`；需要明确实现时采用哪一种（梯度/海森）。
- `c_j`、`α_{jk}` 被描述为诊断目标值在 `[0,1]`，但先验与似然均为高斯分布（取值域为实数）；paper.md 未说明是否截断/clip 或使用链接函数。
- 训练/验证/测试划分的具体随机方式、缺失值处理方式、初始化、迭代次数/收敛准则、`σ_*` 与 `μ_*` 的取值、`r_1`/`r_2` 取值：paper.md 未给出。
