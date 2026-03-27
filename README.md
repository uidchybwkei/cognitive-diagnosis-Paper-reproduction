# Unified Cognitive Diagnosis Infra

这个仓库现在在根目录提供了一套统一入口，用来训练、评估并导出五个模型：

- `cdf_cse`
- `fuzzycdf`
- `neuralcdm`
- `dina`
- `irt`

统一入口脚本位于根目录：

- `train.py`
- `eval.py`
- `export_students.py`

## 1. 训练

统一命令格式：

```bash
python train.py --model <model_name> --dataset <dataset_name>
```

示例：

```bash
python train.py --model cdf_cse --dataset dataStructure
python train.py --model fuzzycdf --dataset dataStructure
python train.py --model neuralcdm --dataset dataStructure
python train.py --model dina --dataset dataStructure
python train.py --model irt --dataset dataStructure
```

可选参数：

- `--config`
- `--train_ratio`
- `--val_ratio`
- `--split_seed`
- `--split_mode`

## 2. 评估

统一命令格式：

```bash
python eval.py --model <model_name> --run_dir <run_dir>
```

示例：

```bash
python eval.py --model cdf_cse --run_dir runs/cdf_cse/dataStructure/<timestamp>
python eval.py --model fuzzycdf --run_dir runs/fuzzycdf/dataStructure/<timestamp>
python eval.py --model neuralcdm --run_dir runs/neuralcdm/dataStructure/<timestamp>
python eval.py --model dina --run_dir runs/dina/dataStructure/<timestamp>
python eval.py --model irt --run_dir runs/irt/dataStructure/<timestamp>
```

默认评估 `test` split。

## 3. 导出学生结果

统一命令格式：

```bash
python export_students.py --model <model_name> --run_dir <run_dir>
```

示例：

```bash
python export_students.py --model cdf_cse --run_dir runs/cdf_cse/dataStructure/<timestamp>
python export_students.py --model fuzzycdf --run_dir runs/fuzzycdf/dataStructure/<timestamp>
python export_students.py --model neuralcdm --run_dir runs/neuralcdm/dataStructure/<timestamp>
python export_students.py --model dina --run_dir runs/dina/dataStructure/<timestamp>
python export_students.py --model irt --run_dir runs/irt/dataStructure/<timestamp>
```

默认会写入：

- `<run_dir>/export/students.csv`
- `<run_dir>/export/predictions.npz`

## 4. 输出目录结构

所有运行结果统一写到：

```text
runs/<model>/<dataset>/<timestamp>/
```

目录内尽量统一为：

```text
config.yaml
metrics.json
history.json
params.npz
run_meta.json
export/
  students.csv
  predictions.npz
```

## 5. students.csv 字段

统一导出字段：

- `student_id`
- `global_trait`
- `alpha_k0 ... alpha_k{K-1}`
- `alpha_mean`
- `alpha_sum`
- `alpha_topk`

说明：

- `global_trait` 是一个跨模型的中性字段名，不保证各模型含义完全一致。
- `alpha_k*` 表示学生在各知识点/技能上的掌握度或对应潜在表示。

各模型对应关系：

- `cdf_cse`
  - `global_trait = c`
  - `alpha = alpha`
- `fuzzycdf`
  - `global_trait = theta`
  - `alpha = alpha`
- `neuralcdm`
  - `global_trait = NaN`
  - `alpha = sigmoid(student_emb)`
- `dina`
  - `global_trait = NaN`
  - `alpha = alpha`（学生在各技能上的掌握概率）
- `irt`
  - `global_trait = theta`（学生整体能力）
  - `alpha = sigmoid(theta)`（工程映射：把同一个 theta 广播到所有技能维度）

## 6. 当前做过的妥协

为了把这些模型统一到一套最简单的 infra 中，做了下面这些妥协。

### 6.1 FuzzyCDF

原始 `FuzzyCDF` 依赖题目类型信息，例如：

- `Obj`
- `Sub`

但当前统一输入只要求：

- `R.csv`
- `q.csv`

因此这里采用了一个简化策略：

- **默认所有题都视为 subjective**

这意味着当前的 `fuzzycdf` 训练结果是一个适配版结果，不等同于严格复现原始实现的所有细节。

### 6.2 NeuralCDM

原始 `NeuralCDM` 面向的是二值作答数据。

但当前统一输入中的 `R.csv` 可能是连续分数，并且在 `CDF_CSE` 体系里通常会被归一化到 `[0,1]`。

因此这里采用了简化策略：

- 对归一化分数按阈值 `0.5` 做二值化
- `score >= 0.5` 视为 `1`
- `score < 0.5` 视为 `0`

因此当前 `neuralcdm` 的训练和评估，是建立在这个统一二值化约定上的。

### 6.3 global_trait 的统一命名

这些模型里并没有天然同义的 `c`：

- `CDF_CSE` 有原生 `c`
- `FuzzyCDF` 更接近 `theta`
- `NeuralCDM` 没有自然对应的单一全局标量

因此统一导出里不再使用 `c` 作为公共列名，而改为：

- `global_trait`

这是工程上的统一命名，不代表这些模型的该字段可以直接做严格同义比较。

### 6.4 DINA

DINA 需要二值作答，因此在统一输入中：

- 使用 `cfg.dina.label_threshold`（默认 `0.6`）对归一化到 `[0,1]` 的 `R.csv` 做 `0/1` 化

### 6.5 IRT

当前统一输入中的 `R.csv` 已归一化到 `[0,1]`，因此这里采用 2PL 形式直接对连续分数做拟合（输出仍在 `[0,1]`）；
`Q.csv` 在 IRT 训练中不会被使用。

## 7. 推荐理解方式

如果你把这套 infra 当成一个统一实验平台，建议这样理解：

- `train.py`：统一训练入口
- `eval.py`：统一评估入口
- `export_students.py`：统一学生层导出入口
- `runs/`：统一结果归档位置

如果你要做严格论文复现级别比较，仍然需要分别检查：

- 每个模型的原始数据假设
- 每个模型的标签语义
- 每个模型的潜变量定义

## 8. 目前状态

当前已完成统一接入（可走统一入口训练、评估、导出）：

- `cdf_cse` 可以训练、评估、导出
- `fuzzycdf` 可以训练、评估、导出
- `neuralcdm` 可以训练、评估、导出
- `dina` 可以训练、评估、导出
- `irt` 可以训练、评估、导出

其中：

- `cdf_cse` 最接近原始实现语义
- `fuzzycdf` 和 `neuralcdm` 属于为了统一输入输出而做过适配的版本
