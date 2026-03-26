import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

# =========================
# 1. 读取数据（自动清洗）
# =========================
file_path = "dataset/networkSecurity/R.csv"

df = pd.read_csv(file_path)

# ✅ 自动删除序号列
if (df.iloc[:, 0] == np.arange(len(df))).all():
    print("检测到序号列，已删除")
    df = df.iloc[:, 1:]

data = df.values.astype(np.float32)

# mask
mask = ~np.isnan(data)
data[np.isnan(data)] = 0

n_students, n_items = data.shape
max_score = np.max(data)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用设备:", device)

# =========================
# 2. 自动判断模型类型
# =========================
model_type = "2PL" if max_score <= 1 else "GRM"
print("模型类型:", model_type)

# =========================
# 3. IRT模型（稳定GRM）
# =========================
class IRTModel(nn.Module):
    def __init__(self, n_students, n_items, model_type, max_score):
        super().__init__()
        self.model_type = model_type

        self.theta = nn.Parameter(torch.randn(n_students))
        self.a = nn.Parameter(torch.ones(n_items))

        if model_type == "2PL":
            self.b = nn.Parameter(torch.zeros(n_items))
        else:
            self.k = int(max_score)
            self.raw_b = nn.Parameter(torch.randn(n_items, self.k))

    def get_ordered_b(self):
        return torch.cumsum(torch.exp(self.raw_b), dim=1)

    def forward(self):
        theta = self.theta.unsqueeze(1)
        a = self.a.unsqueeze(0)

        if self.model_type == "2PL":
            b = self.b.unsqueeze(0)
            logits = a * (theta - b)
            return torch.sigmoid(logits)

        else:
            b = self.get_ordered_b()
            probs = []

            for k in range(self.k):
                b_k = b[:, k].unsqueeze(0)
                logits = a * (theta - b_k)
                probs.append(torch.sigmoid(logits))

            probs = torch.stack(probs, dim=2)
            return torch.sum(probs, dim=2)

# =========================
# 4. 初始化
# =========================
model = IRTModel(n_students, n_items, model_type, max_score).to(device)

data_tensor = torch.tensor(data).to(device)
mask_tensor = torch.tensor(mask).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# =========================
# 5. 训练
# =========================
epochs = 300

for epoch in range(epochs):
    model.train()

    pred = model()

    loss = ((pred - data_tensor) ** 2)
    loss = loss * mask_tensor
    loss = loss.mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# =========================
# 6. 预测
# =========================
model.eval()
with torch.no_grad():
    pred = model().cpu().numpy()

# =========================
# 7. 评估（论文标准版）
# =========================
true_vals = data[mask]
pred_vals = pred[mask]

# -------- 原始 --------
mae_raw = mean_absolute_error(true_vals, pred_vals)
rmse_raw = np.sqrt(mean_squared_error(true_vals, pred_vals))

# -------- 标准化（论文常用）--------
true_norm = true_vals / max_score
pred_norm = pred_vals / max_score

mae_norm = mean_absolute_error(true_norm, pred_norm)
rmse_norm = np.sqrt(mean_squared_error(true_norm, pred_norm))

# -------- 总分RMSE（大数值来源）--------
true_total = np.sum(data, axis=1)
pred_total = np.sum(pred, axis=1)

rmse_total = np.sqrt(mean_squared_error(true_total, pred_total))

print("\n===== 评估结果 =====")
print("【原始】 MAE:", mae_raw, " RMSE:", rmse_raw)
print("【标准化】 MAE:", mae_norm, " RMSE:", rmse_norm)
print("【总分RMSE】:", rmse_total)

# =========================
# 8. 保存结果
# =========================
output_dir = "output_irt"
os.makedirs(output_dir, exist_ok=True)

# 预测
pd.DataFrame(pred).to_csv(
    os.path.join(output_dir, "irt_predictions.csv"),
    index=False
)

# θ
theta = model.theta.detach().cpu().numpy()
pd.DataFrame(theta, columns=["theta"]).to_csv(
    os.path.join(output_dir, "theta.csv"),
    index=False
)

# 题目参数
a = model.a.detach().cpu().numpy()

if model_type == "2PL":
    b = model.b.detach().cpu().numpy()
    item_df = pd.DataFrame({"a": a, "b": b})
else:
    b = model.get_ordered_b().detach().cpu().numpy()
    item_df = pd.DataFrame(
        np.column_stack([a.reshape(-1, 1), b]),
        columns=["a"] + [f"b_{k}" for k in range(b.shape[1])]
    )

item_df.to_csv(
    os.path.join(output_dir, "item_params.csv"),
    index=False
)

# 指标（全部保存）
metrics = pd.DataFrame({
    "MAE_raw": [mae_raw],
    "RMSE_raw": [rmse_raw],
    "MAE_norm": [mae_norm],
    "RMSE_norm": [rmse_norm],
    "RMSE_total": [rmse_total]
})

metrics.to_csv(
    os.path.join(output_dir, "metrics.csv"),
    index=False
)

print("\n✅ 所有结果已保存到:", output_dir)