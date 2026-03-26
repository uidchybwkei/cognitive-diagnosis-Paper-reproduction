import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

# =========================
# 1. 输入文件路径
# =========================
response_path = "dataset/networkSecurity/R.csv"
q_matrix_path = "dataset/networkSecurity/q.csv"

# =========================
# 2. 读取 & 自动清洗 R
# =========================
df_R = pd.read_csv(response_path)

# ✅ 去掉第一列（序号列）
if (df_R.iloc[:, 0] == np.arange(len(df_R))).all():
    print("检测到序号列，已删除")
    df_R = df_R.iloc[:, 1:]

R = df_R.values.astype(np.float32)

# =========================
# 转换为0/1（DINA必须）
# =========================
full_score = np.max(R)
threshold = 0.6 * full_score  # 👉 可调整

R = (R >= threshold).astype(np.float32)

# =========================
# 读取Q矩阵
# =========================
Q = pd.read_csv(q_matrix_path).values.astype(np.float32)

# =========================
# 强制对齐
# =========================
if R.shape[1] != Q.shape[0]:
    print("⚠️ R和Q题目数不一致，自动对齐")

    min_items = min(R.shape[1], Q.shape[0])
    R = R[:, :min_items]
    Q = Q[:min_items, :]

print("R shape:", R.shape)
print("Q shape:", Q.shape)

# =========================
# mask
# =========================
mask = ~np.isnan(R)
R[np.isnan(R)] = 0

n_students, n_items = R.shape
_, n_skills = Q.shape

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用设备:", device)

# =========================
# 3. DINA模型（稳定版）
# =========================
class DINA(nn.Module):
    def __init__(self, n_students, n_items, n_skills, Q):
        super().__init__()

        self.alpha = nn.Parameter(torch.randn(n_students, n_skills))

        self.slip = nn.Parameter(torch.rand(n_items))
        self.guess = nn.Parameter(torch.rand(n_items))

        self.Q = torch.tensor(Q).to(device)

    def forward(self):
        alpha_prob = torch.sigmoid(self.alpha)

        alpha_expanded = alpha_prob.unsqueeze(1)  # (N,1,K)
        Q_expanded = self.Q.unsqueeze(0)          # (1,M,K)

        eta = torch.prod(
            alpha_expanded ** Q_expanded + (1 - Q_expanded),
            dim=2
        )

        slip = torch.sigmoid(self.slip).unsqueeze(0)
        guess = torch.sigmoid(self.guess).unsqueeze(0)

        # ✅ 稳定版本（重要）
        prob = eta * (1 - slip) + (1 - eta) * guess

        return prob, alpha_prob

# =========================
# 4. 初始化
# =========================
model = DINA(n_students, n_items, n_skills, Q).to(device)

R_tensor = torch.tensor(R).to(device)
mask_tensor = torch.tensor(mask).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# =========================
# 5. 训练
# =========================
epochs = 300

for epoch in range(epochs):
    model.train()

    pred, _ = model()

    loss = ((pred - R_tensor) ** 2)
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
    pred, alpha_prob = model()
    pred = pred.cpu().numpy()
    alpha_prob = alpha_prob.cpu().numpy()

# =========================
# 7. 评估
# =========================
# =========================
# 7. 评估（MSE / RMSE标准版）
# =========================
true_vals = R[mask]
pred_vals = pred[mask]

# ✅ MSE
mse = mean_squared_error(true_vals, pred_vals)

# ✅ RMSE
rmse = np.sqrt(mse)

# ✅ 总分RMSE（和IRT对齐）
true_total = np.sum(R, axis=1)
pred_total = np.sum(pred, axis=1)

rmse_total = np.sqrt(mean_squared_error(true_total, pred_total))

print("\n===== DINA评估（MSE版） =====")
print("MSE:", mse)
print("RMSE:", rmse)
print("RMSE_total:", rmse_total)

# =========================
# 8. 输出目录
# =========================
base_dir = os.path.dirname(response_path)
output_dir = "output_dina"

os.makedirs(output_dir, exist_ok=True)

# =========================
# 9. 保存结果
# =========================

# 预测
pd.DataFrame(pred).to_csv(
    os.path.join(output_dir, "dina_predictions.csv"),
    index=False
)

# 学生技能概率
pd.DataFrame(alpha_prob).to_csv(
    os.path.join(output_dir, "student_mastery.csv"),
    index=False
)

# 二值技能
alpha_binary = (alpha_prob > 0.5).astype(int)
pd.DataFrame(alpha_binary).to_csv(
    os.path.join(output_dir, "student_mastery_binary.csv"),
    index=False
)

# 题目参数
slip = torch.sigmoid(model.slip).detach().cpu().numpy()
guess = torch.sigmoid(model.guess).detach().cpu().numpy()

item_df = pd.DataFrame({
    "slip": slip,
    "guess": guess
})

item_df.to_csv(
    os.path.join(output_dir, "item_params.csv"),
    index=False
)

# 指标
# 指标
pd.DataFrame({
    "MSE": [mse],
    "RMSE": [rmse],
    "RMSE_total": [rmse_total]
}).to_csv(os.path.join(output_dir, "metrics.csv"), index=False)


print("✅ DINA结果已保存到:", output_dir)