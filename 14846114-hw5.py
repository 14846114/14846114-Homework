# 14846114 曾敬貴 補

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 讀取資料 
csv_path = r'D:\python\hw\hw5.csv'
hw5_csv = pd.read_csv(csv_path)
data = hw5_csv.to_numpy(dtype=np.float64)

hours = data[:, 0]
sulfate = data[:, 1]

# (1) 濃度 vs 時間：散佈圖 + 三次多項式迴歸 
plt.figure(figsize=(8, 5), dpi=144)

# 原始散佈圖
plt.scatter(hours, sulfate, color='black', s=30, label='data')

# 使用 np.polyfit 做多項式最小平方迴歸（deg=3）
deg = 3
coeffs = np.polyfit(hours, sulfate, deg)

# 在更密的格點上計算預測曲線
t_grid = np.linspace(hours.min(), hours.max(), 500)
sulfate_pred = np.polyval(coeffs, t_grid)

plt.plot(t_grid, sulfate_pred, 'b-', linewidth=2, label=f'poly deg={deg} regression')

plt.title('Sulfate concentration vs time')
plt.xlabel('time in hours')
plt.ylabel('sulfate concentration (times $10^{-4}$)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# (2) log(濃度) vs log(時間)：散佈圖 + 線性迴歸 
# 只取正值避免 log(0) 或 log(負)
mask = (hours > 0) & (sulfate > 0)
hours_pos = hours[mask]
sulfate_pos = sulfate[mask]

log_t = np.log(hours_pos)
log_c = np.log(sulfate_pos)

plt.figure(figsize=(8, 5), dpi=144)

# log-log 散佈圖
plt.scatter(log_t, log_c, color='black', s=30, label='log-log data')

# 在 log-log 空間做線性迴歸（deg=1）
coeffs_log = np.polyfit(log_t, log_c, 1)

# 畫預測直線
xg = np.linspace(log_t.min(), log_t.max(), 500)
yg = np.polyval(coeffs_log, xg)

plt.plot(xg, yg, 'b-', linewidth=2, label='linear regression in log-log')

plt.title('log(sulfate concentration) vs log(time)')
plt.xlabel('log(time in hours)')
plt.ylabel('log(sulfate concentration (times $10^{-4}$))')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()