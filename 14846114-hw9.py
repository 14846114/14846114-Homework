# 14846114 曾敬貴 補

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 讀取資料 
csv_path = r'D:\python\hw\hw9.csv'
data = pd.read_csv(csv_path).to_numpy(dtype=np.float64)

t = data[:, 0]               # 時間 (seconds)
flow_velocity = data[:, 1]   # 流量速度 (ml/sec)

dt = 0.01

# 圖1：Gas Flow Velocity 
plt.figure(figsize=(10, 6), dpi=200)
plt.plot(t, flow_velocity, 'r-', linewidth=1.5)
plt.title('Gas Flow Velocity')
plt.xlabel('time in seconds')
plt.ylabel('ml/sec')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 計算淨流量 
net_vol = np.cumsum(flow_velocity) * dt

# 圖2：Gas Net Flow
plt.figure(figsize=(10, 6), dpi=200)
plt.plot(t, net_vol, 'r-', linewidth=1.5)
plt.title('Gas Net Flow')
plt.xlabel('time in seconds')
plt.ylabel('ml')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 二次趨勢擬合與去趨勢 
# 使用 np.polyfit 直接做二次多項式最小平方迴歸
coeffs = np.polyfit(t, net_vol, deg=2)  # 返回 [a2, a1, a0]

trend_curve = np.polyval(coeffs, t)     # a2*t² + a1*t + a0

net_vol_corrected = net_vol - trend_curve

# 圖3：Gas Net Flow (Corrected)
plt.figure(figsize=(10, 6), dpi=200)
plt.plot(t, net_vol_corrected, 'b-', linewidth=1.5)
plt.title('Gas Net Flow (Corrected - Quadratic Detrending)')
plt.xlabel('time in seconds')
plt.ylabel('ml')
plt.grid(True, alpha=0.3)
plt.tight_layout()

# 儲存圖檔
plt.savefig('Gas_Net_Flow_Corrected.png', dpi=400)
plt.show()