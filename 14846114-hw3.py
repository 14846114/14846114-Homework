# 14846114 曾敬貴 補

import numpy as np
import matplotlib.pyplot as plt

# 產生方波資料 
pts = 50
x = np.linspace(-2, 2, pts)
y = np.ones_like(x)
y[x < 0] = -1  # 方波定義：x < 0 為 -1，x >= 0 為 +1

# 傅立葉基底參數
T0 = x.max() - x.min()          # 週期 = 4
omega0 = 2 * np.pi / T0          # 基頻

n = 5                            # 最高諧波階數
# 基底順序：1, cos(ω0 x), ..., cos(n ω0 x), sin(ω0 x), ..., sin(n ω0 x)
k = np.arange(1, n + 1)

# 建構設計矩陣 X 
ones = np.ones_like(x)
cos_terms = np.cos(k[:, np.newaxis] * omega0 * x)   # shape: (n, pts)
sin_terms = np.sin(k[:, np.newaxis] * omega0 * x)   # shape: (n, pts)

X = np.vstack([ones, cos_terms, sin_terms]).T       # shape: (pts, 1 + 2n)

# 用 NumPy 內建 SVD 求最小平方解 
# X = U Σ V^T 
U, s, Vt = np.linalg.svd(X, full_matrices=False)
Sigma_inv = np.diag(1.0 / s)                        # Σ^{-1}

# 偽逆：X^+ = V Σ^{-1} U^T
a = Vt.T @ Sigma_inv @ U.T @ y

# 預測值 y_bar 
y_bar = X @ a

# 畫圖 
plt.figure(figsize=(8, 5), dpi=144)
plt.plot(x, y, 'b-', linewidth=2, label='true values (square wave)')
plt.plot(x, y_bar, 'g-', linewidth=2, label='Fourier approximation (n=5)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Fourier Series Approximation of Square Wave')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()