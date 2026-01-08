# 14846114 曾敬貴 補

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)  # 固定點雲，讓每次都一樣

# 生成資料 
mean1 = np.array([0, 5])
sigma1 = np.array([[0.3, 0.2], [0.2, 1.0]])
N1 = 200
X1 = np.random.multivariate_normal(mean1, sigma1, N1)

mean2 = np.array([3, 4])
sigma2 = np.array([[0.3, 0.2], [0.2, 1.0]])
N2 = 100
X2 = np.random.multivariate_normal(mean2, sigma2, N2)

# LDA：計算 w 
m1 = np.mean(X1, axis=0)
m2 = np.mean(X2, axis=0)

S1 = (X1 - m1).T @ (X1 - m1)
S2 = (X2 - m2).T @ (X2 - m2)
Sw = S1 + S2

w = np.linalg.inv(Sw) @ (m1 - m2)
w /= np.linalg.norm(w)  # 單位化

# class 2(綠) 投影平均 > class 1(紅)
if np.dot(m1, w) > np.dot(m2, w):
    w = -w

# 一維投影 
y1 = X1 @ w
y2 = X2 @ w

m_all = (N1 * m1 + N2 * m2) / (N1 + N2)

# 垂直向量
n_perp = np.array([-w[1], w[0]])
n_perp /= np.linalg.norm(n_perp)

# 偏移更大，讓投影點雲/線更下方
p0 = m_all - 4.5 * n_perp  

X1_proj = p0 + y1[:, np.newaxis] * w
X2_proj = p0 + y2[:, np.newaxis] * w

t_min = min(y1.min(), y2.min()) - 1
t_max = max(y1.max(), y2.max()) + 1
t_line = np.linspace(t_min, t_max, 200)
line_pts = p0 + t_line[:, np.newaxis] * w

# 畫圖 
plt.figure(figsize=(10, 7), dpi=144)  

plt.scatter(X1[:, 0], X1[:, 1], c='red',    alpha=0.7, s=35, label='class 1')
plt.scatter(X2[:, 0], X2[:, 1], c='green',  alpha=0.7, s=35, label='class 2')

plt.scatter(X1_proj[:, 0], X1_proj[:, 1], c='red',    alpha=0.9, s=35)
plt.scatter(X2_proj[:, 0], X2_proj[:, 1], c='green',  alpha=0.9, s=35)

plt.plot(line_pts[:, 0], line_pts[:, 1], 'k-', linewidth=2, label='projection axis')
plt.axhline(0, color='gray', linewidth=0.8, linestyle='--')

plt.xlim(-6, 7)
plt.ylim(-1, 7)
plt.xlabel('x1')
plt.ylabel('x2 / projection')
plt.title('LDA: data and projection onto w')
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()