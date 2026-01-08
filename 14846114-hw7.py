# 14846114 曾敬貴 補

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 讀取資料 
csv_path = r'D:\python\hw\hw7.csv'
data = pd.read_csv(csv_path).to_numpy(dtype=np.float64)
x = data[:, 0]
y = data[:, 1]

# 模型與成本函數 
def model(w, t):
    return w[0] + w[1] * np.sin(w[2] * t + w[3])

def cost(w, x, y):
    y_hat = model(w, x)
    return np.sum((y - y_hat) ** 2)

# 梯度下降參數 
initial_w = np.array([-0.1607108,  2.0808538,  0.3277537, -1.5511576])
alpha = 0.05
max_iters = 500 
n_iters = max_iters - 1

# Analytic Gradient Descent
def analytic_gradient(w, x, y):
    s = np.sin(w[2] * x + w[3])
    c = np.cos(w[2] * x + w[3])
    e = y - (w[0] + w[1] * s)
    
    g0 = -2.0 * np.sum(e)
    g1 = -2.0 * np.sum(e * s)
    g2 = -2.0 * np.sum(e * w[1] * x * c)
    g3 = -2.0 * np.sum(e * w[1] * c)
    
    return np.array([g0, g1, g2, g3])

w_analytic = initial_w.copy()
for _ in range(n_iters):
    grad = analytic_gradient(w_analytic, x, y)
    w_analytic -= alpha * grad

# Numeric Gradient Descent 
eps = 1e-8
def numeric_gradient(w, x, y):
    J0 = cost(w, x, y)
    grad = np.zeros_like(w)
    for k in range(len(w)):
        w_eps = w.copy()
        w_eps[k] += eps
        grad[k] = (cost(w_eps, x, y) - J0) / eps
    return grad

w_numeric = initial_w.copy()
for _ in range(n_iters):
    grad = numeric_gradient(w_numeric, x, y)
    w_numeric -= alpha * grad

# 畫圖準備 
x_gap = (x.max() - x.min()) * 0.2
y_gap = (y.max() - y.min()) * 0.2
xmin, xmax = x.min() - x_gap, x.max() + x_gap
ymin, ymax = y.min() - y_gap, y.max() + y_gap

xt = np.linspace(xmin, xmax, 200)  # 讓曲線平滑
yt_analytic = model(w_analytic, xt)
yt_numeric  = model(w_numeric,  xt)

# 畫圖 
plt.figure(figsize=(10, 6), dpi=144)

# 資料點
plt.scatter(x, y, color='k', edgecolor='w', linewidth=0.9, s=60, zorder=3, label='data')

# 兩條擬合曲線
plt.plot(xt, yt_analytic, linewidth=4, color='blue',  label='Analytic method')
plt.plot(xt, yt_numeric,  linewidth=2, color='red',   label='Numeric method')

plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('Gradient Descent: Analytic vs Numeric Gradient')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()