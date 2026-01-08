# 14846114 曾敬貴 補

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from matplotlib.colors import ListedColormap

# 讀取資料 
csv_path = r'D:\python\hw\hw8.csv' 
data = pd.read_csv(csv_path).to_numpy(dtype=np.float64)

X = data[:, :2]      # 特徵 (m, 2)
y = data[:, 2]       # 標籤 (-1 或 +1)

# 訓練 RBF Kernel SVM 
clf = SVC(kernel='rbf', C=5.0, gamma='scale')
clf.fit(X, y)

# 產生網格並預測（決策區域） 
x_min, x_max = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
y_min, y_max = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 500),
    np.linspace(y_min, y_max, 500)
)

grid = np.c_[xx.ravel(), yy.ravel()]
Z = clf.predict(grid)
Z = Z.reshape(xx.shape)

# 畫圖 
plt.figure(figsize=(8, 7), dpi=144)

# 背景分類區域（淺綠/深綠）
cmap_bg = ListedColormap(['#d5f5d5', '#1b5e20'])  # 負類淺綠、正類深綠
plt.contourf(xx, yy, Z, alpha=0.6, cmap=cmap_bg)

# 決策邊界（Z=0 的等高線）
plt.contour(xx, yy, Z, levels=[0], colors='black', linewidths=2)

# 資料點
plt.scatter(X[y ==  1, 0], X[y ==  1, 1], c='red',   edgecolor='k', s=60, label=r'$\omega_1$ (+1)')
plt.scatter(X[y == -1, 0], X[y == -1, 1], c='blue',  edgecolor='k', s=60, label=r'$\omega_2$ (-1)')

plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('RBF Kernel SVM Decision Boundary (C=5.0, gamma="scale")')
plt.axis('equal')          # 保持 x/y 比例一致
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()