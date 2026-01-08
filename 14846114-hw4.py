# 14846114 曾敬貴 補

import numpy as np

def scale_to_range(X: np.ndarray, to_range=(0, 1), byrow=False):
    """
    將數值線性縮放到指定範圍（預設 [0, 1]）
    
    Parameters
    ----------
    X : np.ndarray
        1D 或 2D 陣列
    to_range : tuple
        目標範圍 (a, b)，預設 (0, 1)
    byrow : bool
        True → 按行（row-wise）縮放
        False → 按欄（column-wise）縮放（2D 時），1D 忽略此參數
    """
    a, b = to_range
    X = np.asarray(X, dtype=float)
    
    # 處理 1D 情況（視為單一維度）
    if X.ndim == 1:
        xmin = X.min()
        xmax = X.max()
        if xmax == xmin:
            return np.full_like(X, a, dtype=float).round(2)
        return (a + (X - xmin) / (xmax - xmin) * (b - a)).round(2)
    
    # 2D 情況：選擇軸
    axis = 1 if byrow else 0
    
    # 計算每行/每欄的最小與最大
    xmin = X.min(axis=axis, keepdims=True)
    xmax = X.max(axis=axis, keepdims=True)
    range_ = xmax - xmin
    
    # 避免除以零：若 range == 0，整行/整欄填 a
    Y = np.where(range_ == 0,
                 a,
                 a + (X - xmin) / range_ * (b - a))
    
    return Y.round(2)

# 測試案例 
print('test case 1:')
A = np.array([1, 2.5, 6, 4, 5])
print(f'A => \n{A}')
print(f'scale_to_range(A) => \n{scale_to_range(A)}\n\n')

print('test case 2:')
A = np.array([[1,12,3,7,8],
              [5,14,1,5,5],
              [4,11,4,1,2],
              [3,13,2,3,5],
              [2,15,6,3,2]])
print(f'A => \n{A}')
print(f'scale_to_range(A) => \n{scale_to_range(A)}\n\n')

print('test case 3:')
A = np.array([[1,2,3,4,5],
              [5,4,1,2,3],
              [3,5,4,1,2]])
print(f'A => \n{A}')
print(f'scale_to_range(A, byrow=True) => \n{scale_to_range(A, byrow=True)}\n\n')