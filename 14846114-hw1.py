# 14846114 曾敬貴 補

import numpy as np
import numpy.linalg as la

def gram_schmidt(S1: np.ndarray):
    m, n = S1.shape
    S2 = np.zeros((m, n))
    
    for j in range(n):
        u = S1[:, j].copy()
        
        # 減去投影到前面所有正交基底的分量
        for i in range(j):
            # <v_j, e_i> / <e_i, e_i>   e_i 是單位向量，所以 <e_i, e_i> = 1
            u -= np.dot(u, S2[:, i]) * S2[:, i]  
        
        norm_u = la.norm(u)
        if norm_u < 1e-12:  # 避免除以零
            raise ValueError(f"Vector {j} is linearly dependent.")
        
        S2[:, j] = u / norm_u
    
    return S2

# 測試資料與輸出
S1 = np.array([[ 7,  4,  7, -3, -9],
               [-1, -4, -4,  1, -4],
               [ 8,  0,  5, -6,  0],
               [-4,  1,  1, -1,  4],
               [ 2,  3, -5,  1,  8]], dtype=np.float64)

S2 = gram_schmidt(S1)

np.set_printoptions(precision=2, suppress=True)
print(f'S1 => \n{S1}')
print(f'S2.T @ S2 => \n{S2.T @ S2}')