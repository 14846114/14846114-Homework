# 14846114 曾敬貴 補 待修正 

import numpy as np
import numpy.linalg as la
import cv2
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

plt.rcParams['figure.dpi'] = 144
np.random.seed(1)  # 確保生成影像可重現

def compute_energy(X: np.ndarray) -> float:
    """計算 2D 訊號的能量：||X||_F^2"""
    return float(la.norm(X, 'fro') ** 2)

def load_or_generate_image(path: str):
    """讀取灰階影像，若失敗則生成一張模擬自然場景的測試圖"""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        print(f"[Info] 成功讀取影像：{path}")
        return img.astype(np.float64)

    print(f"[Info] 無法讀取影像，使用程式生成 256x256 測試影像")
    h, w = 256, 256
    x = np.linspace(0, 1, w)[None, :]      # (1, w)
    y = np.linspace(0, 1, h)[:, None]      # (h, 1)

    # 水平與垂直漸層
    grad_h = 180 * x
    grad_v = 120 * y

    # 多個高斯亮點
    yy, xx = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h), indexing="ij")
    gauss1 = 100 * np.exp(-3.0 * ((xx + 0.5)**2 + (yy + 0.3)**2))
    gauss2 =  80 * np.exp(-4.0 * ((xx - 0.4)**2 + (yy - 0.5)**2))
    gauss3 =  60 * np.exp(-5.0 * ((xx + 0.3)**2 + (yy - 0.6)**2))

    # 紋理與雜訊
    rng = np.random.RandomState(1)
    texture = 40 * rng.rand(h, w)
    noise   = 15 * rng.normal(size=(h, w))
    blur    = cv2.GaussianBlur(texture, (9, 9), 2)

    # 合成影像
    A = grad_h + grad_v + gauss1 + gauss2 + gauss3 + blur + noise
    return np.clip(A, 0, 255).astype(np.float64)

def main():
    # 1. 載入影像 
    IMG_PATH = r"D:\python\hw\svd_demo1.jpg"
    A = load_or_generate_image(IMG_PATH)

    h, w = A.shape
    energy_A = compute_energy(A)

    # 2. 使用 NumPy 內建 SVD
    U, s, Vt = la.svd(A, full_matrices=False)  # full_matrices=False → U: (h,r), Vt: (r,w)
    rank = len(s)
    print(f"[Info] Rank (non-zero singular values): {rank}")

    Sigma = np.diag(s)  # (r, r)

    # 3. 計算 SNR vs r 
    R_MAX = 200
    rs = np.arange(1, min(R_MAX, rank) + 1)
    
    energy_N = np.zeros(R_MAX + 1)
    snr_db   = np.zeros(R_MAX + 1)

    for r in rs:
        # 重建前 r 個成分
        A_bar = U[:, :r] @ Sigma[:r, :r] @ Vt[:r, :]
        noise = A - A_bar
        energy_N[r] = compute_energy(noise)
        snr_db[r] = 10 * np.log10(energy_A / (energy_N[r] + 1e-12))

    # 處理極端值
    snr_db = np.nan_to_num(snr_db, nan=60.0, posinf=60.0, neginf=0.0)

    # 平滑曲線
    r_smooth = np.linspace(rs[2], rs[-1], 500)
    spl = make_interp_spline(rs[2:], snr_db[rs[2:]], k=3)
    snr_smooth = spl(r_smooth)

    # 4. 繪製 SNR 曲線 
    plt.figure(figsize=(6, 4))
    plt.plot(r_smooth, snr_smooth, color='red', linewidth=2)
    plt.xlabel('r')
    plt.ylabel('SNR (dB)')
    plt.xlim(0, 200)
    plt.ylim(10, 50)
    plt.xticks(np.arange(0, 225, 25))
    plt.yticks(np.arange(10, 55, 5))
    plt.grid(False)
    plt.tight_layout()
    plt.show()

    # 5. 驗證能量守恆：||A||_F^2 = sum(σ_i²) 
    lambdas = s ** 2  # σ_i² 就是 A^T A 的特徵值
    total_energy_from_svd = np.sum(lambdas)

    print(f"\n[Verify] Total energy ||A||_F^2        = {energy_A:.6f}")
    print(f"         sum of singular values²      = {total_energy_from_svd:.6f}")
    print(f"         Difference                   = {abs(energy_A - total_energy_from_svd):.6e}")

    # 6. 驗證 noise energy = sum_{i=r+1}^rank σ_i² 
    noise_energy_from_svd = np.cumsum(lambdas[::-1])[::-1]  # 從大到小累積後，取尾巴
    noise_energy_from_svd = np.insert(noise_energy_from_svd, 0, total_energy_from_svd)  # index 0 為全能量

    valid_rs = rs[rs <= rank]
    noise_err_abs = np.max(np.abs(energy_N[valid_rs] - noise_energy_from_svd[valid_rs]))
    noise_err_rel = noise_err_abs / (energy_A + 1e-12)

    approx_energy_from_svd = np.cumsum(lambdas)  # 前 r 項和
    approx_energy_direct = energy_A - energy_N[valid_rs]

    approx_err_abs = np.max(np.abs(approx_energy_direct - approx_energy_from_svd[valid_rs]))
    approx_err_rel = approx_err_abs / (energy_A + 1e-12)

    print(f"[Verify] max |energy_N[r] - tail_sum(σ²)| = {noise_err_abs:.6e}, rel = {noise_err_rel:.6e}")
    print(f"[Verify] max ||A_bar||² - sum_{1}^r(σ²)   = {approx_err_abs:.6e}, rel = {approx_err_rel:.6e}")

if __name__ == "__main__":
    main()