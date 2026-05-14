import math
import random
import matplotlib.pyplot as plt
# Dùng để lấy phân vị của phân phối chuẩn
from scipy import stats

#HÀM BỔ TRỢ MA TRẬN (CORE LOGIC)
def transpose(A):
    """Chuyển vị ma trận."""
    return [[A[j][i] for j in range(len(A))] for i in range(len(A[0]))]

def matmul(A, B):
    """Nhân hai ma trận hoặc Ma trận x Vector."""
    # Kiểm tra nếu B là vector (list 1 chiều)
    if isinstance(B[0], (int, float)):
        return [sum(row[i] * B[i] for i in range(len(B))) for row in A]
    # Nếu B là ma trận (list 2 chiều)
    return [[sum(a * b for a, b in zip(A_row, B_col)) for B_col in zip(*B)] for A_row in A]

def matrix_inverse(A):
    """Nghịch đảo ma trận bằng phương pháp Gauss-Jordan."""
    n = len(A)
    # Tạo ma trận đơn vị I
    I = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    # Ma trận mở rộng [A | I]
    copy_A = [row[:] for row in A]
    
    for i in range(n):
        # Tìm phần tử trục (pivot)
        pivot = copy_A[i][i]
        for j in range(i, n): copy_A[i][j] /= pivot
        for j in range(n): I[i][j] /= pivot
        
        for k in range(n):
            if k != i:
                factor = copy_A[k][i]
                for j in range(i, n): copy_A[k][j] -= factor * copy_A[i][j]
                for j in range(n): I[k][j] -= factor * I[i][j]
    return I

# 1. TASK 7: RESIDUAL PLOTS
def task_7_residual_analysis(X, y, beta):
    """Vẽ 4 biểu đồ chẩn đoán mô hình."""
    n = len(X)
    p = len(beta) - 1
    
    # Tính y_hat và residuals
    y_hat = matmul(X, beta)
    residuals = [yi - yhi for yi, yhi in zip(y, y_hat)]
    
    # Tính Standardized Residuals
    rss = sum(r**2 for r in residuals)
    sigma_hat = math.sqrt(rss / (n - p - 1))
    
    # Tính Leverage (đường chéo ma trận Hat H = X(X'X)^-1 X')
    XT = transpose(X)
    XTX_inv = matrix_inverse(matmul(XT, X))
    H = matmul(X, matmul(XTX_inv, XT))
    leverage = [H[i][i] for i in range(n)]
    
    std_residuals = [r / (sigma_hat * math.sqrt(1 - h)) for r, h in zip(residuals, leverage)]

    # Vẽ biểu đồ
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Residuals vs Fitted
    axes[0, 0].scatter(y_hat, residuals, alpha=0.5)
    axes[0, 0].axhline(0, color='red', linestyle='--')
    axes[0, 0].set_title("Residuals vs Fitted")
    
    # 2. Normal Q-Q
    sorted_std_res = sorted(std_residuals)
    # Xấp xỉ phân vị chuẩn đơn giản
    theoretical_quantiles = [stats.norm.ppf((i + 0.5) / n) for i in range(n)]
    axes[0, 1].scatter(theoretical_quantiles, sorted_std_res, alpha=0.5)
    axes[0, 1].plot([-3, 3], [-3, 3], color='red', linestyle='--')
    axes[0, 1].set_title("Normal Q-Q")

    # 3. Scale-Location
    sqrt_abs_std_res = [math.sqrt(abs(r)) for r in std_residuals]
    axes[1, 0].scatter(y_hat, sqrt_abs_std_res, alpha=0.5)
    axes[1, 0].set_title("Scale-Location")

    # 4. Residuals vs Leverage
    axes[1, 1].scatter(leverage, std_residuals, alpha=0.5)
    axes[1, 1].set_title("Residuals vs Leverage")
    
    plt.tight_layout()
    plt.show()

# 2. TASK 9: MONTE CARLO GAUSS-MARKOV
def task_9_monte_carlo(n_sim = 1000):
    """Mô phỏng minh họa tính Unbiased và Efficiency của OLS."""
    n, true_b0, true_b1 = 50, 1.0, 2.5
    X_vals = [i * 0.2 for i in range(n)]
    X_matrix = [[1.0, x] for x in X_vals]
    XT = transpose(X_matrix)
    XTX_inv = matrix_inverse(matmul(XT, X_matrix))
    
    betas_ols = []
    
    for _ in range(n_sim):
        # Tạo y = 1.0 + 2.5*x + epsilon (epsilon ~ N(0, 1))
        y_sim = [true_b0 + true_b1 * x + random.gauss(0, 1) for x in X_vals]
        
        # Giải OLS: beta = (X'X)^-1 X'y
        XTy = matmul(XT, y_sim)
        beta_hat = matmul(XTX_inv, XTy)
        betas_ols.append(beta_hat[1]) # Lưu beta_1

    # Tính kỳ vọng E[beta_hat]
    expected_beta = sum(betas_ols) / n_sim
    print(f"--- Kết quả Monte Carlo ---")
    print(f"Beta_1 thật: {true_b1}")
    print(f"E[Beta_1_hat]: {expected_beta:.4f}")
    
    plt.hist(betas_ols, bins=30, edgecolor='black', alpha=0.7)
    plt.axvline(true_b1, color='red', label='True Beta')
    plt.title("Phân phối của Beta_1 (Minh họa Gauss-Markov)")
    plt.show()

# PHẦN TEST: CHẠY CẢ TASK 7 VÀ TASK 9
if __name__ == "__main__":
    print("--- Đang bắt đầu quá trình kiểm tra (Testing) ---")

    # --- TEST TASK 7 ---
    print("\n[1/2] Đang chạy Task 7: Vẽ biểu đồ chẩn đoán (Residual Plots)...")
    # Tạo dữ liệu giả lập có chủ đích (có nhiễu chuẩn)
    random.seed(42)
    x_test = [i * 0.5 for i in range(40)]
    # y = 2 + 3x + nhiễu
    y_test = [2 + 3 * xi + random.gauss(0, 1) for xi in x_test]
    
    # Chuẩn bị ma trận X (thêm cột 1 cho Intercept)
    X_matrix_test = [[1.0, xi] for xi in x_test]
    
    # Tính beta bằng cách giải hệ phương trình (X'X)beta = X'y
    XT_test = transpose(X_matrix_test)
    XTX_inv_test = matrix_inverse(matmul(XT_test, X_matrix_test))
    XTy_test = matmul(XT_test, y_test)
    beta_hat_test = matmul(XTX_inv_test, XTy_test)
    
    print(f"Hệ số ước lượng được: Beta_0 = {beta_hat_test[0]:.4f}, Beta_1 = {beta_hat_test[1]:.4f}")
    task_7_residual_analysis(X_matrix_test, y_test, beta_hat_test)
    print("\n[2/2] Đang chạy Task 9: Mô phỏng Monte Carlo (Gauss-Markov)...")
    # Chạy mô phỏng 1000 lần
    task_9_monte_carlo(n_sim=1000)

    print("\n--- Hoàn thành tất cả các bài test! ---")