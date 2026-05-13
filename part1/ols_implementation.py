"""
OLS Implementation from Scratch
================================
Phần 1 — Đồ án 2: Data Fitting và Phương Pháp OLS
Môn: Toán Ứng Dụng và Thống Kê (MTH00051)

Cài đặt thuật toán OLS hoàn toàn từ đầu dựa trên công thức toán học,
không sử dụng sklearn.linear_model hay numpy.linalg.lstsq để thay thế.
"""

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# 1. OLS Fit
# ---------------------------------------------------------------------------

def ols_fit(X: np.ndarray, y: np.ndarray) -> dict:
    """
    Ước lượng hệ số OLS và phương sai nhiễu.

    Công thức:
        β̂ = (XᵀX)⁻¹ Xᵀy                          (Normal Equations)
        σ̂² = RSS / (n − p − 1)                     (unbiased estimator)

    Tham số
    -------
    X : ndarray, shape (n, p+1)
        Ma trận design (đã bao gồm cột 1 cho intercept).
    y : ndarray, shape (n,)
        Vector biến mục tiêu.

    Trả về
    ------
    dict với các khoá:
        beta_hat  : ndarray (p+1,)  — vector hệ số ước lượng
        sigma2    : float           — ước lượng phương sai nhiễu σ̂²
        y_hat     : ndarray (n,)    — giá trị fitted
        residuals : ndarray (n,)    — phần dư ε̂ = y − ŷ
        rss       : float           — Residual Sum of Squares
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    n, k = X.shape          # n observations, k = p+1 columns (incl. intercept)
    p = k - 1               # number of predictors (excl. intercept)

    XtX = X.T @ X
    Xty = X.T @ y

    # β̂ = (XᵀX)⁻¹ Xᵀy
    beta_hat = np.linalg.solve(XtX, Xty)

    y_hat = X @ beta_hat
    residuals = y - y_hat
    rss = float(residuals @ residuals)

    # σ̂² = RSS / (n − p − 1)
    sigma2 = rss / (n - p - 1)

    return {
        "beta_hat": beta_hat,
        "sigma2": sigma2,
        "y_hat": y_hat,
        "residuals": residuals,
        "rss": rss,
    }


# ---------------------------------------------------------------------------
# 2. Hat Matrix
# ---------------------------------------------------------------------------

def hat_matrix(X: np.ndarray) -> np.ndarray:
    """
    Tính Hat Matrix H = X(XᵀX)⁻¹Xᵀ và kiểm tra các tính chất.

    Dùng QR decomposition thay vì nghịch đảo trực tiếp để ổn định số học:
        X = QR  =>  H = QQᵀ

    Tính chất kiểm tra (Mệnh đề 1.1):
        (i)   H² = H          (idempotent)
        (ii)  Hᵀ = H          (đối xứng)
        (iii) rank(H) = p + 1
        (iv)  eigenvalues ∈ {0, 1}

    Tham số
    -------
    X : ndarray, shape (n, p+1)
        Ma trận thiết kế, đã bao gồm cột intercept.

    Trả về
    ------
    H : ndarray, shape (n, n)
    """
    X = np.asarray(X, dtype=float)

    Q, _ = np.linalg.qr(X)
    H = Q @ Q.T

    assert np.allclose(H @ H, H, atol=1e-8), "Hat matrix khong thoa H^2 = H"
    assert np.allclose(H, H.T, atol=1e-8), "Hat matrix khong doi xung"

    expected_rank = X.shape[1]
    actual_rank = int(np.round(np.trace(H)))
    assert actual_rank == expected_rank, (
        f"rank(H) = {actual_rank}, ky vong {expected_rank}"
    )

    eigenvalues = np.linalg.eigvalsh(H)
    assert np.all(
        (eigenvalues < 1e-8) | (eigenvalues > 1 - 1e-8)
    ), "Gia tri rieng cua H nam ngoai {0, 1}"

    return H


# ---------------------------------------------------------------------------
# 3. Model Metrics
# ---------------------------------------------------------------------------

def model_metrics(y: np.ndarray, y_hat: np.ndarray, p: int) -> dict:
    """
    Tính các chỉ số đánh giá mô hình hồi quy.

    Công thức:
        RSS = Σ(yᵢ − ŷᵢ)²
        TSS = Σ(yᵢ − ȳ)²
        R²  = 1 − RSS/TSS
        R̄²  = 1 − (n−1)/(n−p−1) · (1 − R²)
        F   = [(TSS − RSS)/p] / [RSS/(n−p−1)]  ~ F_{p, n−p−1}

    Tham số
    -------
    y     : ndarray (n,) — giá trị thực
    y_hat : ndarray (n,) — giá trị fitted
    p     : int          — số biến dự báo (không tính intercept)

    Trả về
    ------
    dict: rss, tss, r2, r2_adj, f_stat, f_pvalue, n
    """
    y = np.asarray(y, dtype=float)
    y_hat = np.asarray(y_hat, dtype=float)
    n = len(y)

    rss = float(np.sum((y - y_hat) ** 2))
    tss = float(np.sum((y - np.mean(y)) ** 2))

    r2 = 1.0 - rss / tss
    r2_adj = 1.0 - (n - 1) / (n - p - 1) * (1.0 - r2)

    # F-statistic: H₀: β₁ = … = βₚ = 0
    rss_denominator = rss / (n - p - 1)
    if rss_denominator < 1e-15:
        # Perfect fit: F -> infinity, p-value -> 0
        f_stat = np.inf
        f_pvalue = 0.0
    else:
        f_stat = ((tss - rss) / p) / rss_denominator
        f_pvalue = float(1.0 - stats.f.cdf(f_stat, dfn=p, dfd=n - p - 1))

    return {
        "n": n,
        "p": p,
        "rss": rss,
        "tss": tss,
        "r2": r2,
        "r2_adj": r2_adj,
        "f_stat": f_stat,
        "f_pvalue": f_pvalue,
    }


# ---------------------------------------------------------------------------
# 4. Coefficient Inference
# ---------------------------------------------------------------------------

def coef_inference(
    X: np.ndarray,
    y: np.ndarray,
    beta_hat: np.ndarray,
    sigma2: float,
    alpha: float = 0.05,
) -> dict:
    """
    Suy luận thống kê cho từng hệ số hồi quy.

    Công thức:
        Var(β̂ | X) = σ²(XᵀX)⁻¹
        SE(β̂ⱼ)    = σ̂ · √[(XᵀX)⁻¹]ⱼⱼ
        tⱼ         = β̂ⱼ / SE(β̂ⱼ)  ~  t_{n−p−1}
        CI₉₅%      = β̂ⱼ ± t_{α/2, n−p−1} · SE(β̂ⱼ)

    Tham số
    -------
    X        : ndarray (n, p+1)
    y        : ndarray (n,)
    beta_hat : ndarray (p+1,)
    sigma2   : float             — σ̂² từ ols_fit
    alpha    : float             — mức ý nghĩa (mặc định 0.05)

    Trả về
    ------
    dict: se, t_stats, p_values, ci_lower, ci_upper, cov_beta
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    beta_hat = np.asarray(beta_hat, dtype=float)

    n, k = X.shape
    p = k - 1
    df = n - p - 1

    XtX_inv = np.linalg.inv(X.T @ X)
    cov_beta = sigma2 * XtX_inv                          # Var(β̂) = σ²(XᵀX)⁻¹
    se = np.sqrt(np.diag(cov_beta))                      # standard errors

    t_stats = beta_hat / se
    p_values = 2.0 * (1.0 - stats.t.cdf(np.abs(t_stats), df=df))

    t_crit = stats.t.ppf(1.0 - alpha / 2, df=df)
    ci_lower = beta_hat - t_crit * se
    ci_upper = beta_hat + t_crit * se

    return {
        "beta_hat": beta_hat,
        "se": se,
        "t_stats": t_stats,
        "p_values": p_values,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "cov_beta": cov_beta,
        "df": df,
        "alpha": alpha,
    }


# ---------------------------------------------------------------------------
# 5. Variance Inflation Factor (VIF)
# ---------------------------------------------------------------------------

def vif(X: np.ndarray) -> np.ndarray:
    """
    Tính Variance Inflation Factor (VIF) cho từng biến dự báo.

    Công thức:
        VIFⱼ = 1 / (1 − R²ⱼ)

    với R²ⱼ là R² khi hồi quy biến Xⱼ theo tất cả các biến còn lại.
    VIF > 10 cho thấy đa cộng tuyến nghiêm trọng.

    Tham số
    -------
    X : ndarray, shape (n, p+1)
        Ma trận design (cột đầu là intercept toàn 1).
        Chỉ tính VIF cho p cột biến dự báo (bỏ cột intercept).

    Trả về
    ------
    vif_values : ndarray (p,)  — VIF của từng biến dự báo (index 1..p)
    """
    X = np.asarray(X, dtype=float)
    n, k = X.shape

    # Bỏ cột intercept (cột 0), lấy các cột biến dự báo
    predictors = X[:, 1:]
    p = predictors.shape[1]

    vif_values = np.zeros(p)
    for j in range(p):
        # Xⱼ là biến mục tiêu, phần còn lại là features
        y_j = predictors[:, j]
        X_j = np.delete(predictors, j, axis=1)

        # Thêm intercept
        X_j_with_const = np.column_stack([np.ones(n), X_j])

        result = ols_fit(X_j_with_const, y_j)
        metrics = model_metrics(y_j, result["y_hat"], p=X_j.shape[1])
        r2_j = metrics["r2"]

        # Tránh chia cho 0 khi R² = 1 (perfect multicollinearity)
        if r2_j >= 1.0 - 1e-10:
            vif_values[j] = np.inf
        else:
            vif_values[j] = 1.0 / (1.0 - r2_j)

    return vif_values

