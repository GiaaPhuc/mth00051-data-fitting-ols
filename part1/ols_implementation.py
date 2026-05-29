"""
OLS Implementation from Scratch
================================
Phần 1 — Đồ án 2: Data Fitting và Phương Pháp OLS
Môn: Toán Ứng Dụng và Thống Kê (MTH00051)

Cài đặt thuật toán OLS hoàn toàn từ đầu dựa trên công thức toán học,
không sử dụng numpy, scipy, hoặc sklearn.
"""

import math


# ---------------------------------------------------------------------------
# ĐẠI SỐ TUYẾN TÍNH THUẦN PYTHON
# Thay thế: numpy.array / .T / @ / linalg.inv / linalg.solve
# ---------------------------------------------------------------------------

def create_ones(n):
    """Trả về list n phần tử giá trị 1.0."""
    return [1.0] * n


def column_stack(ones_col, X):
    """
    Thêm cột toàn số 1 vào đầu ma trận X (list of lists).
    Thay thế np.column_stack([np.ones(n), X]).
    """
    return [[1.0] + row for row in X]


def transpose(A):
    """
    Chuyển vị ma trận 2D (m x n) -> (n x m).
    Thay thế A.T.
    """
    m = len(A)
    n = len(A[0])
    return [[A[i][j] for i in range(m)] for j in range(n)]


def mat_mul(A, B):
    """
    Nhân hai ma trận 2D: (m x k) @ (k x n) -> (m x n).
    Thay thế toán tử @.
    """
    m = len(A)
    k = len(B)
    n = len(B[0])
    C = [[0.0] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            s = 0.0
            for l in range(k):
                s += A[i][l] * B[l][j]
            C[i][j] = s
    return C


def mat_vec_mul(A, v):
    """
    Nhân ma trận A (m x n) với vector v (n,) -> (m,).
    Thay thế A @ v khi v là list 1D.
    """
    m = len(A)
    n = len(v)
    return [sum(A[i][j] * v[j] for j in range(n)) for i in range(m)]


def mat_inv(A):
    n = len(A)
    I_mat = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    aug = [A[i][:] + I_mat[i] for i in range(n)]
    for col in range(n):
        pivot_row = max(range(col, n), key=lambda r: abs(aug[r][col]))
        aug[col], aug[pivot_row] = aug[pivot_row], aug[col]
        pivot = aug[col][col] 
        if abs(pivot) < 1e-12:
            raise ValueError(f"Ma trận suy biến tại cột {col}, không thể nghịch đảo.")   
        aug[col] = [x / pivot for x in aug[col]]
        for row in range(n):
            if row != col:
                factor = aug[row][col]
                aug[row] = [aug[row][j] - factor * aug[col][j] for j in range(2 * n)]
    return [aug[i][n:] for i in range(n)]


def vec_sub(a, b):
    """Hiệu hai vector: a - b."""
    return [a[i] - b[i] for i in range(len(a))]


def vec_sum_sq(v):
    """Tổng bình phương: Σ vᵢ². Thay thế np.sum(v ** 2)."""
    return sum(x * x for x in v)


def mat_diag(A):
    """Trả về vector đường chéo chính của ma trận vuông. Thay thế np.diag(A)."""
    return [A[i][i] for i in range(len(A))]


def scalar_mat_mul(s, A):
    """Nhân vô hướng s với ma trận A. Thay thế s * A."""
    return [[s * A[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def mat_del_col(A, j):
    """Xoá cột j khỏi ma trận A. Thay thế np.delete(A, j, axis=1)."""
    return [[A[i][k] for k in range(len(A[0])) if k != j] for i in range(len(A))]


# ---------------------------------------------------------------------------
# PHÂN PHỐI THỐNG KÊ THUẦN PYTHON
# Thay thế: scipy.stats.t  và  scipy.stats.f
# ---------------------------------------------------------------------------

def _betacf(a, b, x):
    """
    Khai triển phân số liên tục cho hàm beta không hoàn chỉnh.
    Thuật toán Lentz chỉnh sửa (Numerical Recipes, §6.4).
    """
    MAXIT = 300
    EPS   = 3.0e-10
    FPMIN = 1.0e-300

    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d

    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d  = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c  = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d  = 1.0 / d
        h *= d * c

        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d  = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c  = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d  = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < EPS:
            break

    return h


def _betai(a, b, x):
    """
    Hàm beta không hoàn chỉnh chính quy I_x(a, b).
    Thay thế scipy.special.betainc(a, b, x).
    """
    if x < 0.0 or x > 1.0:
        raise ValueError("x phải nằm trong [0, 1].")
    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0
    bt = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log(1.0 - x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    else:
        return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def t_cdf(t_val, df):
    """
    CDF của phân phối t-Student: P(T ≤ t_val | df).
    Công thức: dùng I_x(df/2, 1/2) với x = df / (df + t²).
    Thay thế scipy.stats.t.cdf(t_val, df).
    """
    x     = df / (df + t_val * t_val)
    p_tail = _betai(df / 2.0, 0.5, x)
    if t_val >= 0:
        return 1.0 - 0.5 * p_tail
    else:
        return 0.5 * p_tail


def t_ppf(q, df):
    """
    Quantile (nghịch đảo CDF) của phân phối t-Student.
    Tìm t sao cho P(T ≤ t | df) = q bằng bisection.
    Thay thế scipy.stats.t.ppf(q, df).
    """
    if q <= 0.0:
        return -math.inf
    if q >= 1.0:
        return math.inf
    if q == 0.5:
        return 0.0

    lo, hi = -1000.0, 1000.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if t_cdf(mid, df) < q:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-10:
            break
    return (lo + hi) / 2.0


def f_cdf(x, dfn, dfd):
    """
    CDF của phân phối F tại điểm x với (dfn, dfd) bậc tự do.
    Công thức: I_z(dfn/2, dfd/2) với z = dfn*x / (dfn*x + dfd).
    Thay thế scipy.stats.f.cdf(x, dfn, dfd).
    """
    if x <= 0.0:
        return 0.0
    z = dfn * x / (dfn * x + dfd)
    return _betai(dfn / 2.0, dfd / 2.0, z)


# ---------------------------------------------------------------------------
# 1. OLS Fit
# ---------------------------------------------------------------------------

def ols_fit(X, y):
    """
    Ước lượng hệ số OLS và phương sai nhiễu.

    Công thức:
        β̂ = (XᵀX)⁻¹ Xᵀy                          (Normal Equations)
        σ̂² = RSS / (n − p − 1)                     (unbiased estimator)

    Tham số
    -------
    X : list of lists, shape (n, p+1)
        Ma trận design (đã bao gồm cột 1 cho intercept).
    y : list, shape (n,)
        Vector biến mục tiêu.

    Trả về
    ------
    dict với các khoá:
        beta_hat  : list (p+1,)  — vector hệ số ước lượng
        sigma2    : float        — ước lượng phương sai nhiễu σ̂²
        y_hat     : list (n,)    — giá trị fitted
        residuals : list (n,)    — phần dư ε̂ = y − ŷ
        rss       : float        — Residual Sum of Squares
    """
    n = len(X)
    k = len(X[0])   # k = p+1 (bao gồm cột intercept)
    p = k - 1

    Xt       = transpose(X)
    XtX      = mat_mul(Xt, X)
    epsilon = 1e-7
    for i in range(len(XtX)):
        XtX[i][i] += epsilon
    XtX_inv  = mat_inv(XtX)
    Xty      = mat_vec_mul(Xt, y)
    beta_hat = mat_vec_mul(XtX_inv, Xty)

    y_hat     = mat_vec_mul(X, beta_hat)
    residuals = vec_sub(y, y_hat)
    rss       = vec_sum_sq(residuals)
    sigma2    = rss / (n - p - 1)

    return {
        "beta_hat":  beta_hat,
        "sigma2":    sigma2,
        "y_hat":     y_hat,
        "residuals": residuals,
        "rss":       rss,
    }


# ---------------------------------------------------------------------------
# 2. Hat Matrix
# ---------------------------------------------------------------------------

def hat_matrix(X):
    """
    Tính Hat Matrix H = X(XᵀX)⁻¹Xᵀ và kiểm tra các tính chất.

    Tính chất (Mệnh đề 1.1):
        (i)   H² = H          (idempotent)
        (ii)  Hᵀ = H          (đối xứng)
        (iii) trace(H) = p + 1

    Tham số
    -------
    X : list of lists, shape (n, p+1)
        Ma trận thiết kế, đã bao gồm cột intercept.

    Trả về
    ------
    H : list of lists, shape (n, n)
    """
    n = len(X)
    k = len(X[0])

    Xt       = transpose(X)
    XtX      = mat_mul(Xt, X)
    XtX_inv  = mat_inv(XtX)
    middle   = mat_mul(X, XtX_inv)
    H        = mat_mul(middle, Xt)

    # Kiểm tra idempotent: H² = H
    H2 = mat_mul(H, H)
    for i in range(n):
        for j in range(n):
            assert abs(H2[i][j] - H[i][j]) < 1e-8, "Hat matrix khong thoa H^2 = H"

    # Kiểm tra đối xứng: Hᵀ = H
    Ht = transpose(H)
    for i in range(n):
        for j in range(n):
            assert abs(H[i][j] - Ht[i][j]) < 1e-8, "Hat matrix khong doi xung"

    # Kiểm tra trace(H) = rank = k = p+1
    actual_rank = round(sum(H[i][i] for i in range(n)))
    assert actual_rank == k, f"rank(H) = {actual_rank}, ky vong {k}"

    return H


# ---------------------------------------------------------------------------
# 3. Model Metrics
# ---------------------------------------------------------------------------

def model_metrics(y, y_hat, p):
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
    y     : list (n,) — giá trị thực
    y_hat : list (n,) — giá trị fitted
    p     : int       — số biến dự báo (không tính intercept)

    Trả về
    ------
    dict: rss, tss, r2, r2_adj, f_stat, f_pvalue, n
    """
    n      = len(y)
    mean_y = sum(y) / n

    rss = sum((y[i] - y_hat[i]) ** 2 for i in range(n))
    tss = sum((y[i] - mean_y)    ** 2 for i in range(n))

    r2     = 1.0 - rss / tss
    r2_adj = 1.0 - (n - 1) / (n - p - 1) * (1.0 - r2)

    rss_denom = rss / (n - p - 1)
    if p == 0 or rss_denom < 1e-15:
        f_stat   = math.inf if rss_denom < 1e-15 else math.nan
        f_pvalue = 0.0      if rss_denom < 1e-15 else math.nan
    else:
        f_stat   = ((tss - rss) / p) / rss_denom
        f_pvalue = 1.0 - f_cdf(f_stat, dfn=p, dfd=n - p - 1)

    return {
        "n":        n,
        "p":        p,
        "rss":      rss,
        "tss":      tss,
        "r2":       r2,
        "r2_adj":   r2_adj,
        "f_stat":   f_stat,
        "f_pvalue": f_pvalue,
    }


# ---------------------------------------------------------------------------
# 4. Coefficient Inference
# ---------------------------------------------------------------------------

def coef_inference(X, y, beta_hat, sigma2, alpha=0.05):
    """
    Suy luận thống kê cho từng hệ số hồi quy.

    Công thức:
        Var(β̂ | X) = σ²(XᵀX)⁻¹
        SE(β̂ⱼ)    = σ̂ · √[(XᵀX)⁻¹]ⱼⱼ
        tⱼ         = β̂ⱼ / SE(β̂ⱼ)  ~  t_{n−p−1}
        CI₉₅%      = β̂ⱼ ± t_{α/2, n−p−1} · SE(β̂ⱼ)

    Tham số
    -------
    X        : list of lists (n, p+1)
    y        : list (n,)
    beta_hat : list (p+1,)
    sigma2   : float  — σ̂² từ ols_fit
    alpha    : float  — mức ý nghĩa (mặc định 0.05)

    Trả về
    ------
    dict: se, t_stats, p_values, ci_lower, ci_upper, cov_beta
    """
    n = len(X)
    k = len(X[0])
    p  = k - 1
    df = n - p - 1

    Xt       = transpose(X)
    XtX      = mat_mul(Xt, X)
    XtX_inv  = mat_inv(XtX)
    cov_beta = scalar_mat_mul(sigma2, XtX_inv)

    diag     = mat_diag(cov_beta)
    se       = [math.sqrt(d) for d in diag]

    t_stats  = [beta_hat[j] / se[j] for j in range(k)]
    p_values = [2.0 * (1.0 - t_cdf(abs(t_stats[j]), df=df)) for j in range(k)]

    t_crit   = t_ppf(1.0 - alpha / 2, df=df)
    ci_lower = [beta_hat[j] - t_crit * se[j] for j in range(k)]
    ci_upper = [beta_hat[j] + t_crit * se[j] for j in range(k)]

    return {
        "beta_hat": beta_hat,
        "se":       se,
        "t_stats":  t_stats,
        "p_values": p_values,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "cov_beta": cov_beta,
        "df":       df,
        "alpha":    alpha,
    }


# ---------------------------------------------------------------------------
# 5. Variance Inflation Factor (VIF)
# ---------------------------------------------------------------------------

def vif(X):
    """
    Tính Variance Inflation Factor (VIF) cho từng biến dự báo.

    Công thức:
        VIFⱼ = 1 / (1 − R²ⱼ)

    với R²ⱼ là R² khi hồi quy biến Xⱼ theo tất cả các biến còn lại.
    VIF > 10 cho thấy đa cộng tuyến nghiêm trọng.

    Tham số
    -------
    X : list of lists, shape (n, p+1)
        Ma trận design (cột đầu là intercept toàn 1).
        Chỉ tính VIF cho p cột biến dự báo (bỏ cột intercept).

    Trả về
    ------
    vif_values : list (p,) — VIF của từng biến dự báo (index 1..p)
    """
    n = len(X)
    k = len(X[0])
    p = k - 1

    # Lấy chỉ các cột biến dự báo (bỏ cột intercept)
    predictors = [row[1:] for row in X]

    vif_values = []
    for j in range(p):
        y_j = [predictors[i][j] for i in range(n)]
        X_j = mat_del_col(predictors, j)

        # Khi chỉ có 1 predictor, sau khi xoá không còn biến nào để hồi quy
        if not X_j or not X_j[0]:
            vif_values.append(1.0)
            continue

        X_j_design = column_stack(create_ones(n), X_j)
        result     = ols_fit(X_j_design, y_j)
        p_j        = len(X_j[0])
        metrics    = model_metrics(y_j, result["y_hat"], p=p_j)
        r2_j       = metrics["r2"]

        if r2_j >= 1.0 - 1e-10:
            vif_values.append(math.inf)
        else:
            vif_values.append(1.0 / (1.0 - r2_j))

    return vif_values


# ---------------------------------------------------------------------------
# UNIT TESTS
# ---------------------------------------------------------------------------

def test_ols_fit():
    # Test 1: Simple linear relation y = 1 + 2*x
    X = [[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]]
    y = [3.0, 5.0, 7.0]
    res = ols_fit(X, y)
    assert abs(res["beta_hat"][0] - 1.0) < 1e-7
    assert abs(res["beta_hat"][1] - 2.0) < 1e-7
    assert abs(res["sigma2"]) < 1e-7
    
    # Test 2: Perfect fit with multiple features (y = 1 + 2*x1 + 1*x2)
    X = [[1.0, 1.0, 2.0], [1.0, 2.0, 1.0], [1.0, 3.0, 3.0], [1.0, 4.0, 1.0]]
    y = [5.0, 6.0, 10.0, 10.0]
    res = ols_fit(X, y)
    assert abs(res["beta_hat"][0] - 1.0) < 1e-5
    assert abs(res["beta_hat"][1] - 2.0) < 1e-5
    assert abs(res["beta_hat"][2] - 1.0) < 1e-5
    print("test_ols_fit PASSED")




def test_hat_matrix():
    # Test 1: Identity/projection matrix for single variable
    X = [[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]]
    H = hat_matrix(X)
    assert abs(H[0][0] - 5/6) < 1e-7
    assert abs(H[1][1] - 1/3) < 1e-7
    assert abs(H[2][2] - 5/6) < 1e-7
    
    # Test 2: Projection onto 1D constant vector
    X2 = [[1.0], [1.0]]
    H2 = hat_matrix(X2)
    assert abs(H2[0][0] - 0.5) < 1e-7
    assert abs(H2[0][1] - 0.5) < 1e-7
    print("test_hat_matrix PASSED")


def test_model_metrics():
    # Test 1: Perfect fit R^2 = 1.0
    y = [1.0, 2.0, 3.0]
    y_hat = [1.0, 2.0, 3.0]
    res = model_metrics(y, y_hat, p=1)
    assert abs(res["r2"] - 1.0) < 1e-7
    assert abs(res["rss"]) < 1e-7
    
    # Test 2: Known values (TSS=2.0, RSS=0.5, R2=0.75, R2_adj=0.5)
    y = [1.0, 2.0, 3.0]
    y_hat = [1.5, 2.0, 2.5]
    res = model_metrics(y, y_hat, p=1)
    assert abs(res["r2"] - 0.75) < 1e-7
    assert abs(res["rss"] - 0.5) < 1e-7
    assert abs(res["tss"] - 2.0) < 1e-7
    assert abs(res["r2_adj"] - 0.5) < 1e-7
    print("test_model_metrics PASSED")


def test_coef_inference():
    # Test 1: Simple linear relation with small noise variance
    X = [[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]]
    y = [3.0, 5.0, 7.0]
    beta_hat = [1.0, 2.0]
    sigma2 = 0.01
    res = coef_inference(X, y, beta_hat, sigma2)
    assert abs(res["se"][0] - math.sqrt(0.07 / 3.0)) < 1e-7
    assert abs(res["se"][1] - math.sqrt(0.005)) < 1e-7

    
    # Test 2: Known variance case with noise
    X = [[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]]
    y = [2.9, 5.2, 6.9]
    res = coef_inference(X, y, [1.0, 2.0], 0.06)
    assert abs(res["se"][0] - math.sqrt(0.14)) < 1e-7
    assert abs(res["se"][1] - math.sqrt(0.03)) < 1e-7
    print("test_coef_inference PASSED")


def test_vif():
    # Test 1: Orthogonal predictors -> VIF = 1.0
    X = [[1.0, 1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 0.0, 1.0], [1.0, 0.0, -1.0]]
    vif_vals = vif(X)
    assert abs(vif_vals[0] - 1.0) < 1e-7
    assert abs(vif_vals[1] - 1.0) < 1e-7
    
    # Test 2: Heavily correlated predictors -> VIF > 10.0
    X = [[1.0, 1.0, 1.01], [1.0, 2.0, 2.01], [1.0, 3.0, 3.01]]
    vif_vals = vif(X)
    assert vif_vals[0] > 10.0
    assert vif_vals[1] > 10.0
    print("test_vif PASSED")


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass
    print("--- RUNNING CORE UNIT TESTS (ols_implementation.py) ---")
    test_ols_fit()
    test_hat_matrix()
    test_model_metrics()
    test_coef_inference()
    test_vif()

    print("---------------------------------------------------------")

