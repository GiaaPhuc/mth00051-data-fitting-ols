"""
Advanced Methods — Part 2 (Bonus +0.5đ)
=========================================
Cài đặt hai kỹ thuật nâng cao hoàn toàn từ đầu (không dùng sklearn):
  1. Kernel Ridge Regression  — kernel trick + RBF/Polynomial, λ và ℓ chọn qua CV.
  2. Bayesian Linear Regression — conjugate posterior, có uncertainty quantification.

Interface tích hợp với model_comparison.py:
  - compare_advanced_models() trả về pd.DataFrame cùng cột MAE / RMSE / R2,
    dễ dàng pd.concat với kết quả compare_models() để so sánh tổng hợp.

Công thức tham chiếu: Đề đồ án §2.4, trang 11–13.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# PATH SETUP — đồng nhất với model_comparison.py
# ---------------------------------------------------------------------------
_PART2 = Path(__file__).resolve().parent          # .../part2/
_ROOT  = _PART2.parent                            # .../Group_<ID>/

for _p in [str(_ROOT), str(_PART2)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ---------------------------------------------------------------------------
# ĐẠI SỐ TUYẾN TÍNH THUẦN PYTHON
# (Tất cả thuật toán cốt lõi không dùng numpy)
# ---------------------------------------------------------------------------

def _transpose(A: list[list[float]]) -> list[list[float]]:
    """Chuyển vị ma trận 2D: (m×n) → (n×m)."""
    m, n = len(A), len(A[0])
    return [[A[i][j] for i in range(m)] for j in range(n)]


def _mat_mul(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    """Nhân hai ma trận 2D: (m×k) @ (k×n) → (m×n)."""
    m, k, n = len(A), len(B), len(B[0])
    C = [[0.0] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            s = 0.0
            for l in range(k):
                s += A[i][l] * B[l][j]
            C[i][j] = s
    return C


def _mat_vec_mul(A: list[list[float]], v: list[float]) -> list[float]:
    """Nhân ma trận A (m×n) với vector v (n,) → (m,)."""
    return [sum(A[i][j] * v[j] for j in range(len(v))) for i in range(len(A))]


def _mat_inv(A: list[list[float]]) -> list[list[float]]:
    """
    Nghịch đảo ma trận vuông bằng Gauss–Jordan với partial pivoting.
    Raises ValueError nếu ma trận suy biến.
    """
    n = len(A)
    aug = [A[i][:] + [1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for col in range(n):
        pivot_row = max(range(col, n), key=lambda r: abs(aug[r][col]))
        aug[col], aug[pivot_row] = aug[pivot_row], aug[col]
        pivot = aug[col][col]
        if abs(pivot) < 1e-14:
            raise ValueError(f"Ma trận suy biến tại cột {col}, không thể nghịch đảo.")
        aug[col] = [x / pivot for x in aug[col]]
        for row in range(n):
            if row != col:
                f = aug[row][col]
                aug[row] = [aug[row][j] - f * aug[col][j] for j in range(2 * n)]
    return [aug[i][n:] for i in range(n)]


def _mat_add(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    """Cộng hai ma trận cùng kích thước."""
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def _scalar_mat(s: float, A: list[list[float]]) -> list[list[float]]:
    """Nhân vô hướng s với ma trận A."""
    return [[s * A[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def _vec_sub(a: list[float], b: list[float]) -> list[float]:
    """Hiệu hai vector: a - b."""
    return [a[i] - b[i] for i in range(len(a))]


def _sq_euclidean(x: list[float], z: list[float]) -> float:
    """Tính ||x - z||^2 (bình phương khoảng cách Euclidean)."""
    return sum((x[i] - z[i]) ** 2 for i in range(len(x)))


# ---------------------------------------------------------------------------
# HELPERS I/O — dùng numpy chỉ để chuyển đổi kiểu dữ liệu
# ---------------------------------------------------------------------------

def _to_list_2d(X: pd.DataFrame | np.ndarray) -> list[list[float]]:
    if isinstance(X, pd.DataFrame):
        return X.values.tolist()
    return np.asarray(X, dtype=float).tolist()


def _to_list_1d(y: pd.Series | np.ndarray | Sequence[float]) -> list[float]:
    if isinstance(y, pd.Series):
        return y.values.ravel().tolist()
    return np.asarray(y, dtype=float).ravel().tolist()


def _compute_metrics(y_true: list[float], y_pred: list[float]) -> dict[str, float]:
    """
    Tính MAE, RMSE, R^2 trên tập test (công thức đề §2.3.3).

    MAE  = (1/n) Sum(|y_i - y_hat_i|)
    RMSE = sqrt[(1/n) Sum((y_i - y_hat_i)^2)]
    R2   = 1 - RSS / TSS
    """
    n = len(y_true)
    errors = [y_true[i] - y_pred[i] for i in range(n)]
    mae = sum(abs(e) for e in errors) / n
    rmse = math.sqrt(sum(e * e for e in errors) / n)
    mean_y = sum(y_true) / n
    tss = sum((yi - mean_y) ** 2 for yi in y_true)
    rss = sum(e * e for e in errors)
    r2 = 1.0 - rss / tss if tss > 1e-15 else float("nan")
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


# ===========================================================================
# PHẦN 1: KERNEL RIDGE REGRESSION
# ===========================================================================

def rbf_kernel(x: list[float], z: list[float], ell: float) -> float:
    """
    Kernel RBF (Radial Basis Function / Gaussian kernel):

        k_RBF(x, z) = exp(-||x - z||^2 / (2 * l^2))    (Đề §2.4, eq. 20)

    Args:
        x, z : Hai vector đặc trưng (list 1D, không có intercept).
        ell  : Length-scale ℓ > 0 — kiểm soát độ rộng vùng ảnh hưởng.

    Returns:
        float ∈ (0, 1]: 1 khi x = z, tiệm cận 0 khi hai điểm rất xa.
    """
    return math.exp(-_sq_euclidean(x, z) / (2.0 * ell * ell))


def poly_kernel(x: list[float], z: list[float], degree: int = 2, c: float = 1.0) -> float:
    """
    Polynomial kernel:

        k_poly(x, z) = (x·z + c)^degree

    Args:
        x, z   : Hai vector đặc trưng.
        degree : Bậc đa thức (mặc định 2 — bậc hai).
        c      : Hằng số dịch chuyển (mặc định 1).

    Returns:
        float: Giá trị kernel.
    """
    dot = sum(xi * zi for xi, zi in zip(x, z))
    return (dot + c) ** degree


def gram_matrix(
    X: list[list[float]],
    Z: list[list[float]] | None = None,
    kernel: str = "rbf",
    ell: float = 1.0,
    degree: int = 2,
    c: float = 1.0,
) -> list[list[float]]:
    """
    Tính ma trận Gram K ∈ ℝ^{n×m} với K_{ij} = k(X_i, Z_j).

    Khi Z = None, tính K = k(X, X) là ma trận đối xứng (n×n).

    Args:
        X      : Ma trận đặc trưng (n × p), KHÔNG có intercept.
        Z      : Ma trận đặc trưng thứ hai (m × p) hoặc None.
        kernel : 'rbf' hoặc 'poly'.
        ell    : Length-scale cho RBF.
        degree : Bậc cho polynomial kernel.
        c      : Hằng số cho polynomial kernel.

    Returns:
        list[list[float]]: Ma trận Gram (n × m).
    """
    if Z is None:
        Z = X
    n, m = len(X), len(Z)
    K = [[0.0] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            if kernel == "rbf":
                K[i][j] = rbf_kernel(X[i], Z[j], ell)
            elif kernel == "poly":
                K[i][j] = poly_kernel(X[i], Z[j], degree, c)
            else:
                raise ValueError("kernel phải là 'rbf' hoặc 'poly'.")
    return K


def kernel_ridge_fit(
    X_train: list[list[float]],
    y_train: list[float],
    lam: float = 1.0,
    kernel: str = "rbf",
    ell: float = 1.0,
    degree: int = 2,
    c: float = 1.0,
) -> dict:
    """
    Huấn luyện Kernel Ridge Regression theo dạng dual.

    Công thức (Đề §2.4, eq. 19):
        alpha = (K + lambda * I)^-1 * y           — hệ số dual
        y_hat(x*) = k(x*)^T * alpha               — dự đoán

    Với K_{ij} = k(x_i, x_j) là ma trận Gram, lambda * I là regularization.

    Args:
        X_train : Ma trận đặc trưng train (n × p), không có intercept.
        y_train : Vector nhãn (n,).
        lam     : Hệ số chính quy hóa λ > 0.
        kernel  : 'rbf' hoặc 'poly'.
        ell     : Length-scale cho RBF.
        degree  : Bậc cho polynomial kernel.
        c       : Hằng số cho polynomial kernel.

    Returns:
        dict:
            alpha         : list (n,) — hệ số dual alpha = (K + lambda * I)^-1 y.
            K             : Ma trận Gram train (n × n).
            X_train       : Dữ liệu train (lưu để predict).
            y_hat         : list (n,) — fitted values.
            residuals     : list (n,).
            kernel_params : dict các tham số kernel.
            lam           : float.
    """
    n = len(X_train)

    # Tính ma trận Gram K (n × n)
    K = gram_matrix(X_train, kernel=kernel, ell=ell, degree=degree, c=c)

    # K + λI
    KlI = [[K[i][j] + (lam if i == j else 0.0) for j in range(n)] for i in range(n)]

    # alpha = (K + lambda * I)^-1 y
    KlI_inv = _mat_inv(KlI)
    alpha = _mat_vec_mul(KlI_inv, y_train)

    # Fitted values: y_hat = K * alpha
    y_hat = [sum(K[i][j] * alpha[j] for j in range(n)) for i in range(n)]
    residuals = _vec_sub(y_train, y_hat)

    return {
        "alpha": alpha,
        "K": K,
        "X_train": X_train,
        "y_hat": y_hat,
        "residuals": residuals,
        "kernel_params": {"kernel": kernel, "ell": ell, "degree": degree, "c": c},
        "lam": lam,
    }


def kernel_ridge_predict(
    fit_result: dict,
    X_test: list[list[float]],
) -> list[float]:
    """
    Dự đoán với Kernel Ridge Regression đã huấn luyện.

    Công thức: y_hat(x*) = k(x*, X_train)^T * alpha
    với k_i(x*) = k(x*, x_i) cho mỗi điểm train x_i.

    Args:
        fit_result : Kết quả từ kernel_ridge_fit().
        X_test     : Ma trận đặc trưng test (m × p), không có intercept.

    Returns:
        list[float]: Vector dự đoán y_hat (m,).
    """
    alpha = fit_result["alpha"]
    X_train = fit_result["X_train"]
    p = fit_result["kernel_params"]

    # K_test[j][i] = k(X_test_j, X_train_i) → ma trận (m × n)
    K_test = gram_matrix(
        X_test, X_train,
        kernel=p["kernel"], ell=p["ell"], degree=p["degree"], c=p["c"],
    )
    return [
        sum(K_test[j][i] * alpha[i] for i in range(len(alpha)))
        for j in range(len(X_test))
    ]


def tune_kernel_cv(
    X: list[list[float]],
    y: list[float],
    kernel: str = "rbf",
    lambdas: Sequence[float] | None = None,
    ells: Sequence[float] | None = None,
    k: int = 5,
    seed: int = 42,
) -> tuple[float, float, float]:
    """
    Chọn siêu tham số (λ, ℓ) cho Kernel Ridge bằng k-fold CV (MSE trung bình).

    Với kernel='poly', ells bị bỏ qua và chỉ tune λ.

    Args:
        X       : Ma trận đặc trưng (n × p), không có intercept.
        y       : Vector nhãn (n,).
        kernel  : 'rbf' hoặc 'poly'.
        lambdas : Lưới λ (mặc định 10^{-2}…10^{2}).
        ells    : Lưới ℓ cho RBF (mặc định [0.1, 0.5, 1.0, 2.0, 5.0]).
        k       : Số fold.
        seed    : Random seed để tái lập kết quả.

    Returns:
        (best_lambda, best_ell, best_cv_mse)
    """
    if lambdas is None:
        lambdas = [10.0 ** e for e in range(-2, 3)]   # 0.01 → 100
    if ells is None:
        ells = [0.1, 0.5, 1.0, 2.0, 5.0] if kernel == "rbf" else [1.0]

    n = len(X)
    rng = np.random.default_rng(seed)
    order = rng.permutation(n).tolist()

    # Chia thành k fold
    base = n // k
    folds: list[list[int]] = []
    start = 0
    for i in range(k):
        end = start + base + (1 if i < n % k else 0)
        folds.append(order[start:end])
        start = end

    best_lam, best_ell, best_mse = float(lambdas[0]), float(ells[0]), float("inf")

    for lam in lambdas:
        for ell in ells:
            fold_mses: list[float] = []

            for fold_idx in range(k):
                val_set = set(folds[fold_idx])
                tr_idx  = [i for i in range(n) if i not in val_set]
                val_idx = list(val_set)

                X_tr  = [X[i] for i in tr_idx]
                y_tr  = [y[i] for i in tr_idx]
                X_val = [X[i] for i in val_idx]
                y_val = [y[i] for i in val_idx]

                fit     = kernel_ridge_fit(X_tr, y_tr, lam=float(lam), kernel=kernel, ell=float(ell))
                y_pred  = kernel_ridge_predict(fit, X_val)
                mse_i   = sum((y_val[i] - y_pred[i]) ** 2 for i in range(len(y_val))) / len(y_val)
                fold_mses.append(mse_i)

            cv_mse = sum(fold_mses) / k
            if cv_mse < best_mse:
                best_mse = cv_mse
                best_lam = float(lam)
                best_ell = float(ell)

    return best_lam, best_ell, best_mse


# ===========================================================================
# PHẦN 2: BAYESIAN LINEAR REGRESSION
# ===========================================================================

def _estimate_sigma2_ols(X: list[list[float]], y: list[float]) -> float:
    """
    Ước lượng phương sai nhiễu sigma^2 từ OLS để làm noise prior cho Bayesian LR.
        sigma_hat^2 = RSS / (n - p - 1)

    Fallback về Var(y) nếu ma trận suy biến.
    """
    n, k = len(X), len(X[0])
    Xt = _transpose(X)
    try:
        XtX_inv = _mat_inv(_mat_mul(Xt, X))
    except ValueError:
        mean_y = sum(y) / n
        return sum((yi - mean_y) ** 2 for yi in y) / max(n - 1, 1)
    beta = _mat_vec_mul(XtX_inv, _mat_vec_mul(Xt, y))
    y_hat = _mat_vec_mul(X, beta)
    rss = sum((y[i] - y_hat[i]) ** 2 for i in range(n))
    return rss / max(n - k, 1)


def bayesian_lr_fit(
    X: list[list[float]],
    y: list[float],
    sigma2: float,
    m0: list[float] | None = None,
    S0: list[list[float]] | None = None,
    tau2: float = 100.0,
) -> dict:
    """
    Cập nhật phân phối posterior cho Bayesian Linear Regression.

    Prior (Gaussian conjugate, Đề §2.4 eq. 21):
        beta ~ N(m_0, S_0)

    Likelihood:
        y | X, beta ~ N(X * beta, sigma^2 * I)

    Posterior (Đề §2.4 eq. 22–23):
        S_n = (S_0^-1 + (1/sigma^2) X^T X)^-1
        m_n = S_n (S_0^-1 m_0 + (1/sigma^2) X^T y)

    Ghi chú: Với prior non-informative (tau^2 -> ∞), m_n hội tụ về ước lượng OLS.

    Args:
        X      : Ma trận design (n × (p+1)), đã có cột intercept.
        y      : Vector nhãn (n,).
        sigma2 : Phương sai nhiễu sigma^2 (đã biết hoặc ước lượng từ OLS).
        m0     : Prior mean (p+1,). Mặc định = vector 0.
        S0     : Prior covariance (p+1 × p+1). Mặc định = tau^2 * I.
        tau2   : Phương sai prior khi S0 = None (giá trị lớn = non-informative).

    Returns:
        dict:
            mn        : list (p+1,) — posterior mean (điểm ước lượng beta).
            Sn        : list[list] (p+1 × p+1) — posterior covariance.
            m0, S0    : Prior.
            sigma2    : float.
            y_hat     : list (n,) — fitted values dùng m_n.
            residuals : list (n,).
    """
    n, k = len(X), len(X[0])

    # Prior mặc định: non-informative Gaussian
    if m0 is None:
        m0 = [0.0] * k
    if S0 is None:
        S0 = [[tau2 if i == j else 0.0 for j in range(k)] for i in range(k)]

    S0_inv = _mat_inv(S0)
    Xt = _transpose(X)
    XtX = _mat_mul(Xt, X)
    Xty = _mat_vec_mul(Xt, y)

    # S_n = (S_0^-1 + (1/sigma^2) X^T X)^-1
    A  = _mat_add(S0_inv, _scalar_mat(1.0 / sigma2, XtX))
    Sn = _mat_inv(A)

    # m_n = S_n (S_0^-1 m_0 + (1/sigma^2) X^T y)
    S0_inv_m0 = _mat_vec_mul(S0_inv, m0)
    b  = [S0_inv_m0[j] + Xty[j] / sigma2 for j in range(k)]
    mn = _mat_vec_mul(Sn, b)

    y_hat     = _mat_vec_mul(X, mn)
    residuals = _vec_sub(y, y_hat)

    return {
        "mn": mn,
        "Sn": Sn,
        "m0": m0,
        "S0": S0,
        "sigma2": sigma2,
        "y_hat": y_hat,
        "residuals": residuals,
    }


def bayesian_lr_predict(
    X_test: list[list[float]],
    fit_result: dict,
) -> dict:
    """
    Dự đoán với Bayesian LR, kèm uncertainty quantification.

    Phân phối predictive marginal:
        p(y* | x*, X, y) = N(y*;  m_n^T x*,  sigma^2_pred(x*))
        sigma^2_pred(x*) = sigma^2 + x*^T S_n x*       <- epistemic + aleatoric uncertainty

    Args:
        X_test     : Ma trận design test (m × (p+1)), đã có cột intercept.
        fit_result : Kết quả từ bayesian_lr_fit().

    Returns:
        dict:
            y_hat    : list (m,) — điểm dự đoán = posterior mean m_n^T x*.
            pred_var : list (m,) — phương sai predictive từng quan sát.
            pred_std : list (m,) — độ lệch chuẩn predictive (= bán kính CI).
    """
    mn     = fit_result["mn"]
    Sn     = fit_result["Sn"]
    sigma2 = fit_result["sigma2"]

    y_hat    = _mat_vec_mul(X_test, mn)
    pred_var = []
    for xi in X_test:
        Sn_xi     = _mat_vec_mul(Sn, xi)
        epistemic = sum(xi[j] * Sn_xi[j] for j in range(len(xi)))
        pred_var.append(sigma2 + epistemic)

    pred_std = [math.sqrt(v) for v in pred_var]
    return {"y_hat": y_hat, "pred_var": pred_var, "pred_std": pred_std}


# ===========================================================================
# TÍCH HỢP — compare_advanced_models()
# ===========================================================================

def compare_advanced_models(
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    X_test: pd.DataFrame | np.ndarray,
    y_test: pd.Series | np.ndarray,
    kr_lambdas: Sequence[float] | None = None,
    kr_ells: Sequence[float] | None = None,
    kr_kernel: str = "rbf",
    k_folds: int = 5,
    seed: int = 42,
    tau2: float = 100.0,
) -> pd.DataFrame:
    """
    Huấn luyện và đánh giá Kernel Ridge + Bayesian LR trên test set.

    Trả về pd.DataFrame cùng cột với compare_models() trong model_comparison.py,
    giúp dễ dàng pd.concat để so sánh tổng hợp tất cả mô hình.

    Args:
        X_train, y_train : Dữ liệu huấn luyện (sau DataPipeline, KHÔNG có intercept).
        X_test,  y_test  : Dữ liệu kiểm tra (sau DataPipeline).
        kr_lambdas       : Lưới λ cho Kernel Ridge CV (mặc định 10^{-2}…10^{2}).
        kr_ells          : Lưới ℓ cho RBF CV (mặc định [0.1, 0.5, 1, 2, 5]).
        kr_kernel        : 'rbf' hoặc 'poly'.
        k_folds          : Số fold cross-validation.
        seed             : Random seed để tái lập kết quả.
        tau2             : Phương sai prior S_0 = tau^2 * I cho Bayesian LR.

    Returns:
        pd.DataFrame: 2 hàng — Kernel Ridge và Bayesian LR, với các cột:
            Model, MAE, RMSE, R2, n_features + cột meta (lambda, ell, ...).
    """
    X_tr_raw = _to_list_2d(X_train)
    X_te_raw = _to_list_2d(X_test)
    y_tr     = _to_list_1d(y_train)
    y_te     = _to_list_1d(y_test)
    n_feat   = len(X_tr_raw[0])

    results: list[dict] = []

    # ----------------------------------------------------------------
    # 1. Kernel Ridge Regression — tune (λ, ℓ) qua CV
    # ----------------------------------------------------------------
    best_lam, best_ell, kr_cv_mse = tune_kernel_cv(
        X_tr_raw, y_tr,
        kernel=kr_kernel,
        lambdas=kr_lambdas,
        ells=kr_ells,
        k=k_folds,
        seed=seed,
    )
    kr_fit     = kernel_ridge_fit(X_tr_raw, y_tr, lam=best_lam, kernel=kr_kernel, ell=best_ell)
    y_pred_kr  = kernel_ridge_predict(kr_fit, X_te_raw)
    m_kr       = _compute_metrics(y_te, y_pred_kr)

    results.append({
        "Model"     : f"Kernel Ridge ({kr_kernel.upper()}, CV λ/ℓ)",
        "MAE"       : m_kr["MAE"],
        "RMSE"      : m_kr["RMSE"],
        "R2"        : m_kr["R2"],
        "n_features": n_feat,
        "lambda"    : best_lam,
        "ell"       : best_ell,
        "cv_mse"    : kr_cv_mse,
    })

    # ----------------------------------------------------------------
    # 2. Bayesian Linear Regression — prior non-informative N(0, tau^2 * I)
    # ----------------------------------------------------------------
    # Thêm intercept cho Bayesian (cần design matrix đầy đủ)
    X_tr_design = [[1.0] + row for row in X_tr_raw]
    X_te_design = [[1.0] + row for row in X_te_raw]

    sigma2_hat   = _estimate_sigma2_ols(X_tr_design, y_tr)
    bayes_fit    = bayesian_lr_fit(X_tr_design, y_tr, sigma2=sigma2_hat, tau2=tau2)
    bayes_pred   = bayesian_lr_predict(X_te_design, bayes_fit)
    y_pred_bayes = bayes_pred["y_hat"]
    m_bayes      = _compute_metrics(y_te, y_pred_bayes)

    results.append({
        "Model"     : "Bayesian LR (conjugate)",
        "MAE"       : m_bayes["MAE"],
        "RMSE"      : m_bayes["RMSE"],
        "R2"        : m_bayes["R2"],
        "n_features": n_feat + 1,    # +1 do có intercept
        "sigma2"    : sigma2_hat,
        "tau2"      : tau2,
    })

    df = pd.DataFrame(results)
    display_cols = ["Model", "MAE", "RMSE", "R2", "n_features"]
    extra_cols   = [c for c in df.columns if c not in display_cols]
    return df[display_cols + extra_cols]


# ===========================================================================
# VISUALIZATIONS
# ===========================================================================

def plot_kernel_actual_vs_predicted(
    fit_result: dict,
    X_test: list[list[float]],
    y_test: list[float],
    title: str = "Kernel Ridge — Thực tế vs Dự đoán",
) -> None:
    """Scatter plot: y_test vs y_hat_test. Đường đỏ = perfect prediction."""
    import matplotlib.pyplot as plt

    y_pred = kernel_ridge_predict(fit_result, X_test)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_test, y_pred, alpha=0.6, edgecolors="k", linewidths=0.4)
    mn, mx = min(y_test + y_pred), max(y_test + y_pred)
    ax.plot([mn, mx], [mn, mx], "r--", label="y = y_hat")
    ax.set_xlabel("Giá trị thực y")
    ax.set_ylabel("Dự đoán y_hat")
    ax.set_title(title)
    ax.legend()
    ax.grid(linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()


def plot_bayesian_credible_intervals(
    X_test: list[list[float]],
    y_test: list[float],
    fit_result: dict,
    n_show: int = 60,
    title: str = "Bayesian LR — 95% Credible Intervals",
) -> None:
    """
    Vẽ credible interval 95% cho n_show điểm đầu tiên.
    X_test phải là design matrix (đã có intercept, dùng kết quả bayesian_lr_fit).
    """
    import matplotlib.pyplot as plt

    pred     = bayesian_lr_predict(X_test, fit_result)
    y_hat    = pred["y_hat"]
    pred_std = pred["pred_std"]

    n   = min(n_show, len(y_test))
    idx = list(range(n))
    lower = [y_hat[i] - 1.96 * pred_std[i] for i in idx]
    upper = [y_hat[i] + 1.96 * pred_std[i] for i in idx]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(idx, [y_test[i] for i in idx], "ko", markersize=4, label="Thực tế y", zorder=3)
    ax.plot(idx, [y_hat[i]  for i in idx], "b-", linewidth=1.5, label="Posterior mean m_n^T x*")
    ax.fill_between(idx, lower, upper, alpha=0.25, color="steelblue", label="95% Credible Interval")
    ax.set_xlabel("Chỉ số quan sát")
    ax.set_ylabel("Giá trị")
    ax.set_title(title)
    ax.legend()
    ax.grid(linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()


def plot_kernel_cv_surface(
    X: list[list[float]],
    y: list[float],
    lambdas: Sequence[float] | None = None,
    ells: Sequence[float] | None = None,
    k: int = 5,
    seed: int = 42,
    title: str = "Kernel Ridge CV — MSE theo (λ, ℓ)",
) -> None:
    """
    Heatmap CV MSE trên lưới (λ, ℓ) để trực quan hóa quá trình chọn tham số.
    """
    import matplotlib.pyplot as plt

    if lambdas is None:
        lambdas = [10.0 ** e for e in range(-2, 3)]
    if ells is None:
        ells = [0.1, 0.5, 1.0, 2.0, 5.0]

    n_lam, n_ell = len(lambdas), len(ells)
    Z = [[0.0] * n_ell for _ in range(n_lam)]

    n = len(X)
    rng  = np.random.default_rng(seed)
    order = rng.permutation(n).tolist()
    base  = n // k
    folds: list[list[int]] = []
    start = 0
    for i in range(k):
        end = start + base + (1 if i < n % k else 0)
        folds.append(order[start:end])
        start = end

    for li, lam in enumerate(lambdas):
        for ei, ell in enumerate(ells):
            mses = []
            for fi in range(k):
                val_set = set(folds[fi])
                tr_idx  = [i for i in range(n) if i not in val_set]
                val_idx = list(val_set)
                fit   = kernel_ridge_fit([X[i] for i in tr_idx], [y[i] for i in tr_idx],
                                         lam=float(lam), kernel="rbf", ell=float(ell))
                y_p   = kernel_ridge_predict(fit, [X[i] for i in val_idx])
                y_v   = [y[i] for i in val_idx]
                mses.append(sum((y_v[i] - y_p[i]) ** 2 for i in range(len(y_v))) / len(y_v))
            Z[li][ei] = sum(mses) / k

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(Z, aspect="auto", origin="lower")
    ax.set_xticks(range(n_ell))
    ax.set_xticklabels([str(e) for e in ells])
    ax.set_yticks(range(n_lam))
    ax.set_yticklabels([str(l) for l in lambdas])
    ax.set_xlabel("ℓ (length-scale)")
    ax.set_ylabel("λ (regularization)")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="CV MSE")
    plt.tight_layout()
    plt.show()


# ===========================================================================
# UNIT TESTS (≥ 2 test / hàm chính)
# ===========================================================================

def test_rbf_kernel_self():
    """rbf_kernel(x, x) = 1 với mọi x và mọi ell."""
    x = [1.0, 2.0, 3.0]
    assert rbf_kernel(x, x, ell=1.0) == 1.0, "k(x,x) phải = 1"
    assert rbf_kernel(x, x, ell=10.0) == 1.0, "k(x,x) phải = 1 với ell bất kỳ"
    print("test_rbf_kernel_self PASSED")


def test_rbf_kernel_decay():
    """rbf_kernel giảm khi hai điểm càng xa nhau."""
    x    = [0.0]
    near = [1.0]
    far  = [10.0]
    assert rbf_kernel(x, near, ell=1.0) > rbf_kernel(x, far, ell=1.0), \
        "Kernel phải lớn hơn khi điểm gần hơn"
    print("test_rbf_kernel_decay PASSED")


def test_gram_matrix_symmetric():
    """Ma trận Gram K(X, X) phải đối xứng: K[i][j] = K[j][i]."""
    X = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    K = gram_matrix(X, kernel="rbf", ell=1.0)
    n = len(K)
    for i in range(n):
        for j in range(n):
            assert abs(K[i][j] - K[j][i]) < 1e-12, f"K[{i}][{j}] ≠ K[{j}][{i}]"
    print("test_gram_matrix_symmetric PASSED")


def test_gram_matrix_diagonal_one():
    """Đường chéo K(x_i, x_i) = 1 với mọi i khi dùng RBF kernel."""
    X = [[1.0, 0.0], [0.0, 2.0], [3.0, 3.0]]
    K = gram_matrix(X, kernel="rbf", ell=2.0)
    for i in range(len(X)):
        assert abs(K[i][i] - 1.0) < 1e-12, f"K[{i}][{i}] phải = 1"
    print("test_gram_matrix_diagonal_one PASSED")


def test_kernel_ridge_fit_low_lambda():
    """Với λ rất nhỏ, Kernel Ridge phải fit gần hoàn hảo trên tập train."""
    rng = np.random.default_rng(0)
    n   = 30
    X   = rng.normal(size=(n, 2)).tolist()
    y   = [2.0 * X[i][0] - 1.5 * X[i][1] + 0.1 for i in range(n)]

    fit   = kernel_ridge_fit(X, y, lam=1e-6, kernel="rbf", ell=1.0)
    y_hat = fit["y_hat"]
    mse   = sum((y[i] - y_hat[i]) ** 2 for i in range(n)) / n
    assert mse < 0.05, f"MSE train = {mse:.4f} — quá lớn với λ rất nhỏ"
    print(f"test_kernel_ridge_fit_low_lambda PASSED (MSE train = {mse:.6f})")


def test_kernel_ridge_predict_shape():
    """kernel_ridge_predict phải trả về đúng m phần tử ứng với m mẫu test."""
    X_train = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    y_train = [1.0, 2.0, 3.0]
    X_test  = [[2.0, 3.0], [4.0, 5.0]]

    fit    = kernel_ridge_fit(X_train, y_train, lam=1.0)
    y_pred = kernel_ridge_predict(fit, X_test)
    assert len(y_pred) == 2, f"Kỳ vọng 2 phần tử, nhận {len(y_pred)}"
    print("test_kernel_ridge_predict_shape PASSED")


def test_bayesian_lr_prior_noninformative():
    """
    Với prior non-informative (tau^2 = 10^8), posterior mean m_n phải xấp xỉ OLS.
    """
    rng  = np.random.default_rng(1)
    n, p = 80, 3
    X_raw = rng.normal(size=(n, p))
    X    = [[1.0] + list(X_raw[i]) for i in range(n)]
    beta = [1.0, 2.0, -1.5, 0.5]
    y    = [sum(X[i][j] * beta[j] for j in range(p + 1)) + rng.normal() * 0.3
            for i in range(n)]

    # OLS
    Xt      = _transpose(X)
    beta_ols = _mat_vec_mul(_mat_inv(_mat_mul(Xt, X)), _mat_vec_mul(Xt, y))

    # Bayesian — prior rất yếu
    sigma2 = _estimate_sigma2_ols(X, y)
    fit    = bayesian_lr_fit(X, y, sigma2=sigma2, tau2=1e8)
    mn     = fit["mn"]

    for j in range(p + 1):
        diff = abs(mn[j] - beta_ols[j])
        assert diff < 0.01, f"beta_Bayes[{j}] = {mn[j]:.4f} vs beta_OLS[{j}] = {beta_ols[j]:.4f}"
    print("test_bayesian_lr_prior_noninformative PASSED")


def test_bayesian_lr_credible_interval_coverage():
    """
    Credible interval 95% phải bao phủ ≥ 90% điểm test trên dữ liệu sạch.
    """
    rng = np.random.default_rng(2)
    n_train, n_test = 100, 50
    X_tr_raw = rng.normal(size=(n_train, 2))
    X_te_raw = rng.normal(size=(n_test, 2))
    X_train  = [[1.0] + list(r) for r in X_tr_raw]
    X_test   = [[1.0] + list(r) for r in X_te_raw]
    beta     = [0.5, 1.5, -1.0]
    sigma_noise = 0.5

    y_train = [sum(X_train[i][j] * beta[j] for j in range(3)) + rng.normal() * sigma_noise
               for i in range(n_train)]
    y_test  = [sum(X_test[i][j]  * beta[j] for j in range(3)) + rng.normal() * sigma_noise
               for i in range(n_test)]

    sigma2 = _estimate_sigma2_ols(X_train, y_train)
    fit    = bayesian_lr_fit(X_train, y_train, sigma2=sigma2)
    pred   = bayesian_lr_predict(X_test, fit)

    covered = sum(
        1 for i in range(n_test)
        if (pred["y_hat"][i] - 1.96 * pred["pred_std"][i]) <= y_test[i]
        <= (pred["y_hat"][i] + 1.96 * pred["pred_std"][i])
    )
    coverage = covered / n_test
    assert coverage >= 0.90, f"Coverage = {coverage:.2%}, kỳ vọng ≥ 90%"
    print(f"test_bayesian_lr_credible_interval_coverage PASSED (coverage = {coverage:.2%})")


def test_compare_advanced_models_smoke():
    """
    Smoke test: compare_advanced_models chạy được và trả về đúng cấu trúc.
    """
    rng = np.random.default_rng(99)
    n   = 100
    X   = rng.normal(size=(n, 4))
    y   = 1.0 + 2.0 * X[:, 0] - 1.5 * X[:, 1] + rng.normal(scale=0.5, size=n)

    split    = 70
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    df = compare_advanced_models(
        X_train, y_train, X_test, y_test,
        kr_lambdas=[0.1, 1.0, 10.0],
        kr_ells=[0.5, 1.0, 2.0],
        k_folds=3,
        seed=99,
    )
    assert len(df) == 2, f"Kỳ vọng 2 dòng, nhận {len(df)}"
    assert {"MAE", "RMSE", "R2"}.issubset(df.columns), "Thiếu cột metrics"
    assert df["R2"].notna().any(), "R2 không được là NaN"
    print("test_compare_advanced_models_smoke PASSED")
    print("\n--- Bảng so sánh mô hình nâng cao (smoke test) ---")
    print(df[["Model", "MAE", "RMSE", "R2"]].to_string(index=False))


# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

    print("=" * 55)
    print("  RUN UNIT TESTS: advanced_methods.py")
    print("=" * 55)

    # Kernel functions
    test_rbf_kernel_self()
    test_rbf_kernel_decay()

    # Gram matrix
    test_gram_matrix_symmetric()
    test_gram_matrix_diagonal_one()

    # Kernel Ridge
    test_kernel_ridge_fit_low_lambda()
    test_kernel_ridge_predict_shape()

    # Bayesian LR
    test_bayesian_lr_prior_noninformative()
    test_bayesian_lr_credible_interval_coverage()

    # Integration
    test_compare_advanced_models_smoke()

    print("=" * 55)
    print("  ALL TESTS PASSED ✓")
    print("=" * 55)
