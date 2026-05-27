"""
Model Comparison — Part 2
=========================
So sánh ≥ 3 mô hình hồi quy trên tập test:
  1. OLS cơ bản (tất cả biến)
  2. OLS chọn biến (backward elimination theo p-value hoặc VIF)
  3. Ridge / Lasso (chọn λ qua k-fold cross-validation)

Tích hợp với DataPipeline: dữ liệu đầu vào là X, y đã được fit/transform.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Callable, Iterable, Literal, Sequence

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from part1.cross_validation import kfold_cv
from part1.ols_implementation import coef_inference, column_stack, create_ones, ols_fit, vif
from part1.ridge_lasso import lasso_fit, ridge_fit


SelectionMethod = Literal["pvalue", "vif"]

DEFAULT_TARGET = "price"
AUTOMOBILE_DATA_PATH = Path(__file__).resolve().parent / "data" / "automobile.csv"
MAX_SELECTION_FEATURES = 40
RANK_TOL = 1e-10


def _to_list_2d(X: pd.DataFrame | np.ndarray) -> list[list[float]]:
    """Chuyển DataFrame/ndarray thành list of lists."""
    if isinstance(X, pd.DataFrame):
        return X.values.tolist()
    return np.asarray(X, dtype=float).tolist()


def _to_list_1d(y: pd.Series | np.ndarray | Sequence[float]) -> list[float]:
    """Chuyển vector mục tiêu thành list 1D."""
    if isinstance(y, pd.Series):
        return y.values.ravel().tolist()
    return np.asarray(y, dtype=float).ravel().tolist()


def drop_duplicate_columns(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loại cột trùng lặp (one-hot có thể sinh cột giống hệt) theo tập train.

    Giữ nguyên thứ tự cột train; test được căn theo cùng tập cột.
    """
    keep_mask = ~X_train.T.duplicated()
    kept_cols = X_train.columns[keep_mask]
    return X_train.loc[:, kept_cols], X_test.reindex(columns=kept_cols, fill_value=0)


def _predictor_matrix(X: list[list[float]]) -> np.ndarray:
    """Ma trận predictor (không intercept) từ design matrix."""
    return np.asarray([row[1:] for row in X], dtype=float)


def _feature_correlations_with_y(X: list[list[float]], y: list[float]) -> list[float]:
    """|corr(x_j, y)| cho từng predictor; NaN → 0."""
    X_np = _predictor_matrix(X)
    y_np = np.asarray(y, dtype=float)
    corrs: list[float] = []
    for j in range(X_np.shape[1]):
        x_j = X_np[:, j]
        if np.std(x_j) < 1e-12 or np.std(y_np) < 1e-12:
            corrs.append(0.0)
            continue
        corr = np.corrcoef(x_j, y_np)[0, 1]
        corrs.append(abs(float(corr)) if np.isfinite(corr) else 0.0)
    return corrs


def _try_ols_fit(X: list[list[float]], y: list[float]) -> bool:
    """Trả về True nếu ols_fit chạy được (n > p + 1 và X không suy biến)."""
    n = len(y)
    p = len(X[0]) - 1
    if p >= n - 1:
        return False
    try:
        ols_fit(X, y)
        return True
    except (ValueError, ZeroDivisionError):
        return False


def _cap_by_correlation(
    X: list[list[float]], y: list[float], active: list[int], max_features: int
) -> list[int]:
    """Giữ tối đa max_features biến có |corr(y)| cao nhất."""
    if len(active) <= max_features:
        return active
    corrs = _feature_correlations_with_y(X, y)
    ranked = sorted(active, key=lambda idx: -corrs[idx])
    return ranked[:max_features]


def _design_rank(X_sub: np.ndarray) -> int:
    """Rank ma trận design (numpy, nhanh hơn ols_fit lặp)."""
    return int(np.linalg.matrix_rank(X_sub, tol=RANK_TOL))


def reduce_to_identifiable_features(
    X: list[list[float]], y: list[float], active: list[int] | None = None
) -> list[int]:
    """
    Chọn tập biến lớn nhất (full rank) với p ≤ n−2 để OLS khả thi.

    Dùng greedy forward + kiểm tra rank bằng numpy (phù hợp Automobile: n≈160, p>200).
    """
    n = len(y)
    max_p = max(1, n - 2)
    X_np = _predictor_matrix(X)
    n_predictors = X_np.shape[1]
    active = list(range(n_predictors)) if active is None else list(active)

    corrs = _feature_correlations_with_y(X, y)
    candidates = sorted(active, key=lambda idx: -corrs[idx])

    selected: list[int] = []
    for idx in candidates:
        if len(selected) >= max_p:
            break
        trial = selected + [idx]
        X_sub = np.column_stack([np.ones(n), X_np[:, trial]])
        if _design_rank(X_sub) == len(trial) + 1:
            selected = trial

    return selected


def to_design_matrix(X: pd.DataFrame | np.ndarray) -> list[list[float]]:
    """
    Thêm cột intercept vào ma trận đặc trưng.

    Args:
        X: Ma trận đặc trưng (n × p), chưa có intercept.

    Returns:
        Ma trận design (n × (p+1)) dạng list of lists.
    """
    rows = _to_list_2d(X)
    return column_stack(create_ones(len(rows)), rows)


def subset_design_matrix(
    X: list[list[float]], feature_indices: Iterable[int]
) -> list[list[float]]:
    """
    Giữ intercept (cột 0) và các cột predictor theo chỉ số 0-based.

    Args:
        X: Ma trận design đã có intercept.
        feature_indices: Chỉ số predictor (0 = cột 1 trong X).
    """
    cols = [0] + [idx + 1 for idx in feature_indices]
    return [[row[c] for c in cols] for row in X]


def predict(X: list[list[float]], beta_hat: Sequence[float]) -> list[float]:
    """Dự đoán ŷ = Xβ."""
    return [sum(x[j] * beta_hat[j] for j in range(len(beta_hat))) for x in X]


def compute_test_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> dict[str, float]:
    """
    Tính MAE, RMSE, R² trên tập test.

    Công thức (PDF §2.3.3):
        MAE  = (1/n) Σ|yᵢ − ŷᵢ|
        RMSE = √[(1/n) Σ(yᵢ − ŷᵢ)²]
        R²   = 1 − RSS_test / TSS_test
    """
    y = list(y_true)
    y_hat = list(y_pred)
    n = len(y)
    if n == 0:
        raise ValueError("y_true không được rỗng.")

    errors = [y[i] - y_hat[i] for i in range(n)]
    mae = sum(abs(e) for e in errors) / n
    rmse = math.sqrt(sum(e * e for e in errors) / n)

    mean_y = sum(y) / n
    tss = sum((yi - mean_y) ** 2 for yi in y)
    rss = sum(e * e for e in errors)
    r2 = 1.0 - rss / tss if tss > 1e-15 else float("nan")

    return {"MAE": mae, "RMSE": rmse, "R2": r2, "n": float(n)}


def select_features_by_pvalue(
    X: list[list[float]], y: list[float], alpha: float = 0.05
) -> list[int]:
    """
    Backward elimination: loại dần biến có p-value lớn nhất cho đến khi
    tất cả biến còn lại có p-value ≤ alpha (kiểm định t trên tập train).

    Với p lớn (Automobile): tự giảm rank trước khi loại theo p-value.
    """
    n_predictors = len(X[0]) - 1
    active = reduce_to_identifiable_features(X, y, list(range(n_predictors)))
    active = _cap_by_correlation(
        X, y, active, min(MAX_SELECTION_FEATURES, max(1, len(y) - 2))
    )

    while len(active) > 1:
        X_sub = subset_design_matrix(X, active)
        if not _try_ols_fit(X_sub, y):
            active = reduce_to_identifiable_features(X, y, active)
            if not active:
                break
            continue

        fit = ols_fit(X_sub, y)
        inference = coef_inference(X_sub, y, fit["beta_hat"], fit["sigma2"], alpha=alpha)
        p_values = inference["p_values"][1:]

        worst_local = max(range(len(p_values)), key=lambda i: p_values[i])
        if p_values[worst_local] <= alpha:
            break
        active.pop(worst_local)

    return active


def select_features_by_vif(
    X: list[list[float]], threshold: float = 10.0, y: list[float] | None = None
) -> list[int]:
    """
    Loại dần biến có VIF cao nhất cho đến khi mọi VIF ≤ threshold.

    Tham số y (tùy chọn) dùng để giảm rank khi p ≥ n trước khi tính VIF.
    """
    n_predictors = len(X[0]) - 1
    if y is None:
        active = list(range(n_predictors))
    else:
        active = reduce_to_identifiable_features(X, y, list(range(n_predictors)))
        active = _cap_by_correlation(
            X, y, active, min(MAX_SELECTION_FEATURES, max(1, len(y) - 2))
        )

    while active:
        X_sub = subset_design_matrix(X, active)
        if not _try_ols_fit(X_sub, y if y is not None else [0.0] * len(X_sub)):
            if y is None:
                raise ValueError("Ma trận suy biến; truyền y để giảm rank tự động.")
            active = reduce_to_identifiable_features(X, y, active)
            if not active:
                break
            continue

        vif_values = vif(X_sub)
        max_vif = max(vif_values)

        if max_vif <= threshold:
            break

        worst = vif_values.index(max_vif)
        active.pop(worst)

    return active


def _default_lambda_grid(n_predictors: int) -> list[float]:
    """Lưới λ: thu gọn khi số biến lớn (Automobile sau one-hot)."""
    if n_predictors > MAX_SELECTION_FEATURES:
        return [0.1, 1.0, 10.0, 100.0]
    return [10.0 ** exp for exp in range(-3, 4)]


def _make_regularized_model_fn(
    method: str, lam: float, max_lasso_iter: int = 1000
) -> Callable:
    """Tạo hàm model_fn cho kfold_cv với Ridge hoặc Lasso."""

    def model_fn(X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        X_list = X_train.tolist()
        y_list = y_train.tolist()
        if method == "ridge":
            result = ridge_fit(X_list, y_list, lam)
        elif method == "lasso":
            result = lasso_fit(X_list, y_list, lam, max_iter=max_lasso_iter)
        else:
            raise ValueError("method phải là 'ridge' hoặc 'lasso'.")
        return np.asarray(result["beta_hat"], dtype=float)

    return model_fn


def tune_lambda_cv(
    X: list[list[float]],
    y: list[float],
    method: Literal["ridge", "lasso"] = "ridge",
    lambdas: Sequence[float] | None = None,
    k: int = 5,
    seed: int = 42,
    max_lasso_iter: int = 1000,
) -> tuple[float, float]:
    """
    Chọn siêu tham số λ bằng k-fold cross-validation (MSE trung bình).

    Returns:
        (best_lambda, best_cv_mse)
    """
    n_predictors = len(X[0]) - 1
    if lambdas is None:
        lambdas = _default_lambda_grid(n_predictors)

    X_np = np.asarray(X, dtype=float)
    y_np = np.asarray(y, dtype=float)

    best_lambda = float(lambdas[0])
    best_score = float("inf")

    for lam in lambdas:
        model_fn = _make_regularized_model_fn(method, float(lam), max_lasso_iter)
        cv_score, _ = kfold_cv(X_np, y_np, k=k, model_fn=model_fn, seed=seed)
        if cv_score < best_score:
            best_score = cv_score
            best_lambda = float(lam)

    return best_lambda, best_score


def _evaluate_model(
    name: str,
    X_train: list[list[float]],
    y_train: list[float],
    X_test: list[list[float]],
    y_test: list[float],
    fit_fn: Callable[[list[list[float]], list[float]], list[float]],
    extra: dict | None = None,
) -> dict:
    """Huấn luyện mô hình, dự đoán test và gom metrics."""
    beta_hat = fit_fn(X_train, y_train)
    y_pred = predict(X_test, beta_hat)
    metrics = compute_test_metrics(y_test, y_pred)

    row = {
        "Model": name,
        "MAE": metrics["MAE"],
        "RMSE": metrics["RMSE"],
        "R2": metrics["R2"],
        "n_features": len(beta_hat) - 1,
    }
    if extra:
        row.update(extra)
    return row


def compare_models(
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    X_test: pd.DataFrame | np.ndarray,
    y_test: pd.Series | np.ndarray,
    selection_method: SelectionMethod = "pvalue",
    alpha: float = 0.05,
    vif_threshold: float = 10.0,
    k_folds: int = 5,
    lambda_grid: Sequence[float] | None = None,
    seed: int = 42,
    drop_duplicates: bool = True,
) -> pd.DataFrame:
    """
    So sánh ≥ 3 mô hình và trả về bảng MAE, RMSE, R² trên test set.

    Mô hình:
        - OLS (Full) — tối đa hóa số biến khả thi (n > p, full rank)
        - OLS (Selected) — chọn biến theo p-value hoặc VIF
        - Ridge (CV λ)
        - Lasso (CV λ)

    Args:
        X_train, y_train: Dữ liệu huấn luyện (sau DataPipeline).
        X_test, y_test: Dữ liệu kiểm tra (sau DataPipeline).
        selection_method: 'pvalue' hoặc 'vif' cho OLS chọn biến.
        alpha: Ngưỡng p-value cho backward elimination.
        vif_threshold: Ngưỡng VIF.
        k_folds: Số fold cho chọn λ Ridge/Lasso.
        lambda_grid: Lưới λ thử nghiệm (mặc định 10^-3 … 10^3).
        seed: Random seed cho k-fold CV (tái lập kết quả).
        drop_duplicates: Loại cột trùng sau one-hot (cần cho Automobile).

    Returns:
        pd.DataFrame: Bảng so sánh metrics trên test set.
    """
    if drop_duplicates and isinstance(X_train, pd.DataFrame) and isinstance(X_test, pd.DataFrame):
        X_train, X_test = drop_duplicate_columns(X_train, X_test)

    X_tr = to_design_matrix(X_train)
    X_te = to_design_matrix(X_test)
    y_tr = _to_list_1d(y_train)
    y_te = _to_list_1d(y_test)

    results: list[dict] = []
    n_predictors = len(X_tr[0]) - 1
    if lambda_grid is None:
        lambda_grid = _default_lambda_grid(n_predictors)

    max_p = max(1, len(y_tr) - 2)
    reg_indices = _cap_by_correlation(
        X_tr, y_tr, list(range(n_predictors)), min(MAX_SELECTION_FEATURES, max_p)
    )
    X_tr_reg = subset_design_matrix(X_tr, reg_indices)
    X_te_reg = subset_design_matrix(X_te, reg_indices)
    lasso_max_iter = 200 if n_predictors > MAX_SELECTION_FEATURES else 1000

    # 1. OLS — dùng tập biến lớn nhất mà ma trận không suy biến (p < n)
    full_indices = reduce_to_identifiable_features(
        X_tr, y_tr, list(range(n_predictors))
    )
    X_tr_full = subset_design_matrix(X_tr, full_indices)
    X_te_full = subset_design_matrix(X_te, full_indices)

    def fit_ols_full(_X: list[list[float]], y: list[float]) -> list[float]:
        return ols_fit(X_tr_full, y)["beta_hat"]

    results.append(
        _evaluate_model(
            "OLS (Full)",
            X_tr_full,
            y_tr,
            X_te_full,
            y_te,
            fit_ols_full,
            extra={"feature_indices": full_indices},
        )
    )

    # 2. OLS chọn biến
    if selection_method == "pvalue":
        selected = select_features_by_pvalue(X_tr, y_tr, alpha=alpha)
        sel_label = f"OLS (Selected, p≤{alpha})"
    else:
        selected = select_features_by_vif(X_tr, threshold=vif_threshold, y=y_tr)
        sel_label = f"OLS (Selected, VIF≤{vif_threshold})"

    X_tr_sel = subset_design_matrix(X_tr, selected)
    X_te_sel = subset_design_matrix(X_te, selected)

    def fit_ols_selected(_X: list[list[float]], y: list[float]) -> list[float]:
        return ols_fit(X_tr_sel, y)["beta_hat"]

    results.append(
        _evaluate_model(
            sel_label,
            X_tr_sel,
            y_tr,
            X_te_sel,
            y_te,
            fit_ols_selected,
            extra={"selected_indices": selected},
        )
    )

    # 3. Ridge — chọn λ bằng CV (dùng tập biến thu gọn khi p lớn)
    ridge_lambda, ridge_cv = tune_lambda_cv(
        X_tr_reg,
        y_tr,
        method="ridge",
        lambdas=lambda_grid,
        k=k_folds,
        seed=seed,
    )

    def fit_ridge(_X: list[list[float]], y: list[float]) -> list[float]:
        return ridge_fit(X_tr_reg, y, ridge_lambda)["beta_hat"]

    results.append(
        _evaluate_model(
            "Ridge (CV λ)",
            X_tr_reg,
            y_tr,
            X_te_reg,
            y_te,
            fit_ridge,
            extra={"lambda": ridge_lambda, "cv_mse": ridge_cv, "feature_indices": reg_indices},
        )
    )

    # 4. Lasso — chọn λ bằng CV
    lasso_lambda, lasso_cv = tune_lambda_cv(
        X_tr_reg,
        y_tr,
        method="lasso",
        lambdas=lambda_grid,
        k=k_folds,
        seed=seed,
        max_lasso_iter=lasso_max_iter,
    )

    def fit_lasso(_X: list[list[float]], y: list[float]) -> list[float]:
        return lasso_fit(X_tr_reg, y, lasso_lambda, max_iter=lasso_max_iter)["beta_hat"]

    results.append(
        _evaluate_model(
            "Lasso (CV λ)",
            X_tr_reg,
            y_tr,
            X_te_reg,
            y_te,
            fit_lasso,
            extra={"lambda": lasso_lambda, "cv_mse": lasso_cv, "feature_indices": reg_indices},
        )
    )

    df = pd.DataFrame(results)
    display_cols = ["Model", "MAE", "RMSE", "R2", "n_features"]
    extra_cols = [c for c in df.columns if c not in display_cols]
    return df[display_cols + extra_cols]


def load_automobile_splits(
    data_path: Path | str | None = None,
    target: str = DEFAULT_TARGET,
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Đọc automobile.csv, làm sạch missing và chia train/test.

    Returns:
        X_train, y_train, X_test, y_test (chưa qua DataPipeline).
    """
    path = Path(data_path) if data_path is not None else AUTOMOBILE_DATA_PATH
    df = pd.read_csv(path).replace("?", np.nan)
    if target not in df.columns:
        raise ValueError(f"Khong tim thay cot muc tieu '{target}' trong {path.name}.")

    df = df.dropna(subset=[target])
    X = df.drop(columns=[target])
    y = df[target].astype(float)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    split = max(1, int((1.0 - test_size) * len(df)))
    train_idx, test_idx = idx[:split], idx[split:]

    return X.iloc[train_idx], y.iloc[train_idx], X.iloc[test_idx], y.iloc[test_idx]


def compare_models_on_automobile(
    data_path: Path | str | None = None,
    test_size: float = 0.2,
    seed: int = 42,
    **compare_kwargs,
) -> pd.DataFrame:
    """
    Pipeline đầy đủ: automobile.csv → DataPipeline → compare_models.
    """
    from part2.data_pipeline import DataPipeline

    X_train, y_train, X_test, y_test = load_automobile_splits(
        data_path=data_path, test_size=test_size, seed=seed
    )
    pipeline = DataPipeline()
    X_tr = pipeline.fit_transform(X_train)
    X_te = pipeline.transform(X_test)
    return compare_models(
        X_tr, y_train, X_te, y_test, seed=seed, **compare_kwargs
    )


def plot_coefficient_importance(
    beta_hat: Sequence[float],
    feature_names: Sequence[str] | None = None,
    title: str = "Hệ số hồi quy (sau chuẩn hóa)",
    top_k: int | None = 20,
) -> None:
    """
    Vẽ biểu đồ hệ số hồi quy (bỏ intercept) để giải thích mô hình.
    """
    import matplotlib.pyplot as plt

    coefs = list(beta_hat[1:])
    if feature_names is None:
        names = [f"x{i}" for i in range(len(coefs))]
    else:
        names = list(feature_names[: len(coefs)])

    pairs = sorted(zip(names, coefs), key=lambda t: abs(t[1]), reverse=True)
    if top_k is not None:
        pairs = pairs[:top_k]

    labels, values = zip(*pairs) if pairs else ([], [])
    colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in values]

    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(labels))))
    ax.barh(range(len(labels)), values, color=colors)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Hệ số β (scale đã chuẩn hóa)")
    ax.set_title(title)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Unit tests (≥ 2 test / hàm chính)
# ---------------------------------------------------------------------------

def test_compute_test_metrics_perfect():
    y = [1.0, 2.0, 3.0, 4.0]
    metrics = compute_test_metrics(y, y)
    assert metrics["MAE"] == 0.0
    assert metrics["RMSE"] == 0.0
    assert metrics["R2"] == 1.0
    print("test_compute_test_metrics_perfect PASSED")


def test_compute_test_metrics_known():
    y_true = [3.0, -0.5, 2.0, 7.0]
    y_pred = [2.5, 0.0, 2.0, 8.0]
    metrics = compute_test_metrics(y_true, y_pred)
    expected_mae = sum(abs(a - b) for a, b in zip(y_true, y_pred)) / 4
    expected_rmse = math.sqrt(sum((a - b) ** 2 for a, b in zip(y_true, y_pred)) / 4)
    assert abs(metrics["MAE"] - expected_mae) < 1e-12
    assert abs(metrics["RMSE"] - expected_rmse) < 1e-12
    assert metrics["R2"] < 1.0
    print("test_compute_test_metrics_known PASSED")


def test_select_features_by_pvalue():
    """Biến x2 không liên quan → bị loại khỏi mô hình."""
    rng = np.random.default_rng(0)
    n = 120
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    noise = rng.normal(scale=0.3, size=n)
    y = (2.0 + 3.0 * x1 + noise).tolist()
    X = to_design_matrix(np.column_stack([x1, x2]))

    selected = select_features_by_pvalue(X, y, alpha=0.05)
    assert selected == [0], f"Kỳ vọng chỉ giữ x1, nhận được {selected}"
    print("test_select_features_by_pvalue PASSED")


def test_select_features_by_vif():
    """x2 = x1 gần tuyến tính → VIF cao, một trong hai bị loại."""
    n = 100
    x1 = np.linspace(0, 1, n)
    x2 = x1 + np.random.default_rng(1).normal(scale=0.1, size=n)
    y = (1.0 + 2.0 * x1).tolist()
    X = to_design_matrix(np.column_stack([x1, x2]))

    selected = select_features_by_vif(X, threshold=5.0)
    assert len(selected) == 1
    print("test_select_features_by_vif PASSED")


def test_reduce_to_identifiable_features():
    """p > n → giảm còn ≤ n−1 biến và ols_fit thành công."""
    rng = np.random.default_rng(11)
    n, p = 50, 80
    X_raw = rng.normal(size=(n, p))
    y = (X_raw[:, 0] + rng.normal(scale=0.1, size=n)).tolist()
    X = to_design_matrix(X_raw)

    active = reduce_to_identifiable_features(X, y)
    assert len(active) <= n - 2
    assert _try_ols_fit(subset_design_matrix(X, active), y)
    print("test_reduce_to_identifiable_features PASSED")


def test_drop_duplicate_columns():
    """Cột trùng sau one-hot bị loại trên train, test được căn cột."""
    X_train = pd.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3], "c": [4, 5, 6]})
    X_test = pd.DataFrame({"a": [7, 8], "b": [7, 8], "c": [9, 10]})
    tr, te = drop_duplicate_columns(X_train, X_test)
    assert list(tr.columns) == ["a", "c"]
    assert list(te.columns) == ["a", "c"]
    print("test_drop_duplicate_columns PASSED")


def test_compare_models_automobile():
    """Integration: compare_models chay duoc tren automobile.csv."""
    table = compare_models_on_automobile(test_size=0.2, seed=42, k_folds=3)
    assert len(table) >= 4
    assert {"MAE", "RMSE", "R2"}.issubset(table.columns)
    assert table["R2"].notna().all()
    assert (table["n_features"] > 0).all()
    print("test_compare_models_automobile PASSED")


def test_compare_models_smoke():
    """Smoke test: compare_models chạy được và trả về ≥ 3 dòng."""
    rng = np.random.default_rng(42)
    n = 150
    X_raw = rng.normal(size=(n, 4))
    y = 1.0 + 2.0 * X_raw[:, 0] - 1.5 * X_raw[:, 1] + rng.normal(scale=0.5, size=n)

    split = 100
    X_train, X_test = X_raw[:split], X_raw[split:]
    y_train, y_test = y[:split], y[split:]

    table = compare_models(
        X_train, y_train, X_test, y_test, k_folds=3, seed=42
    )
    assert len(table) >= 3
    assert {"MAE", "RMSE", "R2"}.issubset(table.columns)
    assert table["R2"].notna().any()
    print("test_compare_models_smoke PASSED")


def test_tune_lambda_cv():
    """Ridge với λ lớn trên dữ liệu nhiễu — hàm chọn λ phải trả về giá trị hợp lệ."""
    rng = np.random.default_rng(7)
    n, p = 80, 3
    X_raw = rng.normal(size=(n, p))
    y = 1.0 + X_raw @ np.array([1.0, -0.5, 0.3]) + rng.normal(scale=0.8, size=n)
    X = to_design_matrix(X_raw)

    lam, cv_mse = tune_lambda_cv(X, y.tolist(), method="ridge", k=3, seed=7)
    assert lam > 0
    assert cv_mse > 0
    print("test_tune_lambda_cv PASSED")


if __name__ == "__main__":
    print("--- RUN UNIT TESTS: model_comparison.py ---")
    test_compute_test_metrics_perfect()
    test_compute_test_metrics_known()
    test_drop_duplicate_columns()
    test_reduce_to_identifiable_features()
    test_select_features_by_pvalue()
    test_select_features_by_vif()
    test_tune_lambda_cv()
    test_compare_models_smoke()
    test_compare_models_automobile()
    print("--- ALL TESTS PASSED ---")
