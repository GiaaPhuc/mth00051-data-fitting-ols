import numpy as np



def kfold_cv(X, y, k = 5, model_fn=None, seed=42):
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    folds = np.array_split(indices, k)

    if model_fn is None:
        def model_fn(X_train, y_train):
            XtX = X_train.T @ X_train
            Xty = X_train.T @ y_train
            beta_hat = np.linalg.solve(XtX, Xty)
            return beta_hat
    
    mse_list = []

    for i in range(k):
        test_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        beta_hat = model_fn(X_train, y_train)
        y_pred = X_test @ beta_hat  
        mse_i = np.mean((y_test - y_pred) **2)
        mse_list.append(mse_i)

    cv_score = np.mean(mse_list)
    return cv_score, mse_list
    
def test_kfold_cv():
    """Test 1: Kiểm tra trên dữ liệu tuyến tính hoàn hảo (không nhiễu)."""
    np.random.seed(42)
    n,p = 100, 3

    X_raw = np.random.randn(n, p)
    X = np.column_stack([np.ones(n), X_raw])    
    beta_true = np.array([2.0, 1.5, -0.5, 3.0])
    y = X @ beta_true

    cv_score, mse_list = kfold_cv(X, y, k=5)
    assert cv_score < 1e-20, f"CV score quá lớn: {cv_score}"
    assert len(mse_list) == 5, "Phải có đúng 5 MSE (k=5)"
    print("Test 1 PASSED: Dữ liệu không nhiễu → CV score gần = 0")

def compare_models_cv(X, y, model_fns, model_names, k = 5):
    results = {}
    for name, fn in zip(model_names, model_fns):
        cv_score, _ = kfold_cv(X, y, k = k, model_fn = fn)
        results[name] = cv_score
        print(f"  {name:20s} | CV Score (MSE): {cv_score:.6f}")

    best = min(results, key = results.get)
    print(f"\n Mô hình tốt nhât: {best} (MSE = {results[best]:.6f})")

    return results
    
     
def test_kfold_cv_with_noise():
    """Test 2: Kiểm tra trên dữ liệu có nhiễu — CV score phải hợp lý."""
    np.random.seed(123)
    n, p = 200, 2
    
    X_raw = np.random.randn(n, p)
    X = np.column_stack([np.ones(n), X_raw])
    beta_true = np.array([1.0, 2.0, -1.0])
    noise = np.random.randn(n) * 0.5           
    y = X @ beta_true + noise
    
    cv_score, mse_list = kfold_cv(X, y, k=10)
    
    assert 0.1 < cv_score < 1.0, f"CV score không hợp lý: {cv_score}"
    assert len(mse_list) == 10, "Phải có đúng 10 MSE (k=10)"
    print(f"Test 2 PASSED: CV score = {cv_score:.4f} (kỳ vọng gần = 0.25)")

def test_compare_with_sklearn():
    """Test 3: Kiểm chứng kết quả K-Fold CV với thư viện sklearn."""
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import KFold, cross_val_score
    
    np.random.seed(999)
    n, p = 150, 4
    X_raw = np.random.randn(n, p)
    X = np.column_stack([np.ones(n), X_raw])
    beta_true = np.array([5.0, 1.2, -3.4, 2.1, 0.5])
    y = X @ beta_true + np.random.randn(n) * 1.5
    
    cv_score_custom, _ = kfold_cv(X, y, k=5, seed=42)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    model_sk = LinearRegression(fit_intercept=False) # Không fit intercept vì X đã có cột 1
    
    scores_sk = cross_val_score(model_sk, X, y, cv=kf, scoring='neg_mean_squared_error')
    cv_score_sklearn = -np.mean(scores_sk)
    
    print("\n--- SO SÁNH VỚI SKLEARN ---")
    print(f"Điểm CV tự viết  : {cv_score_custom:.4f}")
    print(f"Điểm CV sklearn  : {cv_score_sklearn:.4f}")
    
    diff = abs(cv_score_custom - cv_score_sklearn)
    assert diff < 0.5, f"Lệch nhau quá xa: {diff}"
    print(f"Test 3 PASSED: Kết quả tương đương với sklearn (chênh lệch chỉ {diff:.4f})")

if __name__ == "__main__":
    print("--- CHẠY UNIT TEST ---")
    test_kfold_cv()
    test_kfold_cv_with_noise()
    test_compare_with_sklearn()
    print("----------------------")