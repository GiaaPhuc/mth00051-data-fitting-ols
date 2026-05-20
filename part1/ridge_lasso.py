import math

# --- Helper Functions for Linear Algebra ---

def transpose(M):
    """Returns the transpose of a 2D matrix (list of lists)."""
    if not M: return []
    return [[M[i][j] for i in range(len(M))] for j in range(len(M[0]))]

def matmul(A, B):
    """Multiplies two 2D matrices A and B."""
    m, n = len(A), len(A[0])
    p = len(B[0])
    C = [[0.0] * p for _ in range(m)]
    for i in range(m):
        for j in range(p):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]
    return C

def mat_vec_mul(A, v):
    """Multiplies a 2D matrix A by a vector v."""
    m, n = len(A), len(A[0])
    res = [0.0] * m
    for i in range(m):
        for j in range(n):
            res[i] += A[i][j] * v[j]
    return res

def vec_dot(u, v):
    """Returns the dot product of two vectors."""
    return sum(ui * vi for ui, vi in zip(u, v))

def vec_sub(u, v):
    """Returns the element-wise subtraction of two vectors."""
    return [ui - vi for ui, vi in zip(u, v)]

def solve_linear_system(A, b):
    """Solves Ax = b using Gaussian elimination with partial pivoting."""
    n = len(A)
    # Augment A with b
    M = [row[:] + [bi] for row, bi in zip(A, b)]
    
    for i in range(n):
        # Pivoting
        max_row = i
        for k in range(i + 1, n):
            if abs(M[k][i]) > abs(M[max_row][i]):
                max_row = k
        M[i], M[max_row] = M[max_row], M[i]
        
        pivot = M[i][i]
        if abs(pivot) < 1e-15:
            # Add a small epsilon to the diagonal if singular (like ridge regularization does anyway)
            pivot += 1e-15
            
        for k in range(i + 1, n):
            factor = M[k][i] / pivot
            for j in range(i, n + 1):
                M[k][j] -= factor * M[i][j]
                
    # Back substitution
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = M[i][n]
        for j in range(i + 1, n):
            x[i] -= M[i][j] * x[j]
        if abs(M[i][i]) < 1e-15:
             x[i] = 0.0
        else:
            x[i] /= M[i][i]
    return x

def mean(v):
    """Returns the mean of a list."""
    return sum(v) / len(v) if v else 0.0

def std(v):
    """Returns the population standard deviation of a list."""
    if not v: return 0.0
    m = mean(v)
    variance = sum((xi - m)**2 for xi in v) / len(v)
    return math.sqrt(variance)

# --- Main Regression Functions ---

def ridge_fit(X: list, y: list, lam: float) -> dict:
    """Estimates Ridge Regression coefficients (L2 regularization) without NumPy.

    Formula: beta = (X^T X + lam * I)^-1 X^T y

    Args:
        X (list): Design matrix (list of lists) of shape (n, p+1).
        y (list): Target vector.
        lam (float): Regularization parameter.

    Returns:
        dict: Contains beta_hat, beta_std, y_hat, residuals.
    """
    n = len(X)
    k = len(X[0])
    
    # 1. Standardization (excluding intercept column at index 0)
    # Extract columns
    cols = [[X[i][j] for i in range(n)] for j in range(k)]
    means = [mean(c) for c in cols]
    stds = [std(c) for c in cols]
    
    X_std = []
    for i in range(n):
        row = [X[i][0]] # Keep intercept as is
        for j in range(1, k):
            s = stds[j] if stds[j] != 0 else 1.0
            row.append((X[i][j] - means[j]) / s)
        X_std.append(row)
        
    y_mean = mean(y)
    y_std = [yi - y_mean for yi in y]
    
    # 2. Closed-form solution: (X^T X + lam * I) * beta = X^T y
    Xt = transpose(X_std)
    XtX = matmul(Xt, X_std)
    Xty = mat_vec_mul(Xt, y_std)
    
    # Add lambda to diagonal
    for i in range(1, k):
        XtX[i][i] += lam
        
    beta_std = solve_linear_system(XtX, Xty)
    
    # 3. Unstandardize to original scale
    beta_hat = [0.0] * k
    for j in range(1, k):
        s = stds[j] if stds[j] != 0 else 1.0
        beta_hat[j] = beta_std[j] / s
        
    # Intercept calculation
    # beta_hat[0] = beta_std[0] + y_mean - sum(beta_hat[j] * mean[j])
    sum_weighted_means = sum(beta_hat[j] * means[j] for j in range(1, k))
    beta_hat[0] = beta_std[0] + y_mean - sum_weighted_means
    
    y_hat = [vec_dot(X[i], beta_hat) for i in range(n)]
    residuals = vec_sub(y, y_hat)
    
    return {
        "beta_hat": beta_hat,
        "beta_std": beta_std,
        "y_hat": y_hat,
        "residuals": residuals
    }


def lasso_fit(X: list, y: list, lam: float, tol: float = 1e-4, max_iter: int = 1000) -> dict:
    """Estimates Lasso Regression coefficients (L1 regularization) using Coordinate Descent.

    Args:
        X (list): Design matrix (list of lists).
        y (list): Target vector.
        lam (float): Regularization parameter.
        tol (float): Convergence tolerance.
        max_iter (int): Maximum iterations.

    Returns:
        dict: Contains beta_hat, beta_std, y_hat, residuals, iterations.
    """
    n = len(X)
    k = len(X[0])
    
    # 1. Standardization
    cols = [[X[i][j] for i in range(n)] for j in range(k)]
    means = [mean(c) for c in cols]
    stds = [std(c) for c in cols]
    
    X_std = []
    for i in range(n):
        row = [X[i][0]]
        for j in range(1, k):
            s = stds[j] if stds[j] != 0 else 1.0
            row.append((X[i][j] - means[j]) / s)
        X_std.append(row)
        
    y_mean = mean(y)
    y_std = [yi - y_mean for yi in y]
    
    # 2. Coordinate Descent
    beta_std = [0.0] * k
    # Precalculate column squared sums
    z = [sum(X_std[i][j]**2 for i in range(n)) for j in range(k)]
    
    iterations = 0
    alpha = lam / 2.0
    for it in range(max_iter):
        iterations += 1
        beta_old = beta_std[:]
        
        for j in range(k):
            # Calculate rho_j
            # rho_j = sum( X_ij * (y_i - sum(X_im * beta_m for m!=j)) )
            # Efficiently: y_pred = X @ beta_std; residual_minus_j = y_std - (y_pred - X_j * beta_j)
            y_pred = [vec_dot(X_std[i], beta_std) for i in range(n)]
            rho_j = 0.0
            for i in range(n):
                r_minus_j = y_std[i] - (y_pred[i] - X_std[i][j] * beta_std[j])
                rho_j += X_std[i][j] * r_minus_j
            
            if z[j] == 0:
                beta_std[j] = 0.0
            else:
                if j == 0:
                    beta_std[j] = rho_j / z[j]
                else:
                    # Soft-thresholding
                    if rho_j < -alpha:
                        beta_std[j] = (rho_j + alpha) / z[j]
                    elif rho_j > alpha:
                        beta_std[j] = (rho_j - alpha) / z[j]
                    else:
                        beta_std[j] = 0.0
        
        # Check convergence
        max_diff = max(abs(beta_std[j] - beta_old[j]) for j in range(k))
        if max_diff < tol:
            break
            
    # 3. Unstandardize
    beta_hat = [0.0] * k
    for j in range(1, k):
        s = stds[j] if stds[j] != 0 else 1.0
        beta_hat[j] = beta_std[j] / s
        
    sum_weighted_means = sum(beta_hat[j] * means[j] for j in range(1, k))
    beta_hat[0] = beta_std[0] + y_mean - sum_weighted_means
    
    y_hat = [vec_dot(X[i], beta_hat) for i in range(n)]
    residuals = vec_sub(y, y_hat)
    
    return {
        "beta_hat": beta_hat,
        "beta_std": beta_std,
        "y_hat": y_hat,
        "residuals": residuals,
        "iterations": iterations
    }


def plot_regularization_trace(X: list, y: list, lambdas: list, method: str = 'ridge'):
    """Plots the regularization trace using matplotlib."""
    import matplotlib.pyplot as plt
    
    coefs_std = []
    for lam in lambdas:
        if method.lower() == 'ridge':
            res = ridge_fit(X, y, lam)
        elif method.lower() == 'lasso':
            res = lasso_fit(X, y, lam)
        else:
            raise ValueError("method must be 'ridge' or 'lasso'")
        coefs_std.append(res['beta_std'][1:])
        
    # Transpose coefs_std to get traces for each feature
    traces = [[coefs_std[i][j] for i in range(len(coefs_std))] for j in range(len(coefs_std[0]))]
    
    plt.figure(figsize=(10, 6))
    for i, trace in enumerate(traces):
        plt.plot(lambdas, trace, label=f'Feature {i+1}')
        
    plt.xscale('log')
    plt.xlabel('Lambda (log scale)')
    plt.ylabel('Standardized Coefficients')
    plt.title(f'{method.capitalize()} Regularization Trace')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()
