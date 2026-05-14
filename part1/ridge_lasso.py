import numpy as np

def ridge_fit(X: np.ndarray, y: np.ndarray, lam: float) -> dict:
    """Estimates Ridge Regression coefficients (L2 regularization).

    Formula: beta = (X^T X + lam * I)^-1 X^T y

    Args:
        X (np.ndarray): Design matrix of shape (n, p+1) including intercept column.
        y (np.ndarray): Target vector of shape (n,).
        lam (float): Regularization parameter (penalty).

    Returns:
        dict: A dictionary containing:
            - beta_hat: Coefficients on the original scale.
            - beta_std: Coefficients on the standardized scale.
            - y_hat: Predicted values.
            - residuals: Residual vector.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, k = X.shape

    # 1. Automatic standardization (centering and scaling)
    means = np.mean(X[:, 1:], axis=0)
    stds = np.std(X[:, 1:], axis=0)
    stds[stds == 0] = 1.0  

    X_std = np.copy(X)
    X_std[:, 1:] = (X[:, 1:] - means) / stds

    y_mean = np.mean(y)
    y_std = y - y_mean

    # 2. Closed-form solution: (X^T X + lam * I) * beta = X^T y
    I = np.eye(k)
    XtX = X_std.T @ X_std
    Xty = X_std.T @ y_std

    beta_std = np.linalg.solve(XtX + lam * I, Xty)

    # 3. Unstandardize to original scale
    beta_hat = np.zeros(k)
    beta_hat[1:] = beta_std[1:] / stds
    beta_hat[0] = beta_std[0] + y_mean - np.sum(beta_hat[1:] * means)

    y_hat = X @ beta_hat
    residuals = y - y_hat

    return {
        "beta_hat": beta_hat,
        "beta_std": beta_std,
        "y_hat": y_hat,
        "residuals": residuals
    }


def lasso_fit(X: np.ndarray, y: np.ndarray, lam: float, tol: float = 1e-4, max_iter: int = 1000) -> dict:
    """Estimates Lasso Regression coefficients (L1 regularization) using Coordinate Descent.

    Args:
        X (np.ndarray): Design matrix of shape (n, p+1) including intercept column.
        y (np.ndarray): Target vector of shape (n,).
        lam (float): Regularization parameter (penalty).
        tol (float): Convergence tolerance for stopping criteria.
        max_iter (int): Maximum number of iterations.

    Returns:
        dict: A dictionary containing:
            - beta_hat: Coefficients on the original scale.
            - beta_std: Coefficients on the standardized scale.
            - y_hat: Predicted values.
            - residuals: Residual vector.
            - iterations: Number of iterations performed.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, k = X.shape

    # 1. Standardization
    means = np.mean(X[:, 1:], axis=0)
    stds = np.std(X[:, 1:], axis=0)
    stds[stds == 0] = 1.0 

    X_std = np.copy(X)
    X_std[:, 1:] = (X[:, 1:] - means) / stds

    y_mean = np.mean(y)
    y_std = y - y_mean

    # 2. Coordinate Descent
    beta_std = np.zeros(k)
    z = np.sum(X_std ** 2, axis=0)  

    iterations = 0
    alpha = lam / 2.0
    for it in range(max_iter):
        iterations += 1
        beta_old = np.copy(beta_std)

        for j in range(k):
            y_pred_minus_j = X_std @ beta_std - X_std[:, j] * beta_std[j]
            rho_j = X_std[:, j] @ (y_std - y_pred_minus_j)

            if z[j] == 0:
                beta_std[j] = 0
            else:
                # Soft-thresholding operator
                if rho_j < -alpha:
                    beta_std[j] = (rho_j + alpha) / z[j]
                elif rho_j > alpha:
                    beta_std[j] = (rho_j - alpha) / z[j]
                else:
                    beta_std[j] = 0.0

        if np.max(np.abs(beta_std - beta_old)) < tol:
            break

    # 3. Unstandardize to original scale
    beta_hat = np.zeros(k)
    beta_hat[1:] = beta_std[1:] / stds
    beta_hat[0] = beta_std[0] + y_mean - np.sum(beta_hat[1:] * means)

    y_hat = X @ beta_hat
    residuals = y - y_hat

    return {
        "beta_hat": beta_hat,
        "beta_std": beta_std,
        "y_hat": y_hat,
        "residuals": residuals,
        "iterations": iterations
    }


def plot_regularization_trace(X: np.ndarray, y: np.ndarray, lambdas: list, method: str = 'ridge'):
    """Plots the regularization trace of coefficients against lambda.

    Args:
        X (np.ndarray): Design matrix of shape (n, p+1).
        y (np.ndarray): Target vector.
        lambdas (list): List of lambda values to evaluate.
        method (str): Regularization method, either 'ridge' or 'lasso'.
    """
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

    coefs_std = np.array(coefs_std)

    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    ax.plot(lambdas, coefs_std)
    ax.set_xscale('log')

    plt.xlabel('Lambda (log scale)')
    plt.ylabel('Standardized Coefficients')
    plt.title(f'{method.capitalize()} Regularization Trace')
    plt.axis('tight')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()
