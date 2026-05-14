import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Union, List


def standardize(X: np.ndarray) -> tuple:
    """Standardizes the features using Z-score (mean=0, std=1).

    Args:
        X (np.ndarray): The feature matrix of shape (n, p).

    Returns:
        tuple: (X_std, mean, std)
            X_std: Standardized feature matrix.
            mean: Mean of each feature.
            std: Standard deviation of each feature.
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    # Avoid division by zero
    std[std == 0] = 1.0
    X_std = (X - mean) / std
    return X_std, mean, std


def ridge_fit(X: np.ndarray, y: np.ndarray, lam: float) -> Dict[str, Union[np.ndarray, float]]:
    """Fits a Ridge regression model using the closed-form solution.

    Formula: beta = (X^T * X + lam * I)^-1 * X^T * y

    Args:
        X (np.ndarray): The feature matrix of shape (n, p).
        y (np.ndarray): The target vector of shape (n,).
        lam (float): Regularization parameter (lambda).

    Returns:
        Dict[str, Union[np.ndarray, float]]: A dictionary containing:
            - beta_hat: Coefficients on the original scale (including intercept).
            - beta_std: Coefficients on the standardized scale.
            - y_hat: Predicted values.
            - residuals: Difference between actual and predicted values.
    """
    n, p = X.shape
    X_std, X_mean, X_std_dev = standardize(X)
    
    # Center y
    y_mean = np.mean(y)
    y_centered = y - y_mean
    
    # Closed-form solution for standardized scale (no intercept in X_std yet)
    # Most implementations don't penalize the intercept, but we strictly follow the 
    # formula on the augmented matrix if needed. Here we fit standardized data.
    I = np.eye(p)
    beta_std_core = np.linalg.solve(X_std.T @ X_std + lam * I, X_std.T @ y_centered)
    
    # Reconstruct original scale coefficients
    # beta_j = beta_std_j * (std_y / std_x_j)
    # Since we didn't scale y by its std (only centered), std_y is implicitly 1.0 here
    # or we can assume y was not scaled.
    beta_hat_core = beta_std_core / X_std_dev
    intercept = y_mean - np.dot(X_mean, beta_hat_core)
    
    beta_hat = np.concatenate(([intercept], beta_hat_core))
    
    # Standardized beta (including intercept which is effectively 0 for centered data)
    beta_std = np.concatenate(([0.0], beta_std_core))
    
    X_with_intercept = np.column_stack([np.ones(n), X])
    y_hat = X_with_intercept @ beta_hat
    residuals = y - y_hat
    
    return {
        "beta_hat": beta_hat,
        "beta_std": beta_std,
        "y_hat": y_hat,
        "residuals": residuals
    }


def soft_threshold(a: float, b: float) -> float:
    """Applies the soft-thresholding operator.

    Args:
        a (float): The value to threshold.
        b (float): The threshold (lambda).

    Returns:
        float: The thresholded value.
    """
    return np.sign(a) * np.maximum(0, np.abs(a) - b)


def lasso_fit(X: np.ndarray, y: np.ndarray, lam: float, max_iter: int = 1000, tol: float = 1e-4) -> Dict[str, Union[np.ndarray, float]]:
    """Fits a Lasso regression model using Coordinate Descent.

    Args:
        X (np.ndarray): The feature matrix of shape (n, p).
        y (np.ndarray): The target vector of shape (n,).
        lam (float): Regularization parameter (lambda).
        max_iter (int): Maximum number of iterations. Defaults to 1000.
        tol (float): Convergence tolerance. Defaults to 1e-4.

    Returns:
        Dict[str, Union[np.ndarray, float]]: A dictionary containing:
            - beta_hat: Coefficients on the original scale (including intercept).
            - beta_std: Coefficients on the standardized scale.
            - y_hat: Predicted values.
            - residuals: Difference between actual and predicted values.
    """
    n, p = X.shape
    X_std, X_mean, X_std_dev = standardize(X)
    
    y_mean = np.mean(y)
    y_centered = y - y_mean
    
    beta_std_core = np.zeros(p)
    
    for _ in range(max_iter):
        beta_old = beta_std_core.copy()
        
        for j in range(p):
            # Calculate residual without feature j
            r = y_centered - (X_std @ beta_std_core - X_std[:, j] * beta_std_core[j])
            rho = X_std[:, j] @ r
            
            # Coordinate update
            beta_std_core[j] = soft_threshold(rho, lam) / (X_std[:, j] @ X_std[:, j])
            
        if np.linalg.norm(beta_std_core - beta_old) < tol:
            break
            
    # Transform back to original scale
    beta_hat_core = beta_std_core / X_std_dev
    intercept = y_mean - np.dot(X_mean, beta_hat_core)
    
    beta_hat = np.concatenate(([intercept], beta_hat_core))
    beta_std = np.concatenate(([0.0], beta_std_core))
    
    X_with_intercept = np.column_stack([np.ones(n), X])
    y_hat = X_with_intercept @ beta_hat
    residuals = y - y_hat
    
    return {
        "beta_hat": beta_hat,
        "beta_std": beta_std,
        "y_hat": y_hat,
        "residuals": residuals
    }


def plot_regularization_trace(X: np.ndarray, y: np.ndarray, lambdas: List[float], method: str = 'ridge'):
    """Plots the regularization trace of the coefficients.

    Args:
        X (np.ndarray): The feature matrix.
        y (np.ndarray): The target vector.
        lambdas (List[float]): A list of lambda values to iterate over.
        method (str): Either 'ridge' or 'lasso'. Defaults to 'ridge'.
    """
    coefs = []
    fit_func = ridge_fit if method.lower() == 'ridge' else lasso_fit
    
    for lam in lambdas:
        res = fit_func(X, y, lam)
        coefs.append(res["beta_std"][1:])  # Exclude intercept
        
    coefs = np.array(coefs)
    
    plt.figure(figsize=(10, 6))
    for i in range(coefs.shape[1]):
        plt.plot(lambdas, coefs[:, i], label=f'Feature {i+1}')
        
    plt.xscale('log')
    plt.xlabel('Lambda (log scale)')
    plt.ylabel('Standardized Coefficients')
    plt.title(f'{method.capitalize()} Regularization Trace')
    plt.legend()
    plt.grid(True)
    plt.show()
