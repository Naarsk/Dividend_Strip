import logging

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

logger = logging.getLogger(__name__)
logger.propagate = True  # should be True or omitted (default)

def soft_threshold(x, thresh):
    return np.sign(x) * np.maximum(np.abs(x) - thresh, 0.0)

def ADMM_elastic_net(x, y, alpha, l1_ratio, rho=1.0, max_iter=1000, tol=1e-4):

    """
        Solve multitask elastic net with positivity constraints for tensor X:
         X: I x K x H
         Y: I x H
         B: K x H

        Objective (sum over tasks h=1,...,H):
         0.5 * ||Y[:,h] - X[:,:,h] @ B[:,h]||_2^2 / I
         + alpha * ( (1 - l1_ratio)/2 * ||B[:,h]||_2^2 + l1_ratio * ||B[:,h]||_1 )
        subject to B >= 0

        Parameters:
        - X: ndarray (I x K x H)
        - Y: ndarray (I x H)
        - alpha: regularization strength
        - l1_ratio: mixing parameter between L1 and L2
        - rho, max_iter, tol: ADMM parameters

        Returns:
        - B: ndarray (K x H)
    """

    I, K, H = x.shape
    B = np.zeros((K, H))
    Z = np.zeros_like(B)
    U = np.zeros_like(B)

    chol_factors = []
    for h in range(H):
        XtX = x[:, :, h].T @ x[:, :, h]
        M = XtX/I + (rho + 2 * alpha * (1 - l1_ratio)) * np.eye(K)
        try:
            L = np.linalg.cholesky(M)
        except np.linalg.LinAlgError:
            logger.warning(f'Cholesky failed at task {h}, adding jitter.')
            jitter = 1e-8
            L = np.linalg.cholesky(M + jitter * np.eye(K))
        chol_factors.append(L)

    for iteration in range(max_iter):
        for h in range(H):
            xh = x[:, :, h]
            yh = y[:, h]
            q = xh.T @ yh/I + rho * (Z[:, h] - U[:, h])
            L = chol_factors[h]
            y_sol = np.linalg.solve(L, q)
            B[:, h] = np.linalg.solve(L.T, y_sol)

        V = B + U
        thresh = alpha * l1_ratio / rho
        shrink = 1 / (1 + 2 * alpha * (1 - l1_ratio) / rho)
        Z_old = Z.copy()
        Z = soft_threshold(V, thresh) * shrink
        Z = np.maximum(Z, 0)

        U += B - Z

        r_norm = np.linalg.norm(B - Z, ord='fro')
        s_norm = np.linalg.norm(rho * (Z - Z_old), ord='fro')

        if r_norm < tol and s_norm < tol:
            logger.info(f'Converged at iteration {iteration}')
            break

        if iteration == max_iter - 1:
            logger.warning("Regression did not converge")

    return Z

def cross_validate_elastic_net(x, y, l1_ratios, seed=49, cv_splits=5, n_alphas=50, alpha_min_ratio=1e-3, max_iter=1000):

    """
        Cross-validate multitask elastic net using ADMM over a grid of l1_ratios and alphas.

        Parameters:
        - X: ndarray (I x K x H)
        - Y: ndarray (I x H)
        - l1_ratios: list of floats, each in [0, 1]
        - cv_splits: int, number of CV folds
        - n_alphas: number of alphas per l1_ratio
        - alpha_min_ratio: smallest alpha is alpha_max * alpha_min_ratio

        Returns:
        - best_params: (l1_ratio, alpha)
    """

    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=seed)
    best_score = np.inf
    best_params = None

    for l1_ratio in l1_ratios:
        alphas = generate_alpha_grid_multitask(x, y, l1_ratio=l1_ratio, n_alphas=n_alphas,
                                               alpha_min_ratio=alpha_min_ratio)
        for alpha in alphas:
            val_scores = []
            for train_idx, val_idx in kf.split(x):
                x_train, x_val = x[train_idx], x[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                try:
                    B = ADMM_elastic_net(x_train, y_train, alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter)
                    y_pred = np.einsum("ikh,kh->ih", x_val, B)
                    mse = mean_squared_error(y_val, y_pred)
                except Exception as e:
                    logger.error(f"Failed for alpha={alpha:.4g}, l1_ratio={l1_ratio:.4g}: {e}")
                    mse = np.inf
                val_scores.append(mse)

            avg_score = np.mean(val_scores)
            logger.info(f'l1_ratio={l1_ratio:.4g}, alpha={alpha:.4f}, MSE={avg_score:.4g}')
            if avg_score < best_score:
                best_score = avg_score
                best_params = (l1_ratio, alpha)

    logger.info(f'Best params: l1_ratio={best_params[0]:.4g}, alpha={best_params[1]:.4g}, MSE={best_score:.4g}')
    return best_params

def generate_alpha_grid_multitask(x, y, l1_ratio=1.0, n_alphas=100, alpha_min_ratio=1e-3):

    """
        Generate a logarithmically spaced alpha grid for multitask elastic net.

        Parameters:
        - X: ndarray (I x K x H)
        - Y: ndarray (I x H)
        - l1_ratio: float in (0, 1]; mix between L1 and L2 penalties
        - n_alphas: int; number of alphas to generate
        - alpha_min_ratio: float; ratio of alpha_min / alpha_max

        Returns:
        - alphas: ndarray (n_alphas,)
    """

    I, K, H = x.shape
    alpha_max = 0.0

    for h in range(H):
        xh = x[:, :, h]
        yh = y[:, h]
        cov = xh.T @ yh / I
        alpha_max_h = np.max(np.abs(cov))
        if np.isfinite(alpha_max_h):
            alpha_max = max(alpha_max, alpha_max_h)

    if alpha_max <= 0 or not np.isfinite(alpha_max):
        logger.warning("Degenerate alpha_max detected. Falling back to 1e-3.")
        alpha_max = 1e-3

    if l1_ratio > 0:
        alpha_max /= l1_ratio
    else:
        raise ValueError("l1_ratio must be > 0 to generate a meaningful alpha grid")

    alphas = np.logspace(np.log10(alpha_max), np.log10(alpha_max * alpha_min_ratio), n_alphas)
    return alphas