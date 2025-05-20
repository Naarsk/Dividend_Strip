import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold


def soft_threshold(X, thresh):
    return np.sign(X) * np.maximum(np.abs(X) - thresh, 0.0)

def multitask_elastic_net(X, Y, alpha, l1_ratio, rho=1.0, max_iter=1000, tol=1e-4):
    """
    Solve multitask elastic net with positivity constraints for tensor X:
      X: I x K x H
      Y: I x H
      B: K x H

    Objective (sum over tasks h=1..H):
      0.5 * ||Y[:,h] - X[:,:,h] @ B[:,h]||_2^2
      + alpha * ( (1 - l1_ratio) * ||B[:,h]||_2^2 + l1_ratio * ||B[:,h]||_1 )
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
    I, K, H = X.shape

    # Initialize variables
    B = np.zeros((K, H))
    Z = np.zeros_like(B)
    U = np.zeros_like(B)

    # Precompute Cholesky factorizations for each task to solve linear system efficiently
    chol_factors = []
    for h in range(H):
        XtX = X[:, :, h].T @ X[:, :, h]
        # Matrix to invert: XtX + rho*I + 2*alpha*(1-l1_ratio)*I
        M = XtX + (rho + 2 * alpha * (1 - l1_ratio)) * np.eye(K)
        try:
            L = np.linalg.cholesky(M)
        except np.linalg.LinAlgError:
            # Add small jitter if not positive definite
            print(f'Cholesky failed at task {h}, adding jitter.')
            jitter = 1e-8
            L = np.linalg.cholesky(M + jitter * np.eye(K))
        chol_factors.append(L)

    for iteration in range(max_iter):
        B_prev = B.copy()

        # B update (solve for each h)
        for h in range(H):
            Xh = X[:, :, h]
            Yh = Y[:, h]
            q = Xh.T @ Yh + rho * (Z[:, h] - U[:, h])
            L = chol_factors[h]

            # Solve L L.T B[:,h] = q
            y = np.linalg.solve(L, q)
            B[:, h] = np.linalg.solve(L.T, y)

        # Z update (proximal step of elastic net + positivity)
        V = B + U
        thresh = alpha * l1_ratio / rho
        shrink = 1 / (1 + 2 * alpha * (1 - l1_ratio) / rho)
        Z_old = Z.copy()

        Z = soft_threshold(V, thresh) * shrink
        Z = np.maximum(Z, 0)  # positivity constraint

        # Dual update
        U += B - Z

        # Check convergence
        r_norm = np.linalg.norm(B - Z, ord='fro')
        s_norm = np.linalg.norm(rho * (Z - Z_old), ord='fro')

        if r_norm < tol and s_norm < tol:
            print(f'Converged at iteration {iteration}')
            break

    return Z

def cross_validate_elastic_net(X, y, alphas, l1_ratios, cv_splits=10, positive = True):
    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=42)

    best_score = np.inf
    best_params = None

    for alpha in alphas:
        for l1_ratio in l1_ratios:
            assert np.isfinite(alpha) and alpha >= 0
            assert np.isfinite(l1_ratio) and 0 <= l1_ratio <= 1

            val_scores = []

            for train_idx, val_idx in kf.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                Y_train, Y_val = y[train_idx], y[val_idx]

                B = multitask_elastic_net(X_train, Y_train, alpha=alpha, l1_ratio=l1_ratio)

                Y_pred = np.einsum("ikh,kh->ih", X_val, B)
                mse = mean_squared_error(Y_val, Y_pred)
                val_scores.append(mse)

            avg_score = np.mean(val_scores)
            print(f'alpha={alpha}, l1_ratio={l1_ratio}, MSE={avg_score:.4f}')

            if avg_score < best_score:
                best_score = avg_score
                best_params = (alpha, l1_ratio)

    print(f'Best params: alpha={best_params[0]}, l1_ratio={best_params[1]}, MSE={best_score:.4f}')
    return best_params

def generate_alpha_grid(X, y, n_alphas=100, alpha_min_ratio=1e-3):
    n_samples = X.shape[0]
    alpha_max = np.max(np.abs(X.T @ y)) / n_samples
    if not np.isfinite(alpha_max) or alpha_max <= 0:
        alpha_max = 1e-4  # or a small positive fallback
    alphas = np.logspace(np.log10(alpha_max), np.log10(alpha_max * alpha_min_ratio), n_alphas)
    return alphas

