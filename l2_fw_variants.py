import numpy as np

# === Utility Functions ===

def l2_objective(x, A, alpha):
    """Objective function for A = -W + alpha*I"""
    return x.T @ A @ x

def l2_gradient(x, A, alpha):
    """Gradient of the objective function"""
    return 2 * A @ x

def lmo_simplex(grad):
    e = np.zeros_like(grad)
    e[np.argmin(grad)] = 1.0
    return e

def get_active_set(x, tol=1e-10):
    n = len(x)
    return [np.eye(n)[i] for i in range(n) if x[i] > tol]

def worst_vertex(grad, active_set):
    worst = None
    worst_value = -np.inf
    for v in active_set:
        dot = grad @ v
        if dot > worst_value:
            worst_value = dot
            worst = v
    return worst

def exact_line_search(x, d, A):
    """Exact line search for quadratic objective"""
    numerator = d.T @ A @ x
    denominator = d.T @ A @ d
    if abs(denominator) < 1e-12:
        return 1.0
    gamma = -numerator / denominator
    return np.clip(gamma, 0, 1)

# === Variant 1: Standard Frank-Wolfe ===

def frank_wolfe(A, alpha, max_iters=300, tol=1e-4):
    n = A.shape[0]
    x = np.ones(n) / n
    history = []

    for k in range(max_iters):
        grad = l2_gradient(x, A, alpha)
        s = lmo_simplex(grad)
        d = s - x
        gap = -grad @ d
        if gap < tol:
            break
        gamma = exact_line_search(x, d, A)
        x = x + gamma * d
        history.append(l2_objective(x, A, alpha))
    return x, history

# === Variant 2: Away-Step Frank-Wolfe ===

def away_step_fw(A, alpha, max_iters=300, tol=1e-4):
    n = A.shape[0]
    x = np.ones(n) / n
    history = []

    for k in range(max_iters):
        grad = l2_gradient(x, A, alpha)
        active_set = get_active_set(x)
        s = lmo_simplex(grad)
        v = worst_vertex(grad, active_set)

        d_fw = s - x
        d_away = x - v
        gap_fw = -grad @ d_fw
        gap_away = -grad @ d_away

        if max(gap_fw, gap_away) < tol:
            break

        if gap_fw >= gap_away:
            d = d_fw
            gamma_max = 1.0
        else:
            d = d_away
            idx = np.argmax([np.allclose(v, np.eye(n)[i]) for i in range(n)])
            gamma_max = x[idx]  # FIXED: Correct upper bound

        gamma = min(gamma_max, exact_line_search(x, d, A))
        x = x + gamma * d
        history.append(l2_objective(x, A, alpha))
    return x, history

# === Variant 3: Pairwise Frank-Wolfe ===

def pairwise_fw(A, alpha, max_iters=300, tol=1e-4):
    n = A.shape[0]
    x = np.ones(n) / n
    history = []

    for k in range(max_iters):
        grad = l2_gradient(x, A, alpha)
        active_set = get_active_set(x)
        s = lmo_simplex(grad)
        v = worst_vertex(grad, active_set)

        d = s - v
        idx = np.argmax([np.allclose(v, np.eye(n)[i]) for i in range(n)])
        gamma_max = x[idx]
        gap = -grad @ d
        if gap < tol:
            break

        gamma = min(gamma_max, exact_line_search(x, d, A))
        x = x + gamma * d
        history.append(l2_objective(x, A, alpha))
    return x, history
