import numpy as np

def l0_objective(x, A, alpha, beta):
    """Objective function for L0 regularized max-clique problem."""
    sparsity_penalty = beta * np.count_nonzero(x)
    return x.T @ A @ x - sparsity_penalty

def l0_gradient(x, A):
    """Gradient of the L0 objective function."""
    return 2 * A @ x
