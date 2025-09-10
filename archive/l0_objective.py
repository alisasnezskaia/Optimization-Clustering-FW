import numpy as np

def l0_objective(x, A, alpha, beta):
    """Objective function for L0 regularized max-clique problem."""
    sparsity_penalty = beta * np.count_nonzero(x)
    return x.T @ A @ x - sparsity_penalty

def l0_gradient(x, A):
    """Gradient of the L0 objective function."""
    return 2 * A @ x

# formulation according to the paper 
def obj_l0(x, A, alpha=0.07, beta=5):
    val = x.T @ (A @ x)
    return float(val) + alpha * np.sum(np.exp(-beta * x) - 1)

def grad_l0(x, A, alpha=0.07, beta=5):
    return 2 * (A @ x) - alpha * beta * np.exp(-beta * x)