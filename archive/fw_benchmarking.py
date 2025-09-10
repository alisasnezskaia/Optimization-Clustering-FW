import time
import pandas as pd
import numpy as np
from archive.l2_fw_variants import frank_wolfe, away_step_fw, pairwise_fw, l2_objective


def sparsity(x):
    return np.count_nonzero(x)


def extract_clique_from_solution(x, W, threshold=0.01):
    """
    Extracts a clique from the solution vector x using a greedy strategy.
    """
    candidates = np.argsort(-x)  # sort by weight, descending
    clique = []

    for i in candidates:
        if x[i] < threshold:
            break
        if all(W[i, j] == 1 for j in clique):
            clique.append(i)

    return clique


def recover_adjacency_from_A(A, alpha):
    """
    Recover adjacency matrix W from matrix A = -W + alpha * I
    """
    W = -(A - alpha * np.eye(A.shape[0]))
    W = np.maximum(W, 0)  # Ensure non-negative
    W = (W > 0.5).astype(int)  # Threshold to get binary matrix
    np.fill_diagonal(W, 0)  # Remove self-loops
    return W


def benchmark_algorithms(A, alpha=0.1, max_iters=300, tol=1e-4):
    algorithms = {
        "Frank-Wolfe": frank_wolfe,
        "Away-Step FW": away_step_fw,
        "Pairwise FW": pairwise_fw
    }

    benchmark_results = []
    W = recover_adjacency_from_A(A, alpha)

    for name, algo in algorithms.items():
        start_time = time.time()
        x, history = algo(A, alpha, max_iters=max_iters, tol=tol)
        elapsed = time.time() - start_time

        clique = extract_clique_from_solution(x, W)
        clique_size = len(clique)

        benchmark_results.append({
            "Algorithm": name,
            "Objective": l2_objective(x, A, alpha),
            "Iterations": len(history),
            "Time (s)": round(elapsed, 4),
            "Sparsity": sparsity(x),
            "Clique Size": clique_size,
            "Clique Nodes": clique  # you could skip this if you want a cleaner table
        })

    df = pd.DataFrame(benchmark_results)
    return df
