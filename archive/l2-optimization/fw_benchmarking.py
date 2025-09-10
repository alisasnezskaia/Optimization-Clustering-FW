import time
import pandas as pd
import numpy as np
from archive.l2_fw_variants import frank_wolfe, away_step_fw, pairwise_fw, l2_objective


def sparsity(x):
    return np.count_nonzero(x)

def benchmark_algorithms(A, alpha=0.1, max_iters=300, tol=1e-4):
    algorithms = {
        "Frank-Wolfe": frank_wolfe,
        "Away-Step FW": away_step_fw,
        "Pairwise FW": pairwise_fw
    }

    benchmark_results = []

    for name, algo in algorithms.items():
        start_time = time.time()
        x, history = algo(A, alpha, max_iters=max_iters, tol=tol)
        elapsed = time.time() - start_time

        benchmark_results.append({
            "Algorithm": name,
            "Objective": l2_objective(x, A, alpha),
            "Iterations": len(history),
            "Time (s)": round(elapsed, 4),
            "Sparsity": sparsity(x)
        })

    df = pd.DataFrame(benchmark_results)
    return df
