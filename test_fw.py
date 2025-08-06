from parse_graphs import load_dimacs_graph, create_optimization_matrix
from fw_benchmarking import benchmark_algorithms
from l0_objective import l0_objective
from visualization import plot_convergence, compare_algorithms
import numpy as np

# Load adjacency matrix
W = load_dimacs_graph("graphs/keller4.clq")
print(f"Graph loaded: {W.shape[0]} nodes, {np.sum(W)/2:.0f} edges")

# Create optimization matrix
alpha = 0.1
A = create_optimization_matrix(W, alpha)

# Run benchmarks
df = benchmark_algorithms(A, alpha=alpha)
print(df)

# Visualize results
compare_algorithms(df)

KNOWN_MAX_CLIQUE = 11
print("\n" + "="*50)
print(df[["Algorithm", "Clique Size", "Objective", "Time (s)"]])
print("="*50)
print(f"Best found clique size: {df['Clique Size'].max()} out of {KNOWN_MAX_CLIQUE}")
print(f"Success rate: {df['Clique Size'].max()/KNOWN_MAX_CLIQUE*100:.1f}%")

# Test L0 regularized problem
beta = 0.5  # Sparsity penalty
x = np.ones(W.shape[0]) / W.shape[0]  # Initial solution
l0_obj = l0_objective(x, A, alpha, beta)
print(f"L0 Objective Value: {l0_obj}")
