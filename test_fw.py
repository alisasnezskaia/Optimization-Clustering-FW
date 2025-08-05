from parse_graphs import load_dimacs_graph, create_optimization_matrix
from fw_benchmarking import benchmark_algorithms
import numpy as np

# Load adjacency matrix
W = load_dimacs_graph("graphs/C125.9.clq")
print(f"Graph loaded: {W.shape[0]} nodes, {np.sum(W)/2:.0f} edges")

# Create optimization matrix
alpha = 0.1
A = create_optimization_matrix(W, alpha)

# Run benchmarks
df = benchmark_algorithms(A, alpha=alpha)
print(df)

KNOWN_MAX_CLIQUE = 34
print("\n" + "="*50)
print(df[["Algorithm", "Clique Size", "Objective", "Time (s)"]])
print("="*50)
print(f"Best found clique size: {df['Clique Size'].max()} out of {KNOWN_MAX_CLIQUE}")
print(f"Success rate: {df['Clique Size'].max()/KNOWN_MAX_CLIQUE*100:.1f}%")
