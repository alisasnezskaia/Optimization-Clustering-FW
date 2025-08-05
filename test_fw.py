from parse_graphs import load_dimacs_graph
from fw_benchmarking import benchmark_algorithms

A = load_dimacs_graph("graphs/C125.9.clq")  
df = benchmark_algorithms(A, alpha=0.1)
print(df)