import numpy as np

def load_dimacs_graph(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    nodes = 0
    edges = []
    
    for line in lines:
        if line.startswith('p'):
            parts = line.strip().split()
            nodes = int(parts[2])
        elif line.startswith('e'):
            parts = line.strip().split()
            i, j = int(parts[1]) - 1, int(parts[2]) - 1  # 0-based indexing
            edges.append((i, j))
    
    A = np.zeros((nodes, nodes))
    for i, j in edges:
        A[i, j] = 1
        A[j, i] = 1  # since the graph is undirected
    return A
