import numpy as np
import scipy
from scipy.io import mmread

# Parses a DIMACS .clq file and converts it into an adjacency matrix
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
    
    W = np.zeros((nodes, nodes))
    for i, j in edges:
        W[i, j] = 1
        W[j, i] = 1  # since the graph is undirected
    return W

# Loads a MATLAB .mlx file and extracts the stored matrix
def get_adj_matrix(path):
    M = mmread(path).tocsr()
    # check if the loaded matrix is symmetric
    is_symmetric = (M != M.T).nnz == 0
    if not is_symmetric:
        M = M.maximum(M.T)

    M.setdiag(0)
    M.eliminate_zeros()
    return M

def create_optimization_matrix(W, alpha):
    """Convert adjacency matrix W to optimization matrix A = -W + alpha*I"""
    A = -W + alpha * np.eye(W.shape[0])
    return A
