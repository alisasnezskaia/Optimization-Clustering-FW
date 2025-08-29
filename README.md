# Max-Clique Problem via Frank-Wolfe Variants

**Group 37** - Optimization Methods for Clustering

## Project Overview

This project explores continuous optimization approaches for approximating the maximum clique in a graph. It implements several Frank-Wolfe (FW) variants, including Standard Frank-Wolfe, Away-Step Frank-Wolfe and Pairwise Frank-Wolfe.

These methods are applied on L2 and L0 relaxations of the combinatorial max-clique problem, with flexible step size strategies (exact line search, Armijo, diminishing, logarithmic).

Additionally, a Projected Gradient (PG) method on the simplex is implemented for comparison, highlighting differences in convergence speed, objective values, extracted clique size, and computational complexity.

## Objectives
<img width="1054" height="355" alt="image" src="https://github.com/user-attachments/assets/92fe9510-1f46-4f02-9c0d-06728fb15ae3" />


### Theoretical Analysis
- In-depth study of Frank-Wolfe algorithm theory
- Review of referenced papers:
  - `FW.pdf` - Core Frank-Wolfe theory
  - `FW_variants.pdf` - Algorithm variants
  - `FW_survey.pdf` - Comprehensive survey

### Implementation Requirements
Develop efficient implementations of three algorithms:
1. **Frank-Wolfe** (standard)
2. **Pairwise Frank-Wolfe**
3. **Away Step Frank-Wolfe**

**Optimization Tools:** Line search, Linear Minimization Oracle (LMO), and other efficiency enhancements

### Testing & Evaluation
Test algorithms on two problem formulations:
- **L2 regularized max-clique problem** (Section 3, equation 31 in `clustering.pdf`)
- **L0 regularized max-clique problem** (Section 3, equation 33 in `clustering.pdf`)

**Datasets:** 4 DIMACS Instances

**Analysis:** Results visualization through plots and tables

