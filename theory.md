#### 1. Linear Minimization Oracle (LMO) Implementation

In the Frank–Wolfe framework, at iteration \(t\), the linearized subproblem is:

\[
s_t = \arg\min_{y \in \Delta} \langle \nabla f(x_t), y \rangle,
\]

where \(x_t \in \Delta \subset \mathbb{R}^n\) is the current iterate and \(\Delta\) is the standard \(n\)-dimensional simplex:

\[
\Delta = \{ x \in \mathbb{R}^n \mid x_i \ge 0, \; \sum_{i=1}^{n} x_i = 1 \}.
\]

The role of LMO is to identify the feasible point in \(\Delta\) that minimizes the linear approximation of the objective at \(x_t\) and since the simplex is the convex hull of the standard basis vectors \(\{ e_1, e_2, \dots, e_n \}\), the LMO solution is always attained at a **vertex of the simplex**:

\[
s_t = e_{i^*}, \quad i^* = \arg\min_i [\nabla f(x_t)]_i.
\]

- \(e_i \in \mathbb{R}^n\) is the **unit basis vector** with 1 in the \(i\)-th coordinate and 0 elsewhere.  
- \(i^*\) is the index of the vertex corresponding to the **most promising coordinate** in the current gradient.

The LMO is implemented in code as:

```python
def LMO(gradient):
    i = np.argmin(gradient) # find index of the smallest value of grad
    s_fw = np.zeros_like(gradient)
    s_fw[i] = 1.0 # set i-th coordinate to one
    return s_fw, i
```
Note: remember that we pass -grad to LMO, that's why it is argmin
In simple terms: if you look at our simplex for n=3 it is a triangle, for n=4 it is a tetrathedron and at each iteration we have to pick up the right corner(based on the index of the smallest value of the gradient). Corners are bias vectors of our feasible set and by choosing the right corner we are showing the best direction of FW step. (very simple exactly how it should be for the simplex)

#### 2. Step-size strategies

##### Plan for Line Search Strategies

##### Classical Frank–Wolfe (FW)
- L2 Objective
  - Exact line search: cheap (closed-form for quadratic), maximizes progress per iteration → best choice.
  - Armijo: safe backup, adapts to gradient, slightly slower than exact.
  - Log-Adaptive: cheap alternative, early iterations get larger steps → faster initial progress.
  
- L0 Objective
  - Armijo: adaptive, cheap, guarantees monotone ascent even for steep exponential gradients → primary choice.
  - Log-Adaptive: early large steps help move weight toward promising clique vertices; practical and efficient.
  - Diminishing: very cheap baseline; progress may be slower due to non-adaptive step size.

---

##### Pairwise Frank–Wolfe (PFW)
- L2 Objective
  - Exact line search: closed-form along pairwise edge, maximizes progress → best choice.
  - Armijo: safe alternative, adapts to gradient along edge, slightly slower.
  - Log-Adaptive: early large steps allow fast exploration of promising vertices.
  
- L0 Objective
  - Armijo: adaptive, cheap, handles steep gradients along sparse directions.
  - Log-Adaptive: early large steps redistribute weight efficiently; cheap and practical.
---

##### Away-Step Frank–Wolfe (AFW)
- L2 Objective
  - Exact line search: removes weight from non-promising vertices optimally, cheap for quadratic → best choice.
  - Armijo: safe alternative, guarantees sufficient ascent along away direction.
  - Log-Adaptive: early large steps improve redistribution of weight; cheap.
  
- L0 Objective
  - Armijo: adaptive, cheap, works well for steep exponential gradients.
  - Log-Adaptive: early large steps redistribute weight efficiently; practical.

---

##### Projected Gradient method
- Since we compare it inly to FW classical, I would use the same methods, which will also (in theory) show that some of them are expensive because of the projections

---

Conclusions:
- For L0 objective exact line search is expensive for all FW variants
- Log adaptive is an interesting approach that is well-balanced and works good with high sparsity
- Armijo and Diminishing are cheap alternatives to exact line search, especially in cases with L0


