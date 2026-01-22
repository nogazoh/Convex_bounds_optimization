import cvxpy as cp
import numpy as np


def calculate_expected_loss(Y, H, D, num_sources):
    """
    Computes L_mat[j,t] = E_{i ~ p_t} [loss_j(i)]
    Assumes columns of D are normalized (sum_i p_{it} = 1)
    """
    L_mat = np.zeros((num_sources, num_sources))

    for j in range(num_sources):
        # loss per sample
        if H.ndim == 3:
            loss_vec = np.abs(H[:, :, j] - Y)
        else:
            loss_vec = np.abs(H[:, j] - Y)

        if loss_vec.ndim == 2:
            loss_vec = np.sum(loss_vec, axis=1) / 2.0

        for t in range(num_sources):
            p_t = D[:, t]
            L_mat[j, t] = np.sum(p_t * loss_vec)

    return L_mat


def solve_convex_problem_mosek(
    Y, D, H,
    delta=1e-2,
    epsilon=1e-2, L_mat=None,
    solver_type='SCS',

):
    # --------------------------------------------------
    # Shape handling
    # --------------------------------------------------
    if D.ndim == 3:
        D = D.reshape(-1, D.shape[2])

    N, k = D.shape

    # --------------------------------------------------
    # 🔧 CRITICAL FIX: normalize D to probabilities
    # --------------------------------------------------
    col_sums = D.sum(axis=0, keepdims=True)
    col_sums[col_sums == 0] = 1.0
    D = D / col_sums   # now sum_i D[i,t] = 1

    # --------------------------------------------------
    # Expected loss
    # --------------------------------------------------
    # L_mat = calculate_expected_loss(Y, H, D, k)

    # --------------------------------------------------
    # Variables
    # --------------------------------------------------
    w = cp.Variable(k, nonneg=True, name='w')
    Q = cp.Variable((N, k), nonneg=True, name='Q')
    R = cp.Variable((k, k), nonneg=True, name='R')

    # --------------------------------------------------
    # Objective:
    # sum_{i,j} w_j p_{ij} log(q_{ij} / w_j)
    # --------------------------------------------------
    obj_terms = []
    for j in range(k):
        obj_terms.append(
            cp.sum(cp.multiply(D[:, j], -cp.rel_entr(w[j], Q[:, j])))
        )

    objective = cp.Maximize(cp.sum(obj_terms))

    # --------------------------------------------------
    # Constraints
    # --------------------------------------------------
    constraints = [
        cp.sum(w) == 1,
        cp.sum(Q, axis=1) == 1,
        cp.sum(R, axis=0) == 1,
    ]

    # --------------------------------------------------
    # KL constraint:
    # (1/k) * sum_t sum_i p_it sum_j r_jt log(r_jt / q_ij) <= delta
    # --------------------------------------------------
    kl_terms = []
    for t in range(k):
        p_t = D[:, t]  # already normalized
        inner = []
        for j in range(k):
            re = cp.rel_entr(R[j, t], Q[:, j])
            inner.append(cp.sum(cp.multiply(p_t, re)))
        kl_terms.append(cp.sum(inner))

    constraints.append((1.0 / k) * cp.sum(kl_terms) <= delta)

    # --------------------------------------------------
    # Loss constraints:
    # sum_j r_jt L_j^t <= epsilon
    # --------------------------------------------------
    if isinstance(epsilon, (list, np.ndarray)):
        for t in range(k):
            constraints.append(
                cp.sum(cp.multiply(R[:, t], L_mat[:, t])) <= epsilon[t]
            )
    else:
        for t in range(k):
            constraints.append(
                cp.sum(cp.multiply(R[:, t], L_mat[:, t])) <= epsilon
            )

    # --------------------------------------------------
    # Solve
    # --------------------------------------------------
    prob = cp.Problem(objective, constraints)

    try:
        prob.solve(
            solver=cp.SCS,
            verbose=False,
            eps=1e-4,
            max_iters=10000
        )
    except Exception:
        return None

    if prob.status not in ["optimal", "optimal_inaccurate"]:
        return None

    return w.value
