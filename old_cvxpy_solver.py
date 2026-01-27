import cvxpy as cp
import numpy as np


def calculate_expected_loss(Y, H, D, num_sources):
    """
    Calculates the expected loss matrix L[j, t].
    """
    L_mat = np.zeros((num_sources, num_sources))  # (hypothesis j, domain t)

    for j in range(num_sources):
        # Loss calculation (Absolute Error)
        loss_vec = np.abs(H[:, j] - Y)

        for t in range(num_sources):
            p_t = D[:, t]
            # Expected loss: weighted sum
            expected_loss = np.sum(p_t * loss_vec)
            L_mat[j, t] = expected_loss

    return L_mat


def solve_convex_problem_mosek(
    Y,
    D,
    H,
    delta=1e-2,
    epsilon=1e-2,
    solver_type='SCS'
):
    """
    Solves the convex optimization problem using the specified solver.
    Returns weights w if successful, or None if infeasible/failed.
    """

    # --------------------------------------------------
    # Shape handling
    # --------------------------------------------------
    if D.ndim == 3:
        N_orig, C, k = D.shape
        D = D.reshape(-1, k)
        H = H.reshape(-1, k)
        Y = Y.reshape(-1)
    else:
        N, k = D.shape

    N, k = D.shape

    # --------------------------------------------------
    # Expected loss
    # --------------------------------------------------
    L_mat = calculate_expected_loss(Y, H, D, k)

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
        term = cp.sum(cp.multiply(
            D[:, j],
            -cp.rel_entr(w[j], Q[:, j])
        ))
        obj_terms.append(term)

    objective = cp.Maximize(cp.sum(obj_terms))

    # --------------------------------------------------
    # Constraints
    # --------------------------------------------------
    constraints = []
    constraints.append(cp.sum(w) == 1)
    constraints.append(cp.sum(Q, axis=1) == 1)
    constraints.append(cp.sum(R, axis=0) == 1)

    # --------------------------------------------------
    # KL constraint
    # (1/k) * sum_t sum_i p_it sum_j r_jt log(r_jt / q_ij) <= delta
    # --------------------------------------------------
    kl_terms = []
    for t in range(k):
        p_t = D[:, t]
        inner_sum_terms = []
        for j in range(k):
            re = cp.rel_entr(R[j, t], Q[:, j])
            weighted_re = cp.multiply(p_t, re)
            inner_sum_terms.append(cp.sum(weighted_re))
        kl_terms.append(cp.sum(inner_sum_terms))

    constraints.append((1.0 / k) * cp.sum(kl_terms) <= delta)

    # --------------------------------------------------
    # Loss constraints
    # sum_j r_jt L_j^t <= epsilon
    # --------------------------------------------------
    for t in range(k):
        constraints.append(
            cp.sum(cp.multiply(R[:, t], L_mat[:, t])) <= epsilon
        )

    # --------------------------------------------------
    # Solve
    # --------------------------------------------------
    prob = cp.Problem(objective, constraints)

    print(f"   >>> [CVXPY] Starting {solver_type} solve (N={N}, k={k})...")

    try:
        if solver_type == 'SCS':
            prob.solve(
                solver=cp.SCS,
                verbose=False,
                eps=1e-3,
                max_iters=5000
            )
        elif solver_type == 'MOSEK':
            prob.solve(solver=cp.MOSEK, verbose=False)
        elif solver_type == 'CLARABEL':
            prob.solve(solver=cp.CLARABEL, verbose=False)
        else:
            print(f"   !!! Unknown solver {solver_type}, defaulting to SCS")
            prob.solve(solver=cp.SCS, verbose=False, eps=1e-3)

    except Exception as e:
        print(f"   !!! Solver Exception: {e}")
        return None

    if prob.status not in ["optimal", "optimal_inaccurate"]:
        print(f"   !!! [CVXPY] Failed/Infeasible. Status: {prob.status}")
        return None

    print(f"   >>> [CVXPY] Solved! Weights: {np.round(w.value, 3)}")
    return w.value
