import cvxpy as cp
import numpy as np
from cvxpy_solver import calculate_expected_loss


# ============================================================
# Problem (3.31): Smoothed KL MSDA (NEW epsilon constraint via q̄)
# ============================================================
def solve_convex_problem_smoothed_kl_331(
    Y,
    D,
    H,
    epsilon=1e-2,
    eta=1e-2,
    solver_type="SCS",
    q_min=1e-12,
    w_min=1e-12,
    scs_eps=1e-4,
    scs_max_iters=20000,
    normalize_D=True,
):
    """
    Solves your modified problem ("3_31"):

        max_{w,Q}  sum_{i,j} D_{ij} * log(Q_{ij} / w_j)
                   + (eta / (kN)) * sum_{i,j} log(Q_{ij})

        s.t.  w in simplex (and >= w_min)
              Q rows in simplex (and >= q_min)
              NEW risk constraints (replace w^T L[:,t] <= eps):
                  For each t:
                      sum_{i,j} Q_{ij} D_{ij} (L_mat[j,t] - eps_t) <= 0

    Notes:
    - D expected shape: (N,k). If D is (N,C,k) replicated, we reduce to (N,k).
    - If normalize_D=True: columns of D are normalized to sum_i D[i,t]=1 (same as before).
    """

    # --------------------------------------------------------
    # Shape handling (SAFE)
    # --------------------------------------------------------
    if D.ndim == 3:
        D = D[:, 0, :]  # (N,k)

    if D.ndim != 2:
        raise ValueError(f"D must be 2D (N,k) after reduction. Got shape={D.shape}")

    N, k = D.shape

    # --------------------------------------------------------
    # Normalize D columns to probabilities
    # --------------------------------------------------------
    if normalize_D:
        col_sums = D.sum(axis=0, keepdims=True)
        col_sums[col_sums == 0] = 1.0
        D = D / col_sums  # now sum_i D[i,t] = 1

    # --------------------------------------------------------
    # Expected losses L_mat[j,t]
    # --------------------------------------------------------
    L_mat = calculate_expected_loss(Y, H, D, k)  # shape (k,k): [j,t]

    # --------------------------------------------------------
    # Variables
    # --------------------------------------------------------
    w = cp.Variable(k, nonneg=True, name="w")
    Q = cp.Variable((N, k), nonneg=True, name="Q")

    # --------------------------------------------------------
    # Objective
    # main: sum_{i,j} D_{ij} * log(Q_{ij}/w_j)
    # implemented via -rel_entr(w_j, Q_ij) = w_j*log(Q_ij/w_j)
    # then weighted by D[:,j] and summed over i
    # --------------------------------------------------------
    main_terms = [
        cp.sum(cp.multiply(D[:, j], -cp.rel_entr(w[j], Q[:, j])))
        for j in range(k)
    ]
    main_obj = cp.sum(main_terms)

    # Smooth barrier term on Q
    smooth_obj = (eta / (k * N)) * cp.sum(cp.log(Q))

    objective = cp.Maximize(main_obj + smooth_obj)

    # --------------------------------------------------------
    # Constraints (simplex + lower bounds)
    # --------------------------------------------------------
    constraints = [
        cp.sum(w) == 1,
        w >= w_min,
        cp.sum(Q, axis=1) == 1,
        Q >= q_min,
    ]

    # --------------------------------------------------------
    # NEW epsilon constraints:
    #   S_j := sum_i D_{ij} Q_{ij}
    #   For each t:  sum_j S_j (L_mat[j,t] - eps_t) <= 0
    # This equals: sum_{i,j} D_{ij} Q_{ij} (L_mat[j,t] - eps_t) <= 0
    # --------------------------------------------------------
    S = cp.sum(cp.multiply(D, Q), axis=0)  # shape (k,)

    if isinstance(epsilon, (list, tuple, np.ndarray)):
        eps_vec = np.asarray(epsilon).reshape(-1)
        if eps_vec.shape[0] != k:
            raise ValueError(f"epsilon vector must have length k={k}. Got {eps_vec.shape[0]}")
        for t in range(k):
            constraints.append(S @ (L_mat[:, t] - eps_vec[t]) <= 0)
    else:
        eps_val = float(epsilon)
        for t in range(k):
            constraints.append(S @ (L_mat[:, t] - eps_val) <= 0)

    # --------------------------------------------------------
    # Solve
    # --------------------------------------------------------
    prob = cp.Problem(objective, constraints)

    try:
        st = solver_type.upper()
        if st == "MOSEK":
            prob.solve(solver=cp.MOSEK, verbose=False)
        elif st == "CLARABEL":
            prob.solve(solver=cp.CLARABEL, verbose=False)
        else:
            prob.solve(solver=cp.SCS, verbose=False, eps=scs_eps, max_iters=scs_max_iters)
    except Exception as e:
        print(f"[3.31] Solver exception: {e}")
        return None, None

    if prob.status not in ["optimal", "optimal_inaccurate"]:
        print(f"[3.31] Optimization failed | status={prob.status}")
        return None, None

    return np.asarray(w.value).reshape(-1), np.asarray(Q.value)
