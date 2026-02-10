import cvxpy as cp
import numpy as np
from cvxpy_solver import calculate_expected_loss


# ============================================================
# Problem (3.21): Smoothed KL MSDA (NO R VARIABLE)
# ============================================================
def solve_convex_problem_smoothed_kl_321(
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
    Solves Problem (3.21) in your implementation style:

        max_{w,Q}  sum_{i,j} D_{ij} * log(Q_{ij} / w_j)
                   + (eta / (kN)) * sum_{i,j} log(Q_{ij})

        s.t.  w in simplex (and >= w_min)
              Q rows in simplex (and >= q_min)
              risk constraints:
                  for each t:
                      w^T L[:,t] <= epsilon_t
              where epsilon_t = epsilon if scalar, else epsilon[t].

    Notes:
    - Expects D to be (N,k) and columns normalized to sum to 1 (if normalize_D=True we enforce it).
    - Expects Y to be either (N,C) one-hot (recommended) or (N,) labels only if your H matches.
    - Expects H to be either (N,C,k) probabilities per source, or (N,k) in a simplified case.
    """

    # --------------------------------------------------------
    # Shape handling (SAFE for your calculate_expected_loss)
    # --------------------------------------------------------
    # If D is (N,C,k) but replicated across C, reduce to (N,k)
    if D.ndim == 3:
        # assume D = tile(D_loaded[:,None,:], (1,C,1))
        D = D[:, 0, :]  # (N,k)

    if D.ndim != 2:
        raise ValueError(f"D must be 2D (N,k) after reduction. Got shape={D.shape}")

    N, k = D.shape

    # --------------------------------------------------------
    # Normalize D columns to probabilities (critical for L_mat)
    # --------------------------------------------------------
    if normalize_D:
        col_sums = D.sum(axis=0, keepdims=True)
        col_sums[col_sums == 0] = 1.0
        D = D / col_sums  # now sum_i D[i,t] = 1

    # --------------------------------------------------------
    # Expected losses L_mat[j,t]
    # --------------------------------------------------------
    L_mat = calculate_expected_loss(Y, H, D, k)

    # --------------------------------------------------------
    # Variables
    # --------------------------------------------------------
    w = cp.Variable(k, nonneg=True, name="w")
    Q = cp.Variable((N, k), nonneg=True, name="Q")

    # --------------------------------------------------------
    # Objective: sum_{i,j} D_{ij} * log(Q_{ij}/w_j)  + smoothing
    # Implemented via -rel_entr(w_j, Q_ij) = w_j*log(Q_ij/w_j)
    # but we weight each column by D[:,j] (like your 3.10 code).
    # --------------------------------------------------------
    main_terms = [
        cp.sum(cp.multiply(D[:, j], -cp.rel_entr(w[j], Q[:, j])))
        for j in range(k)
    ]
    main_obj = cp.sum(main_terms)

    # Smooth term (barrier-like): encourages Q away from zeros
    smooth_obj = (eta / (k * N)) * cp.sum(cp.log(Q))

    objective = cp.Maximize(main_obj + smooth_obj)

    # --------------------------------------------------------
    # Constraints
    # --------------------------------------------------------
    constraints = [
        cp.sum(w) == 1,
        w >= w_min,
        cp.sum(Q, axis=1) == 1,
        Q >= q_min,
    ]

    # --- epsilon: scalar vs vector case ---
    if isinstance(epsilon, (list, tuple, np.ndarray)):
        eps_vec = np.asarray(epsilon).reshape(-1)
        if eps_vec.shape[0] != k:
            raise ValueError(f"epsilon vector must have length k={k}. Got {eps_vec.shape[0]}")
        for t in range(k):
            constraints.append(w @ L_mat[:, t] <= eps_vec[t])
    else:
        eps_val = float(epsilon)
        for t in range(k):
            constraints.append(w @ L_mat[:, t] <= eps_val)

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
        print(f"[3.21] Solver exception: {e}")
        return None, None

    if prob.status not in ["optimal", "optimal_inaccurate"]:
        print(f"[3.21] Optimization failed | status={prob.status}")
        return None, None

    return np.asarray(w.value).reshape(-1), np.asarray(Q.value)
