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
    return_kkt_details=False,
    kkt_tol=1e-8,
):
    """
    Solves modified problem (3.31) and optionally returns KKT dual details.

    KKT details returned:
      - mu_t    : duals of the domain-risk constraints
      - gamma_i : duals of row-simplex constraints sum_j Q[i,j] = 1
    """

    # --------------------------------------------------------
    # Shape handling
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
        D = D / col_sums

    # --------------------------------------------------------
    # Expected losses L_mat[j,t]
    # --------------------------------------------------------
    L_mat = calculate_expected_loss(Y, H, D, k)  # shape (k,k)

    # --------------------------------------------------------
    # Variables
    # --------------------------------------------------------
    w = cp.Variable(k, nonneg=True, name="w")
    Q = cp.Variable((N, k), nonneg=True, name="Q")

    # --------------------------------------------------------
    # Objective
    # --------------------------------------------------------
    main_terms = [
        cp.sum(cp.multiply(D[:, j], -cp.rel_entr(w[j], Q[:, j])))
        for j in range(k)
    ]
    main_obj = cp.sum(main_terms)
    smooth_obj = (eta / (k * N)) * cp.sum(cp.log(Q))
    objective = cp.Maximize(main_obj + smooth_obj)

    # --------------------------------------------------------
    # Constraints
    # --------------------------------------------------------
    constraints = []

    # w-simplex constraints
    w_sum_constraint = (cp.sum(w) == 1)
    constraints.append(w_sum_constraint)
    constraints.append(w >= w_min)

    # Q row simplex constraints: these correspond to gamma_i
    row_simplex_constraints = []
    for i in range(N):
        c = (cp.sum(Q[i, :]) == 1)
        row_simplex_constraints.append(c)
        constraints.append(c)

    constraints.append(Q >= q_min)

    # --------------------------------------------------------
    # Epsilon handling
    # --------------------------------------------------------
    if isinstance(epsilon, (list, tuple, np.ndarray)):
        eps_vec = np.asarray(epsilon).reshape(-1)
        if eps_vec.shape[0] != k:
            raise ValueError(f"epsilon vector must have length k={k}. Got {eps_vec.shape[0]}")
    else:
        eps_vec = np.full(k, float(epsilon))

    # --------------------------------------------------------
    # Risk constraints: these correspond to mu_t
    # S_all[t, j] = sum_i D[i, t] * Q[i, j]
    # --------------------------------------------------------
    S_all = D.T @ Q  # shape (k, k)

    risk_constraints = []
    for t in range(k):
        c = (S_all[t, :] @ (L_mat[:, t] - eps_vec[t]) <= 0)
        risk_constraints.append(c)
        constraints.append(c)

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
        if return_kkt_details:
            return None, None, None
        return None, None

    if prob.status not in ["optimal", "optimal_inaccurate"]:
        print(f"[3.31] Optimization failed | status={prob.status}")
        if return_kkt_details:
            return None, None, None
        return None, None

    w_val = np.asarray(w.value).reshape(-1)
    Q_val = np.asarray(Q.value)

    # --------------------------------------------------------
    # Optional KKT details
    # --------------------------------------------------------
    if return_kkt_details:
        gamma_vals = np.array(
            [float(np.squeeze(c.dual_value)) for c in row_simplex_constraints],
            dtype=float
        )
        mu_vals = np.array(
            [float(np.squeeze(c.dual_value)) for c in risk_constraints],
            dtype=float
        )

        kkt_details = {
            "mu": mu_vals.tolist(),
            "gamma": gamma_vals.tolist(),
            "mu_is_zero": (np.abs(mu_vals) <= kkt_tol).tolist(),
            "gamma_is_zero": (np.abs(gamma_vals) <= kkt_tol).tolist(),
            "mu_zero_count": int(np.sum(np.abs(mu_vals) <= kkt_tol)),
            "gamma_zero_count": int(np.sum(np.abs(gamma_vals) <= kkt_tol)),
            "mu_min": float(np.min(mu_vals)) if len(mu_vals) else None,
            "mu_max": float(np.max(mu_vals)) if len(mu_vals) else None,
            "gamma_min": float(np.min(gamma_vals)) if len(gamma_vals) else None,
            "gamma_max": float(np.max(gamma_vals)) if len(gamma_vals) else None,
            "kkt_tol": float(kkt_tol),
            "status": prob.status,
        }

        return w_val, Q_val, kkt_details

    return w_val, Q_val