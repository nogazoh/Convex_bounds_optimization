import cvxpy as cp
import numpy as np
from cvxpy_solver import calculate_expected_loss


# ============================================================
# Helpers
# ============================================================
def _flatten_if_needed(Y, D, H):
    """
    Supports either:
      - D: (N, k), H: (N, k), Y: (N,)
    or
      - D: (N0, C, k), H: (N0, C, k), Y: (N0, C)
    Returns flattened (Y, D, H) and (N, k).
    """
    if D.ndim == 3:
        _, _, k = D.shape
        D = D.reshape(-1, k)
        H = H.reshape(-1, k)
        Y = Y.reshape(-1)
    N, k = D.shape
    return Y, D, H, N, k


def compute_p_tilde(D, eps=1e-15):
    """
    Domain-anchored normalized expert density:
        p_tilde[i,j] = p_{i,j} / sum_s p_{i,s}

    If rows already sum to 1, p_tilde == D.
    eps prevents division by 0.
    """
    row_sum = np.sum(D, axis=1, keepdims=True)
    return D / np.maximum(row_sum, eps)


# ============================================================
# Problem (3.32): Domain-Anchored Smoothed + NEW epsilon constraint via q̄
# ============================================================
def solve_convex_problem_domain_anchored_smoothed_332(
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
    normalize_domains=True,
    ptilde_eps=1e-15,
):
    """
    Problem 3.32 = your Problem 3.22 but with the NEW epsilon constraint:

    Objective (same as 3.22):
        max_{w,Q}  sum_{i,j} w_j p_{i,j} log(q_{j|i}/w_j)
                   + (eta/k) * sum_{i,j} p_tilde[i,j] * log(q_{j|i})

    Constraints (same simplex constraints):
        sum_j w_j = 1, w_j >= w_min
        sum_j q_{j|i} = 1, q_{j|i} >= q_min

    NEW risk constraints (replace w^T L[:,t] <= epsilon):
        For each t:
            sum_{i=1}^N sum_{j=1}^k q_{j|i} * p_{i,j} * (L_mat[j,t] - eps_t) <= 0

    In matrix form:
        Let S_j := sum_i p_{i,j} q_{j|i}  (i.e., sum_i D[i,j] * Q[i,j])
        Then:  S^T (L_mat[:,t] - eps_t) <= 0  for all t.
    """

    # -------------------------
    # Shapes
    # -------------------------
    Y, D, H, N, k = _flatten_if_needed(Y, D, H)

    # Optional: normalize D rows (turn scores into per-sample distributions)
    if normalize_domains:
        D = compute_p_tilde(D, eps=ptilde_eps)

    # -------------------------
    # Loss matrix L_mat[j,t]
    # -------------------------
    L_mat = calculate_expected_loss(Y, H, D, k)

    # -------------------------
    # p_tilde for anchoring
    # -------------------------
    p_tilde = compute_p_tilde(D, eps=ptilde_eps)

    # -------------------------
    # Variables
    # -------------------------
    w = cp.Variable(k, nonneg=True, name="w")
    Q = cp.Variable((N, k), nonneg=True, name="Q")

    # -------------------------
    # Objective (unchanged)
    # -------------------------
    main_terms = []
    for j in range(k):
        main_terms.append(
            cp.sum(cp.multiply(D[:, j], -cp.rel_entr(w[j], Q[:, j])))
        )
    main_obj = cp.sum(main_terms)

    anchored_smooth_obj = (eta / k) * cp.sum(cp.multiply(p_tilde, cp.log(Q)))

    objective = cp.Maximize(main_obj + anchored_smooth_obj)

    # -------------------------
    # Constraints: simplex + lower bounds (unchanged)
    # -------------------------
    constraints = [
        cp.sum(w) == 1,
        w >= w_min,
        cp.sum(Q, axis=1) == 1,
        Q >= q_min,
    ]

    # -------------------------
    # NEW epsilon constraints
    #   S_j := sum_i D_{ij} Q_{ij}
    #   For each t: S^T (L_mat[:,t] - eps_t) <= 0
    # -------------------------
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

    # -------------------------
    # Solve
    # -------------------------
    prob = cp.Problem(objective, constraints)

    eps_print = epsilon if isinstance(epsilon, (int, float, np.floating)) else "vector"
    print(
        f"[Problem (3.32)] Solving domain-anchored smoothed optimization | "
        f"N={N}, k={k}, epsilon={eps_print}, eta={eta:.2e}, solver={solver_type}"
    )

    try:
        if solver_type.upper() == "SCS":
            prob.solve(
                solver=cp.SCS,
                verbose=False,
                eps=scs_eps,
                max_iters=scs_max_iters,
            )
        elif solver_type.upper() == "MOSEK":
            prob.solve(solver=cp.MOSEK, verbose=False)
        elif solver_type.upper() == "CLARABEL":
            prob.solve(solver=cp.CLARABEL, verbose=False)
        else:
            print("[Problem (3.32)] Unknown solver, falling back to SCS")
            prob.solve(
                solver=cp.SCS,
                verbose=False,
                eps=scs_eps,
                max_iters=scs_max_iters,
            )
    except Exception as e:
        print(f"[Problem (3.32)] Solver exception: {e}")
        return None, None

    if prob.status not in ["optimal", "optimal_inaccurate"]:
        print(f"[Problem (3.32)] Optimization failed | status={prob.status}")
        return None, None

    # -------------------------
    # Results + diagnostics
    # -------------------------
    w_val = np.asarray(w.value).reshape(-1)
    Q_val = np.asarray(Q.value)

    print(
        f"[Problem (3.32)] Solved successfully | status={prob.status}\n"
        f"  • w (mixture over sources): {np.round(w_val, 6)}\n"
        f"    sum={w_val.sum():.6f}, min={w_val.min():.2e}\n"
        f"  • q_{{j|i}} diagnostics:\n"
        f"    row-sum min/max = "
        f"({Q_val.sum(axis=1).min():.6f}, {Q_val.sum(axis=1).max():.6f}), "
        f"min entry = {Q_val.min():.2e}"
    )

    # Diagnostics for the NEW constraint:
    # For each t, compute S^T (L[:,t] - eps_t) which should be <= 0.
    S_val = np.sum(D * Q_val, axis=0)  # (k,)

    if isinstance(epsilon, (list, tuple, np.ndarray)):
        eps_vec = np.asarray(epsilon).reshape(-1)
        for t in range(k):
            lhs = float(S_val @ (L_mat[:, t] - eps_vec[t]))
            print(f"  • [Problem (3.32)] constraint(t={t}): S·(L[:,t]-eps_t) = {lhs:.4e} (<= 0)")
    else:
        eps_val = float(epsilon)
        for t in range(k):
            lhs = float(S_val @ (L_mat[:, t] - eps_val))
            print(f"  • [Problem (3.32)] constraint(t={t}): S·(L[:,t]-eps) = {lhs:.4e} (<= 0)")

    return w_val, Q_val


# ============================================================
# Optional: Closed-form q_{j|i} for verification (paper check)
# ============================================================
def closed_form_q_domain_anchored(w, p_row, eta, eps=1e-15):
    """
    Closed-form from your draft for the domain-anchored smoothing:

        q^*_{j|i} = ((w_j + gamma_i) p_{i,j}) / sum_s ((w_s + gamma_i) p_{i,s})
        where gamma_i = eta / (k * sum_s p_{i,s})

    If p_row sums to 1, then gamma_i = eta/k.
    """
    w = np.asarray(w).reshape(-1)
    p_row = np.asarray(p_row).reshape(-1)
    k = len(w)

    denom_p = np.sum(p_row)
    gamma_i = eta / (k * max(denom_p, eps))

    num = (w + gamma_i) * p_row
    den = np.sum(num)
    if den <= eps:
        return np.ones(k) / k
    return num / den
