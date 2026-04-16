import cvxpy as cp
import numpy as np
from cvxpy_solver import calculate_expected_loss


# ============================================================
# Problem (3.33): Smoothed formulation with ORIGINAL p_{i|j}
# (3.23 but with NEW epsilon constraint via q̄)
# OPTIMIZED VERSION: Separates Loss (Full dims) from Opt (Reduced dims)
# ============================================================
def solve_convex_problem_smoothed_original_p_333(
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
        return_kkt_details=False,
        kkt_tol=1e-8,
):
    """
    Solves problem (3.33) and optionally returns KKT dual details.

    KKT details returned:
      - mu_t    : duals of the domain-risk constraints
      - gamma_i : duals of row-simplex constraints sum_j Q[i,j] = 1
    """

    # -------------------------
    # 1) Shape Handling & Loss Calculation
    # -------------------------
    if D.ndim == 3:
        N_orig, C, k = D.shape

        # Flatten ONLY for loss calculation
        D_flat = D.reshape(-1, k)
        H_flat = H.reshape(-1, k)
        Y_flat = Y.reshape(-1)

        L_mat = calculate_expected_loss(Y_flat, H_flat, D_flat, k)

        # Optimization slice (assumes D is identical across classes)
        D_opt = D[:, 0, :]  # shape (N_orig, k)

    else:
        N_orig, k = D.shape
        D_opt = D
        L_mat = calculate_expected_loss(Y, H, D, k)
        C = 1  # only used for pretty print below

    N, k = D_opt.shape

    # -------------------------
    # Variables
    # -------------------------
    w = cp.Variable(k, nonneg=True, name="w")
    Q = cp.Variable((N, k), nonneg=True, name="Q")

    # -------------------------
    # Objective
    # -------------------------
    main_terms = []
    for j in range(k):
        main_terms.append(
            cp.sum(cp.multiply(D_opt[:, j], -cp.rel_entr(w[j], Q[:, j])))
        )
    main_obj = cp.sum(main_terms)

    smooth_obj = eta * cp.sum(cp.multiply(D_opt, cp.log(Q)))
    objective = cp.Maximize(main_obj + smooth_obj)

    # -------------------------
    # Constraints
    # -------------------------
    constraints = []

    # w simplex
    w_sum_constraint = (cp.sum(w) == 1)
    constraints.append(w_sum_constraint)
    constraints.append(w >= w_min)

    # Q row simplex -> gamma_i
    row_simplex_constraints = []
    for i in range(N):
        c = (cp.sum(Q[i, :]) == 1)
        row_simplex_constraints.append(c)
        constraints.append(c)

    constraints.append(Q >= q_min)

    # -------------------------
    # NEW epsilon constraints
    # -------------------------
    if isinstance(epsilon, (list, tuple, np.ndarray)):
        eps_vec = np.asarray(epsilon).reshape(-1)
        if eps_vec.shape[0] != k:
            raise ValueError(f"epsilon vector must have length k={k}. Got {eps_vec.shape[0]}")
    else:
        eps_vec = np.full(k, float(epsilon))

    # S_all[t, j] = sum_i D_opt[i, t] * Q[i, j]
    S_all = D_opt.T @ Q  # shape (k, k)

    # Risk constraints -> mu_t
    risk_constraints = []
    for t in range(k):
        c = (S_all[t, :] @ (L_mat[:, t] - eps_vec[t]) <= 0)
        risk_constraints.append(c)
        constraints.append(c)

    # -------------------------
    # Solve
    # -------------------------
    prob = cp.Problem(objective, constraints)

    print(
        f"[Problem (3.33)] Solving ORIGINAL-p smoothed (NEW eps constraint) | "
        f"N={N} (was {N_orig * C if D.ndim == 3 else N}), k={k}, "
        f"eps={'vector' if isinstance(epsilon, (list, tuple, np.ndarray)) else f'{float(np.asarray(eps_vec)[0]):.2e}'}, "
        f"eta={eta:.2e}, solver={solver_type}"
    )

    try:
        st = solver_type.upper()
        if st == "SCS":
            prob.solve(
                solver=cp.SCS,
                verbose=False,
                eps=scs_eps,
                max_iters=scs_max_iters,
            )
        elif st == "MOSEK":
            prob.solve(solver=cp.MOSEK, verbose=False)
        elif st == "CLARABEL":
            prob.solve(solver=cp.CLARABEL, verbose=False)
        else:
            print("[Problem (3.33)] Unknown solver, falling back to SCS")
            prob.solve(solver=cp.SCS, verbose=False, eps=scs_eps, max_iters=scs_max_iters)
    except Exception as e:
        print(f"[Problem (3.33)] Solver exception: {e}")
        if return_kkt_details:
            return None, None, None
        return None, None

    if prob.status not in ["optimal", "optimal_inaccurate"]:
        print(f"[Problem (3.33)] Optimization failed | status={prob.status}")
        if return_kkt_details:
            return None, None, None
        return None, None

    # -------------------------
    # Results + diagnostics
    # -------------------------
    w_val = np.asarray(w.value).reshape(-1)
    Q_val = np.asarray(Q.value)

    print(
        f"[Problem (3.33)] Solved successfully | status={prob.status}\n"
        f"  • w: {np.round(w_val, 6)}"
    )

    # Diagnostic for NEW constraints
    S_all_val = D_opt.T @ Q_val  # shape (k, k)
    for t in range(k):
        lhs = float(S_all_val[t, :] @ (L_mat[:, t] - eps_vec[t]))
        print(f"  • constraint(t={t}): S_t·(L[:,t]-eps_t) = {lhs:.4e} (<= 0)")

    # -------------------------
    # Optional KKT details
    # -------------------------
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