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
):
    """
    Problem 3.33 = your Problem 3.23, but replaces the risk constraints with:

        For each t:
            sum_{i=1}^N sum_{j=1}^k Q_{ij} * D_opt[i,j] * (L_mat[j,t] - eps_t) <= 0

    Equivalent:
        S_j := sum_i D_opt[i,j] * Q[i,j]
        then S^T (L_mat[:,t] - eps_t) <= 0  for all t.

    Keeps your "optimized" reduction when D is (N0,C,k) tiled across classes.
    """

    # -------------------------
    # 1) Shape Handling & Loss Calculation (unchanged)
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

    N, k = D_opt.shape

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
            cp.sum(cp.multiply(D_opt[:, j], -cp.rel_entr(w[j], Q[:, j])))
        )
    main_obj = cp.sum(main_terms)

    smooth_obj = eta * cp.sum(cp.multiply(D_opt, cp.log(Q)))

    objective = cp.Maximize(main_obj + smooth_obj)

    # -------------------------
    # Constraints (simplex + lower bounds unchanged)
    # -------------------------
    constraints = [
        cp.sum(w) == 1,
        w >= w_min,
        cp.sum(Q, axis=1) == 1,
        Q >= q_min,
    ]

    # -------------------------
    # NEW epsilon constraints (replace w @ L_mat[:,t] <= epsilon)
    # -------------------------
    S = cp.sum(cp.multiply(D_opt, Q), axis=0)  # shape (k,)

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

    print(
        f"[Problem (3.33)] Solving ORIGINAL-p smoothed (NEW eps constraint) | "
        f"N={N} (was {N_orig * C if D.ndim == 3 else N}), k={k}, "
        f"eps={'vector' if isinstance(epsilon, (list, tuple, np.ndarray)) else f'{float(epsilon):.2e}'}, "
        f"eta={eta:.2e}, solver={solver_type}"
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
            print("[Problem (3.33)] Unknown solver, falling back to SCS")
            prob.solve(solver=cp.SCS, verbose=False, eps=scs_eps)
    except Exception as e:
        print(f"[Problem (3.33)] Solver exception: {e}")
        return None, None

    if prob.status not in ["optimal", "optimal_inaccurate"]:
        print(f"[Problem (3.33)] Optimization failed | status={prob.status}")
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

    # Optional diagnostic for NEW constraints:
    S_val = np.sum(D_opt * Q_val, axis=0)  # (k,)
    if isinstance(epsilon, (list, tuple, np.ndarray)):
        eps_vec = np.asarray(epsilon).reshape(-1)
        for t in range(k):
            lhs = float(S_val @ (L_mat[:, t] - eps_vec[t]))
            print(f"  • constraint(t={t}): S·(L[:,t]-eps_t) = {lhs:.4e} (<= 0)")
    else:
        eps_val = float(epsilon)
        for t in range(k):
            lhs = float(S_val @ (L_mat[:, t] - eps_val))
            print(f"  • constraint(t={t}): S·(L[:,t]-eps) = {lhs:.4e} (<= 0)")

    return w_val, Q_val
