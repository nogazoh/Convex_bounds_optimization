import cvxpy as cp
import numpy as np
from loss_functions import calculate_weighted_constraint_matrix


# ============================================================
# Problem (3.33): Smoothed formulation with ORIGINAL p_{i|j}
# (3.23 but with NEW epsilon constraint via q̄)
# OPTIMIZED VERSION: Separates Loss from Optimization
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
        loss_type="01",   # "01" or "ce"
):

    # -------------------------
    # 1) Shape Handling & Loss Calculation
    # -------------------------
    if D.ndim == 3:
        N_orig, C, k = D.shape

        # Loss matrix from weighted loss functions
        D_for_loss = D[:, 0, :]   # reduce replicated D across classes
        L_mat = calculate_weighted_constraint_matrix(
            Y=Y,
            H=H,
            D=D_for_loss,
            num_sources=k,
            loss_type=loss_type,
            normalize_D_cols=True,
        )

        # Optimization slice (assumes D is identical across classes)
        D_opt = D[:, 0, :]  # shape (N_orig, k)

    else:
        N_orig, k = D.shape
        D_opt = D
        L_mat = calculate_weighted_constraint_matrix(
            Y=Y,
            H=H,
            D=D_opt,
            num_sources=k,
            loss_type=loss_type,
            normalize_D_cols=True,
        )

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

    # --------------------------------------------------------
    # NEW epsilon constraints
    # --------------------------------------------------------
    if isinstance(epsilon, (list, tuple, np.ndarray)):
        eps_vec = np.asarray(epsilon).reshape(-1)
        if eps_vec.shape[0] != k:
            raise ValueError(f"epsilon vector must have length k={k}. Got {eps_vec.shape[0]}")
    else:
        eps_vec = np.full(k, float(epsilon))

    for t in range(k):
        S_t = cp.sum(cp.multiply(D_opt[:, t:t + 1], Q), axis=0)  # shape (k,)
        constraints.append(S_t @ (L_mat[:, t] - eps_vec[t]) <= 0)

    # -------------------------
    # Solve
    # -------------------------
    prob = cp.Problem(objective, constraints)

    print(
        f"[Problem (3.33)] Solving ORIGINAL-p smoothed (NEW eps constraint) | "
        f"N={N} (was {N_orig * C if D.ndim == 3 else N}), k={k}, "
        f"eps={'vector' if isinstance(epsilon, (list, tuple, np.ndarray)) else f'{float(epsilon):.2e}'}, "
        f"eta={eta:.2e}, solver={solver_type}, loss_type={loss_type}"
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

    # Optional diagnostic for NEW constraints
    if isinstance(epsilon, (list, tuple, np.ndarray)):
        eps_vec = np.asarray(epsilon).reshape(-1)
        for t in range(k):
            S_t_val = np.sum(Q_val * D_opt[:, t:t + 1], axis=0)
            lhs = float(S_t_val @ (L_mat[:, t] - eps_vec[t]))
            print(f"  • constraint(t={t}): S_t·(L[:,t]-eps_t) = {lhs:.4e} (<= 0)")
    else:
        eps_val = float(epsilon)
        for t in range(k):
            S_t_val = np.sum(Q_val * D_opt[:, t:t + 1], axis=0)
            lhs = float(S_t_val @ (L_mat[:, t] - eps_val))
            print(f"  • constraint(t={t}): S_t·(L[:,t]-eps) = {lhs:.4e} (<= 0)")

    return w_val, Q_val