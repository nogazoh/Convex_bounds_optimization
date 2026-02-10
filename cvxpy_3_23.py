import cvxpy as cp
import numpy as np
from cvxpy_solver import calculate_expected_loss


# ============================================================
# Problem (3.23): Smoothed formulation with ORIGINAL p_{i|j}
# OPTIMIZED VERSION: Separates Loss (Full dims) from Opt (Reduced dims)
# ============================================================
def solve_convex_problem_smoothed_original_p(
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
    Solves Problem (3.23) optimized for speed.
    """

    # -------------------------
    # 1. Shape Handling & Loss Calculation
    # --------------------------------------------------
    if D.ndim == 3:
        N_orig, C, k = D.shape

        # --- Flatten ONLY for Loss Calculation ---
        # We need classes for the loss, but NOT for the optimization variables
        D_flat = D.reshape(-1, k)
        H_flat = H.reshape(-1, k)
        Y_flat = Y.reshape(-1)

        L_mat = calculate_expected_loss(Y_flat, H_flat, D_flat, k)

        # --- OPTIMIZATION PREP (The Fix) ---
        # Take one representative slice per image (since D is tiled/identical across classes)
        # This reduces variables from (N*31) to N
        D_opt = D[:, 0, :]

    else:
        # Already 2D
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
    # Objective
    # -------------------------
    # Main term: sum_{i,j} w_j p_{i,j} log(q_{j|i}/w_j)
    main_terms = []
    for j in range(k):
        # Using D_opt (Reduced size N)
        main_terms.append(
            cp.sum(cp.multiply(D_opt[:, j], -cp.rel_entr(w[j], Q[:, j])))
        )
    main_obj = cp.sum(main_terms)

    # Smoothing with ORIGINAL p_{i,j}: eta * sum_{i,j} p_{i,j} log q_{j|i}
    # Using D_opt (Reduced size N)
    smooth_obj = eta * cp.sum(cp.multiply(D_opt, cp.log(Q)))

    objective = cp.Maximize(main_obj + smooth_obj)

    # -------------------------
    # Constraints
    # -------------------------
    constraints = [
        cp.sum(w) == 1,
        w >= w_min,
        cp.sum(Q, axis=1) == 1,
        Q >= q_min,
    ]

    # Risk constraints (one per domain t)
    # L_mat is already calculated correctly
    for t in range(k):
        constraints.append(w @ L_mat[:, t] <= epsilon)

    # -------------------------
    # Solve
    # -------------------------
    prob = cp.Problem(objective, constraints)

    print(
        f"[Problem (3.23)] Solving ORIGINAL-p smoothed | "
        f"N={N} (was {N_orig * C if D.ndim == 3 else N}), k={k}, "
        f"eps={epsilon:.2e}, eta={eta:.2e}"
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
            print("[Problem (3.23)] Unknown solver, falling back to SCS")
            prob.solve(solver=cp.SCS, verbose=False, eps=scs_eps)
    except Exception as e:
        print(f"[Problem (3.23)] Solver exception: {e}")
        return None, None

    if prob.status not in ["optimal", "optimal_inaccurate"]:
        print(f"[Problem (3.23)] Optimization failed | status={prob.status}")
        return None, None

    # -------------------------
    # Results + diagnostics
    # -------------------------
    w_val = np.asarray(w.value).reshape(-1)
    Q_val = np.asarray(Q.value)

    print(
        f"[Problem (3.23)] Solved successfully | status={prob.status}\n"
        f"  • w: {np.round(w_val, 6)}"
    )

    return w_val, Q_val