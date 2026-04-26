import cvxpy as cp
import numpy as np
from loss_functions import calculate_weighted_constraint_matrix


def solve_convex_problem_soft_kl_diagonal_fast_6A(
    Y,
    D,
    H,
    eta=1e-2,
    beta=1e-2,
    epsilon=1e-2,
    solver_type='SCS',
    loss_type='01',
    scs_eps=1e-3,
    scs_max_iters=5000,
):
    """
    Solver 6A: epigraph / max-alignment version

        max_{w,Q,R,s}
            sum_{i,j} p_{i|j} log(q_{j|i} / w_j)
            - eta * s
            + beta * sum_t r_{t|t} (epsilon_t - L_t^t)

        s.t.
            kappa_t(Q,R) <= s                     for all t
            sum_j r_{j|t} L_t^j <= epsilon_t      for all t
            sum_j q_{j|i} = 1                     for all i
            sum_j r_{j|t} = 1                     for all t
            sum_j w_j = 1
            w_j, q_{j|i}, r_{j|t}, s >= 0

    where
        kappa_t(Q,R) = sum_i p_{i|t} KL(r_{.|t} || q_{.|i})

    Returns:
        (w, Q, R, s_value) if successful
        (None, None, None, None) otherwise
    """

    # --------------------------------------------------
    # 1. Shape handling
    # --------------------------------------------------
    if D.ndim == 3:
        N_orig, _, k = D.shape
        D_opt = D[:, 0, :]
    else:
        N_orig, k = D.shape
        D_opt = D

    # --------------------------------------------------
    # 2. Normalize D by columns so each column behaves
    #    like p_{i|t}
    # --------------------------------------------------
    D_opt = np.asarray(D_opt, dtype=float)
    col_sums = D_opt.sum(axis=0, keepdims=True)
    col_sums[col_sums == 0] = 1.0
    D_opt = D_opt / col_sums

    N, k = D_opt.shape

    # --------------------------------------------------
    # 3. Build loss matrix: L_mat[j, t] = L_t^j
    # --------------------------------------------------
    L_mat = calculate_weighted_constraint_matrix(
        Y=Y,
        H=H,
        D=D_opt,
        num_sources=k,
        loss_type=loss_type,
        normalize_D_cols=False,   # already normalized above
    )
    L_mat = np.asarray(L_mat, dtype=float)

    # --------------------------------------------------
    # 4. Convert epsilon to vector
    # --------------------------------------------------
    if np.isscalar(epsilon):
        epsilon_vec = np.full(k, float(epsilon))
    else:
        epsilon_vec = np.asarray(epsilon, dtype=float)
        if epsilon_vec.shape[0] != k:
            raise ValueError(f"epsilon must have length {k}, got {epsilon_vec.shape}")

    print(
        f"   >>> [CVXPY-SoftKL-Diag-6A] Setup: "
        f"N={N} (Original N={N_orig}), k={k}, "
        f"loss_type={loss_type}, eta={eta}, beta={beta}"
    )

    # --------------------------------------------------
    # 5. CVXPY constants
    # --------------------------------------------------
    D_const = cp.Constant(D_opt)
    L_const = cp.Constant(L_mat)
    eps_const = cp.Constant(epsilon_vec)
    ones_N1 = cp.Constant(np.ones((N, 1)))

    # --------------------------------------------------
    # 6. Variables
    # --------------------------------------------------
    w = cp.Variable(k, nonneg=True, name="w")          # (k,)
    Q = cp.Variable((N, k), nonneg=True, name="Q")     # Q[i,j] = q_{j|i}
    R = cp.Variable((k, k), nonneg=True, name="R")     # R[j,t] = r_{j|t}
    s = cp.Variable(nonneg=True, name="s")             # scalar epigraph variable

    # --------------------------------------------------
    # 7. Main objective:
    #    sum_{i,j} p_{i|j} log(q_{j|i} / w_j)
    # --------------------------------------------------
    w_row = cp.reshape(w, (1, k), order="F")
    W_tiled = ones_N1 @ w_row                          # (N, k)

    main_obj = cp.sum(
        cp.multiply(D_const, -cp.rel_entr(W_tiled, Q))
    )

    # --------------------------------------------------
    # 8. Build kappa_t(Q,R)
    #    kappa_t = sum_i p_{i|t} KL(r_{.|t} || q_{.|i})
    # --------------------------------------------------
    kappa_terms = []

    for t in range(k):
        p_t = D_opt[:, t:t + 1]                            # (N, 1)
        p_t_const = cp.Constant(p_t)

        r_t_row = cp.reshape(R[:, t], (1, k), order="F")  # (1, k)
        R_tiled = ones_N1 @ r_t_row                       # (N, k)

        kl_matrix_t = cp.rel_entr(R_tiled, Q)            # elementwise r log(r/q)
        weighted_kl_t = cp.multiply(p_t_const, kl_matrix_t)
        kappa_t = cp.sum(weighted_kl_t)

        kappa_terms.append(kappa_t)

    # --------------------------------------------------
    # 9. Diagonal reward:
    #    beta * sum_t r_{t|t} (epsilon_t - L_t^t)
    # --------------------------------------------------
    diag_R = cp.diag(R)
    diag_L = np.diag(L_mat)
    diag_reward = cp.sum(
        cp.multiply(diag_R, epsilon_vec - diag_L)
    )

    # --------------------------------------------------
    # 10. Objective
    # --------------------------------------------------
    objective = cp.Maximize(
        main_obj
        - eta * s
        + beta * diag_reward
    )

    # --------------------------------------------------
    # 11. Constraints
    # --------------------------------------------------
    constraints = [
        cp.sum(w) == 1,
        cp.sum(Q, axis=1) == 1,      # sum_j q_{j|i} = 1
        cp.sum(R, axis=0) == 1,      # sum_j r_{j|t} = 1
    ]

    # Risk constraints:
    # sum_j r_{j|t} L_t^j <= epsilon_t
    risk_vec = cp.sum(cp.multiply(R, L_const), axis=0)
    constraints.append(risk_vec <= eps_const)

    # Epigraph constraints:
    # kappa_t(Q,R) <= s
    for kappa_t in kappa_terms:
        constraints.append(kappa_t <= s)

    # --------------------------------------------------
    # 12. Solve
    # --------------------------------------------------
    prob = cp.Problem(objective, constraints)
    print(f"   >>> [CVXPY-SoftKL-Diag-6A] Starting {solver_type} solve...")

    try:
        if solver_type == "SCS":
            prob.solve(
                solver=cp.SCS,
                verbose=False,
                eps=scs_eps,
                max_iters=scs_max_iters,
            )
        elif solver_type == "CLARABEL":
            prob.solve(
                solver=cp.CLARABEL,
                verbose=False,
            )
        else:
            print(f"   !!! Unknown solver {solver_type}, defaulting to SCS")
            prob.solve(
                solver=cp.SCS,
                verbose=False,
                eps=scs_eps,
                max_iters=scs_max_iters,
            )

    except Exception as e:
        print(f"   !!! [CVXPY-SoftKL-Diag-6A] Solver Exception: {e}")
        return None, None, None, None

    if prob.status not in ["optimal", "optimal_inaccurate"]:
        print(f"   !!! [CVXPY-SoftKL-Diag-6A] Failed/Infeasible. Status: {prob.status}")
        return None, None, None, None

    kappa_vals = np.array([kt.value for kt in kappa_terms], dtype=float)
    max_kappa = float(np.max(kappa_vals)) if len(kappa_vals) > 0 else np.nan

    print(f"   >>> [CVXPY-SoftKL-Diag-6A] Solved! Weights: {np.round(w.value, 3)}")
    print(f"   >>> [CVXPY-SoftKL-Diag-6A] Objective value: {prob.value:.6f}")
    print(f"   >>> [CVXPY-SoftKL-Diag-6A] s*: {float(s.value):.6f}")
    print(f"   >>> [CVXPY-SoftKL-Diag-6A] kappa*: {np.round(kappa_vals, 6)}")
    print(f"   >>> [CVXPY-SoftKL-Diag-6A] max_kappa*: {max_kappa:.6f}")

    return w.value, Q.value, R.value, float(s.value)