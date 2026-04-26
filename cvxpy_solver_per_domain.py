import cvxpy as cp
import numpy as np
from loss_functions import calculate_weighted_constraint_matrix


def solve_convex_problem_per_domain(
    Y,
    D,
    H,
    delta=1e-2,
    epsilon=1e-2,
    solver_type='SCS',
    loss_type='01',
):
    """
    Solves the per-domain version:
    - KL constraint per domain t
    - risk constraint per domain t

    Returns:
        (w, Q) if successful
        (None, None) otherwise
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
    # 2. Normalize D once
    # --------------------------------------------------
    D_opt = np.asarray(D_opt, dtype=float)
    col_sums = D_opt.sum(axis=0, keepdims=True)
    col_sums[col_sums == 0] = 1.0
    D_opt = D_opt / col_sums

    # --------------------------------------------------
    # 3. Loss matrix
    # --------------------------------------------------
    L_mat = calculate_weighted_constraint_matrix(
        Y=Y,
        H=H,
        D=D_opt,
        num_sources=k,
        loss_type=loss_type,
        normalize_D_cols=False,   # already normalized above
    )

    # --------------------------------------------------
    # 4. Convert delta/epsilon to vectors
    # --------------------------------------------------
    if np.isscalar(delta):
        delta_vec = np.full(k, float(delta))
    else:
        delta_vec = np.asarray(delta, dtype=float)

    if np.isscalar(epsilon):
        epsilon_vec = np.full(k, float(epsilon))
    else:
        epsilon_vec = np.asarray(epsilon, dtype=float)

    N, k = D_opt.shape
    print(f"   >>> [CVXPY-PerDomain] Setup: N={N} (Original N={N_orig}), k={k}, loss_type={loss_type}")

    # --------------------------------------------------
    # 5. Variables
    # --------------------------------------------------
    w = cp.Variable(k, nonneg=True, name='w')
    Q = cp.Variable((N, k), nonneg=True, name='Q')
    R = cp.Variable((k, k), nonneg=True, name='R')

    # --------------------------------------------------
    # 6. Objective
    # Same meaning as before, written safely בלי broadcast
    # --------------------------------------------------
    obj_terms = []
    for j in range(k):
        obj_terms.append(
            cp.sum(cp.multiply(D_opt[:, j], -cp.rel_entr(w[j], Q[:, j])))
        )
    objective = cp.Maximize(cp.sum(obj_terms))

    # --------------------------------------------------
    # 7. Constraints
    # --------------------------------------------------
    constraints = [
        cp.sum(w) == 1,
        cp.sum(Q, axis=1) == 1,
        cp.sum(R, axis=0) == 1,
    ]

    for t in range(k):
        p_t = D_opt[:, t]

        # KL constraint for domain t
        kl_terms_t = []
        for j in range(k):
            kl_terms_t.append(
                cp.sum(cp.multiply(p_t, cp.rel_entr(R[j, t], Q[:, j])))
            )
        constraints.append(cp.sum(kl_terms_t) <= delta_vec[t])

        # Risk constraint for domain t
        constraints.append(
            R[:, t] @ L_mat[:, t] <= epsilon_vec[t]
        )

    # --------------------------------------------------
    # 8. Solve
    # --------------------------------------------------
    prob = cp.Problem(objective, constraints)
    print(f"   >>> [CVXPY-PerDomain] Starting {solver_type} solve...")

    try:
        if solver_type == 'SCS':
            prob.solve(
                solver=cp.SCS,
                verbose=False,
                eps=1e-3,
                max_iters=5000,
            )
        elif solver_type == 'MOSEK':
            prob.solve(solver=cp.MOSEK, verbose=False)
        elif solver_type == 'CLARABEL':
            prob.solve(solver=cp.CLARABEL, verbose=False)
        else:
            print(f"   !!! Unknown solver {solver_type}, defaulting to SCS")
            prob.solve(
                solver=cp.SCS,
                verbose=False,
                eps=1e-3,
                max_iters=5000,
            )

    except Exception as e:
        print(f"   !!! [CVXPY-PerDomain] Solver Exception: {e}")
        return None, None

    if prob.status not in ["optimal", "optimal_inaccurate"]:
        print(f"   !!! [CVXPY-PerDomain] Failed/Infeasible. Status: {prob.status}")
        return None, None

    print(f"   >>> [CVXPY-PerDomain] Solved! Weights: {np.round(w.value, 3)}")
    return w.value, Q.value