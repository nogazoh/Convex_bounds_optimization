import cvxpy as cp
import numpy as np


def calculate_expected_loss(Y, H, D, num_sources):
    """
    Calculates the expected loss matrix L[j, t].
    """
    L_mat = np.zeros((num_sources, num_sources))

    for j in range(num_sources):
        loss_vec = np.abs(H[:, j] - Y)
        for t in range(num_sources):
            p_t = D[:, t]
            expected_loss = np.sum(p_t * loss_vec)
            L_mat[j, t] = expected_loss
    return L_mat


def solve_convex_problem_per_domain(Y, D, H, delta=1e-2, epsilon=1e-2, solver_type='SCS'):
    """
    Solves Problem D.1 from the paper with Per-Domain constraints.
    delta and epsilon can be SCALARS (applied to all) or LISTS/ARRAYS (per domain).
    """

    # --- STEP 1: Handle Shapes ---
    if D.ndim == 3:
        N_orig, C, k = D.shape
        D_flat = D.reshape(-1, k)
        H_flat = H.reshape(-1, k)
        Y_flat = Y.reshape(-1)
        print(f"[{solver_type}-PerDomain] Flattening input...")
        D = D_flat
        H = H_flat
        Y = Y_flat
    else:
        N, k = D.shape

    N, k = D.shape

    # --- STEP 2: Setup Problem ---
    L_mat = calculate_expected_loss(Y, H, D, k)

    w = cp.Variable(k, nonneg=True, name='w')
    Q = cp.Variable((N, k), nonneg=True, name='Q')
    R = cp.Variable((k, k), nonneg=True, name='R')

    # Objective
    obj_terms = []
    for j in range(k):
        term = cp.sum(cp.multiply(D[:, j], -cp.rel_entr(w[j], Q[:, j])))
        obj_terms.append(term)

    objective = cp.Maximize(cp.sum(obj_terms))

    # Basic Constraints
    constraints = []
    constraints.append(cp.sum(w) == 1)
    constraints.append(cp.sum(Q, axis=1) == 1)
    constraints.append(cp.sum(R, axis=0) == 1)

    # --- PER-DOMAIN CONSTRAINTS ---
    for t in range(k):
        p_t = D[:, t]

        # Determine specific delta/epsilon for this domain
        # If it's a list/array, take the t-th element. If scalar, use as is.
        curr_delta = delta[t] if (isinstance(delta, (list, np.ndarray))) else delta
        curr_epsilon = epsilon[t] if (isinstance(epsilon, (list, np.ndarray))) else epsilon

        # 1. KL Constraint
        kl_inner_sum = []
        for j in range(k):
            re = cp.rel_entr(R[j, t], Q[:, j])
            weighted_re = cp.multiply(p_t, re)
            kl_inner_sum.append(cp.sum(weighted_re))
        constraints.append(cp.sum(kl_inner_sum) <= curr_delta)

        # 2. Risk Constraint
        constraints.append(cp.sum(cp.multiply(R[:, t], L_mat[:, t])) <= curr_epsilon)

    # --- STEP 3: Solve ---
    prob = cp.Problem(objective, constraints)

    try:
        if solver_type == 'SCS':
            print("   >>> SCS Solver (Per-Domain) starting...")
            prob.solve(solver=cp.SCS, verbose=True, eps=1e-3, max_iters=5000)
        elif solver_type == 'MOSEK':
            print("   >>> MOSEK Solver (Per-Domain) starting...")
            prob.solve(solver=cp.MOSEK, verbose=True)
        elif solver_type == 'CLARABEL':
            print("   >>> CLARABEL Solver (Per-Domain) starting...")
            prob.solve(solver=cp.CLARABEL, verbose=True)
        else:
            prob.solve(solver=cp.SCS, verbose=True, eps=1e-3)

    except Exception as e:
        print(f"   !!! Solver Error: {e}")

    if prob.status not in ["optimal", "optimal_inaccurate"]:
        print(f"   [{solver_type}-PerDomain] Warning: Status {prob.status}. Returning uniform.")
        return np.ones(k) / k

    return w.value