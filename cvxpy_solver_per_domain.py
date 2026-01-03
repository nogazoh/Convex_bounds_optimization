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
    Returns weights w if successful, or None if infeasible/failed.
    """

    # --- STEP 1: Handle Shapes ---
    if D.ndim == 3:
        N_orig, C, k = D.shape
        D_flat = D.reshape(-1, k)
        H_flat = H.reshape(-1, k)
        Y_flat = Y.reshape(-1)
        # print(f"[{solver_type}-PerDomain] Flattening input...")
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

    # PRINT 1: Start Notification
    print(f"   >>> [CVXPY-PerDomain] Starting {solver_type} solve (N={N}, k={k})...")

    try:
        if solver_type == 'SCS':
            prob.solve(solver=cp.SCS, verbose=False, eps=1e-3, max_iters=5000)
        elif solver_type == 'MOSEK':
            prob.solve(solver=cp.MOSEK, verbose=False)
        elif solver_type == 'CLARABEL':
            prob.solve(solver=cp.CLARABEL, verbose=False)
        else:
            print(f"   !!! Unknown solver {solver_type}, defaulting to SCS")
            prob.solve(solver=cp.SCS, verbose=False, eps=1e-3)

    except Exception as e:
        print(f"   !!! Solver Exception: {e}")
        return None

    # Check status - Return None if failed (No fallback to uniform)
    if prob.status not in ["optimal", "optimal_inaccurate"]:
        print(f"   !!! [CVXPY-PerDomain] Failed/Infeasible. Status: {prob.status}")
        return None

    # PRINT 2: Success Notification
    print(f"   >>> [CVXPY-PerDomain] Solved! Weights: {np.round(w.value, 3)}")
    return w.value