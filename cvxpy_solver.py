import cvxpy as cp
import numpy as np


def calculate_expected_loss(Y, H, D, num_sources):
    """
    Calculates the expected loss matrix L[j, t].
    """
    L_mat = np.zeros((num_sources, num_sources))  # (hypothesis j, domain t)

    for j in range(num_sources):
        # Loss calculation (Absolute Error)
        loss_vec = np.abs(H[:, j] - Y)

        for t in range(num_sources):
            p_t = D[:, t]
            # Expected loss: weighted sum
            expected_loss = np.sum(p_t * loss_vec)
            L_mat[j, t] = expected_loss

    return L_mat


def solve_convex_problem_mosek(Y, D, H, delta=1e-2, epsilon=1e-2, solver_type='SCS'):
    """
    Solves the convex optimization problem using the specified solver (SCS or MOSEK).

    Args:
        solver_type (str): 'SCS' (fast, default) or 'MOSEK' (precise but slower).
    """

    # --- STEP 1: Handle Shapes ---
    if D.ndim == 3:
        N_orig, C, k = D.shape
        D_flat = D.reshape(-1, k)
        H_flat = H.reshape(-1, k)
        Y_flat = Y.reshape(-1)
        print(f"[{solver_type} Solver] Flattening from ({N_orig}, {C}, {k}) to ({N_orig * C}, {k})")
        D = D_flat
        H = H_flat
        Y = Y_flat
    else:
        N, k = D.shape
        print(f"[{solver_type} Solver] Standard input detected. Shape: ({N}, {k})")

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

    # Constraints
    constraints = []
    constraints.append(cp.sum(w) == 1)
    constraints.append(cp.sum(Q, axis=1) == 1)
    constraints.append(cp.sum(R, axis=0) == 1)

    # KL Constraint
    kl_terms = []
    for t in range(k):
        p_t = D[:, t]
        inner_sum_terms = []
        for j in range(k):
            re = cp.rel_entr(R[j, t], Q[:, j])
            weighted_re = cp.multiply(p_t, re)
            inner_sum_terms.append(cp.sum(weighted_re))
        kl_terms.append(cp.sum(inner_sum_terms))

    constraints.append((1.0 / k) * cp.sum(kl_terms) <= delta)

    # Loss Constraint
    for t in range(k):
        constraints.append(cp.sum(cp.multiply(R[:, t], L_mat[:, t])) <= epsilon)

    # --- STEP 3: Solve with Selected Solver ---
    prob = cp.Problem(objective, constraints)

    try:
        if solver_type == 'SCS':
            print("   >>> SCS Solver starting... (Fast Mode)")
            # eps=1e-3 is usually sufficient for ML weighting tasks
            prob.solve(solver=cp.SCS, verbose=True, eps=1e-3, max_iters=5000)

        elif solver_type == 'MOSEK':
            print("   >>> MOSEK Solver starting... (High Precision Mode)")
            prob.solve(solver=cp.MOSEK, verbose=True)

        elif solver_type == 'CLARABEL':
            print("   >>> CLARABEL Solver starting...")
            prob.solve(solver=cp.CLARABEL, verbose=True)

        else:
            print(f"   !!! Unknown solver {solver_type}, defaulting to SCS")
            prob.solve(solver=cp.SCS, verbose=True, eps=1e-3)

    except Exception as e:
        print(f"   !!! Solver Error: {e}")

    # Check status
    if prob.status not in ["optimal", "optimal_inaccurate"]:
        print(f"   [{solver_type} Solver] Warning: Status {prob.status}. Returning uniform.")
        return np.ones(k) / k

    return w.value