import cvxpy as cp
import numpy as np
from loss_functions import calculate_weighted_constraint_matrix


# def calculate_expected_loss(Y, H, D, num_sources):
#     """
#     Computes L_mat[j,t] = E_{i ~ p_t} [loss_j(i)]
#     Assumes columns of D are normalized (sum_i p_{it} = 1)
#     """
#     L_mat = np.zeros((num_sources, num_sources))
#
#     for j in range(num_sources):
#         # loss per sample
#         if H.ndim == 3:
#             loss_vec = np.abs(H[:, :, j] - Y)
#         else:
#             loss_vec = np.abs(H[:, j] - Y)
#
#         if loss_vec.ndim == 2:
#             loss_vec = np.sum(loss_vec, axis=1) / 2.0
#
#         for t in range(num_sources):
#             p_t = D[:, t]
#             L_mat[j, t] = np.sum(p_t * loss_vec)
#
#     return L_mat


def solve_convex_problem_mosek(
        Y,
        D,
        H,
        delta=1e-2,
        epsilon=1e-2,
        solver_type='SCS',
        loss_type='01',   # "01" or "ce"
):
    """
    Solves the convex optimization problem using the specified solver.

    loss_type:
    - "01" : weighted 0-1 loss in the constraints
    - "ce" : weighted cross-entropy loss in the constraints
    """

    # --------------------------------------------------
    # 1. Shape Handling & Loss Matrix Calculation
    # --------------------------------------------------
    if D.ndim == 3:
        N_orig, C, k = D.shape

        # For optimization we only need one copy across classes
        D_opt = D[:, 0, :]

        # Constraint loss matrix, weighted by D
        L_mat = calculate_weighted_constraint_matrix(
            Y=Y,
            H=H,
            D=D_opt,
            num_sources=k,
            loss_type=loss_type,
            normalize_D_cols=True,
        )

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

    # --------------------------------------------------
    # 2. Normalize D_opt for the Optimization Problem
    # --------------------------------------------------
    col_sums = D_opt.sum(axis=0, keepdims=True)
    col_sums[col_sums == 0] = 1.0
    D_opt = D_opt / col_sums

    N, k = D_opt.shape
    print(f"   >>> [Setup] optimization N={N} (Original N={N_orig}), k={k}, loss_type={loss_type}")

    # --------------------------------------------------
    # 3. Variables
    # --------------------------------------------------
    w = cp.Variable(k, nonneg=True, name='w')
    Q = cp.Variable((N, k), nonneg=True, name='Q')
    R = cp.Variable((k, k), nonneg=True, name='R')

    # --------------------------------------------------
    # 4. Objective
    # --------------------------------------------------
    obj_terms = []
    for j in range(k):
        term = cp.sum(cp.multiply(
            D_opt[:, j],
            -cp.rel_entr(w[j], Q[:, j])
        ))
        obj_terms.append(term)

    objective = cp.Maximize(cp.sum(obj_terms))

    # --------------------------------------------------
    # 5. Constraints
    # --------------------------------------------------
    constraints = []
    constraints.append(cp.sum(w) == 1)
    constraints.append(cp.sum(Q, axis=1) == 1)
    constraints.append(cp.sum(R, axis=0) == 1)

    # --- KL Constraint ---
    kl_terms = []
    for t in range(k):
        p_t = D_opt[:, t]
        inner_sum_terms = []
        for j in range(k):
            re = cp.rel_entr(R[j, t], Q[:, j])
            weighted_re = cp.multiply(p_t, re)
            inner_sum_terms.append(cp.sum(weighted_re))
        kl_terms.append(cp.sum(inner_sum_terms))

    constraints.append((1.0 / k) * cp.sum(kl_terms) <= delta)

    # --- Loss Constraints ---
    if isinstance(epsilon, (list, np.ndarray)):
        for t in range(k):
            constraints.append(
                cp.sum(cp.multiply(R[:, t], L_mat[:, t])) <= epsilon[t]
            )
    else:
        for t in range(k):
            constraints.append(
                cp.sum(cp.multiply(R[:, t], L_mat[:, t])) <= epsilon
            )

    # --------------------------------------------------
    # 6. Solve
    # --------------------------------------------------
    prob = cp.Problem(objective, constraints)

    print(f"   >>> [CVXPY] Starting {solver_type} solve...")

    try:
        if solver_type == 'SCS':
            prob.solve(
                solver=cp.SCS,
                verbose=False,
                eps=1e-3,
                max_iters=5000
            )
        elif solver_type == 'MOSEK':
            prob.solve(solver=cp.MOSEK, verbose=False)
        elif solver_type == 'CLARABEL':
            prob.solve(solver=cp.CLARABEL, verbose=False)
        else:
            print(f"   !!! Unknown solver {solver_type}, defaulting to SCS")
            prob.solve(solver=cp.SCS, verbose=False, eps=1e-3)

    except Exception as e:
        print(f"   !!! Solver Exception: {e}")
        return None, None

    if prob.status not in ["optimal", "optimal_inaccurate"]:
        print(f"   !!! [CVXPY] Failed/Infeasible. Status: {prob.status}")
        return None, None

    print(f"   >>> [CVXPY] Solved! Weights: {np.round(w.value, 3)}")
    return w.value, Q.value