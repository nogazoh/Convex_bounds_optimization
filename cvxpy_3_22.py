import cvxpy as cp
import numpy as np
from cvxpy_solver import calculate_expected_loss


# ============================================================
# Helpers
# ============================================================
def _flatten_if_needed(Y, D, H):
    """
    Supports either:
      - D: (N, k), H: (N, k), Y: (N,)
    or
      - D: (N0, C, k), H: (N0, C, k), Y: (N0, C)
    Returns flattened (Y, D, H) and (N, k).
    """
    if D.ndim == 3:
        _, _, k = D.shape
        D = D.reshape(-1, k)
        H = H.reshape(-1, k)
        Y = Y.reshape(-1)
    N, k = D.shape
    return Y, D, H, N, k


# def calculate_expected_loss(Y, H, D, num_sources):
#     """
#     L_mat[j, t] = E_{x~D_t}[ ell(h_j(x), y) ].
#     Using absolute error (same as your older code).
#     """
#     L_mat = np.zeros((num_sources, num_sources))
#     for j in range(num_sources):
#         loss_vec = np.abs(H[:, j] - Y)
#         for t in range(num_sources):
#             p_t = D[:, t]
#             L_mat[j, t] = np.sum(p_t * loss_vec)
#     return L_mat


def compute_p_tilde(D, eps=1e-15):
    """
    Domain-anchored normalized expert density:
        p_tilde[i,j] = p_{i,j} / sum_s p_{i,s}

    If rows already sum to 1, p_tilde == D.
    eps prevents division by 0.
    """
    row_sum = np.sum(D, axis=1, keepdims=True)
    return D / np.maximum(row_sum, eps)


# ============================================================
# Problem (3.22): Domain-Anchored Smoothed Formulation
# ============================================================
def solve_convex_problem_domain_anchored_smoothed(
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
    normalize_domains=True,
    ptilde_eps=1e-15,
):
    """
    Solves the Domain-Anchored Smoothed formulation (your "Problem 3.22"):

        Define:
            p_tilde[i,j] = p_{i,j} / sum_s p_{i,s}

        max_{w,Q}  sum_{i=1}^N sum_{j=1}^k  w_j p_{i,j} log(q_{j|i}/w_j)
                   + (eta/k) * sum_{i=1}^N sum_{j=1}^k p_tilde[i,j] * log(q_{j|i})

        s.t.  for all t:
              sum_{j=1}^k w_j L_mat[j,t] <= epsilon

              for all i:
              sum_{j=1}^k q_{j|i} = 1 ,  q_{j|i} >= 0

              sum_j w_j = 1 ,  w_j >= 0

    Notes:
    - Concave objective (sum of concave terms) + convex constraints => CVXPY solvable.
    - If p_{i,*} is uniform across j (or if p_tilde becomes uniform), you recover the uniform-laplace style case.
    """

    # -------------------------
    # Shapes
    # -------------------------
    Y, D, H, N, k = _flatten_if_needed(Y, D, H)

    # Optional: normalize D rows (if your "domain distributions" are unnormalized scores)
    if normalize_domains:
        D = compute_p_tilde(D, eps=ptilde_eps)

    # -------------------------
    # Loss matrix
    # -------------------------
    L_mat = calculate_expected_loss(Y, H, D, k)

    # -------------------------
    # p_tilde for anchoring
    # -------------------------
    p_tilde = compute_p_tilde(D, eps=ptilde_eps)

    # -------------------------
    # Variables
    # -------------------------
    w = cp.Variable(k, nonneg=True, name="w")
    Q = cp.Variable((N, k), nonneg=True, name="Q")

    # -------------------------
    # Objective
    # -------------------------
    # First term: sum_{i,j} w_j p_{i,j} log(q_{j|i}/w_j)
    main_terms = []
    for j in range(k):
        main_terms.append(
            cp.sum(cp.multiply(D[:, j], -cp.rel_entr(w[j], Q[:, j])))
        )
    main_obj = cp.sum(main_terms)

    # Anchored smoothing: (eta/k) * sum_{i,j} p_tilde[i,j] log q_{j|i}
    anchored_smooth_obj = (eta / k) * cp.sum(cp.multiply(p_tilde, cp.log(Q)))

    objective = cp.Maximize(main_obj + anchored_smooth_obj)

    # -------------------------
    # Constraints
    # -------------------------
    constraints = [
        cp.sum(w) == 1,
        w >= w_min,
        cp.sum(Q, axis=1) == 1,
        Q >= q_min,
    ]
    for t in range(k):
        constraints.append(w @ L_mat[:, t] <= epsilon)

    # -------------------------
    # Solve
    # -------------------------
    prob = cp.Problem(objective, constraints)

    print(
        f"[Problem (3.22)] Solving domain-anchored smoothed optimization | "
        f"N={N}, k={k}, epsilon={epsilon:.2e}, eta={eta:.2e}, solver={solver_type}"
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
            print("[Problem (3.22)] Unknown solver, falling back to SCS")
            prob.solve(
                solver=cp.SCS,
                verbose=False,
                eps=scs_eps,
                max_iters=scs_max_iters,
            )
    except Exception as e:
        print(f"[Problem (3.22)] Solver exception: {e}")
        return None, None

    if prob.status not in ["optimal", "optimal_inaccurate"]:
        print(f"[Problem (3.22)] Optimization failed | status={prob.status}")
        return None, None

    # -------------------------
    # Results + diagnostics
    # -------------------------
    w_val = np.asarray(w.value).reshape(-1)
    Q_val = np.asarray(Q.value)

    print(
        f"[Problem (3.22)] Solved successfully | status={prob.status}\n"
        f"  • w (mixture over sources): {np.round(w_val, 6)}\n"
        f"    sum={w_val.sum():.6f}, min={w_val.min():.2e}\n"
        f"  • q_{{j|i}} diagnostics:\n"
        f"    row-sum min/max = "
        f"({Q_val.sum(axis=1).min():.6f}, {Q_val.sum(axis=1).max():.6f}), "
        f"min entry = {Q_val.min():.2e}"
    )

    for t in range(k):
        risk_t = w_val @ L_mat[:, t]
        print(
            f"  • [Problem (3.22)] risk(domain {t}) = {risk_t:.4e} "
            f"(<= {epsilon:.4e})"
        )

    return w_val, Q_val


# ============================================================
# Optional: Closed-form q_{j|i} for verification (paper check)
# ============================================================
def closed_form_q_domain_anchored(w, p_row, eta, eps=1e-15):
    """
    Closed-form from your draft for the domain-anchored smoothing:

        q^*_{j|i} = ((w_j + gamma_i) p_{i,j}) / sum_s ((w_s + gamma_i) p_{i,s})
        where gamma_i = eta / (k * sum_s p_{i,s})

    This function computes q^*_i (length k) for a single i, given:
      - w: (k,)
      - p_row: (k,)  (the row p_{i,*})
      - eta: scalar
      - k inferred from len(w)

    If p_row sums to 1, then gamma_i = eta/k.
    """
    w = np.asarray(w).reshape(-1)
    p_row = np.asarray(p_row).reshape(-1)
    k = len(w)

    denom_p = np.sum(p_row)
    gamma_i = eta / (k * max(denom_p, eps))

    num = (w + gamma_i) * p_row
    den = np.sum(num)
    if den <= eps:
        # fallback: uniform
        return np.ones(k) / k
    return num / den
