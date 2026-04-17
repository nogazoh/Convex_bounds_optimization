import numpy as np


# ============================================================
# Helpers
# ============================================================

def _reduce_D_to_2d(D: np.ndarray) -> np.ndarray:
    """
    Converts D to shape (N, K).

    Supported inputs:
    - D shape (N, K)
    - D shape (N, C, K), where D is replicated across classes

    Returns:
    - D_2d shape (N, K)
    """
    D = np.asarray(D)

    if D.ndim == 2:
        return D

    if D.ndim == 3:
        return D[:, 0, :]

    raise ValueError(f"D must have shape (N,K) or (N,C,K). Got {D.shape}")


def _extract_true_labels(Y: np.ndarray) -> np.ndarray:
    """
    Converts one-hot labels Y of shape (N, C) to integer labels of shape (N,).
    """
    Y = np.asarray(Y)
    if Y.ndim != 2:
        raise ValueError(f"Y must have shape (N,C). Got {Y.shape}")
    return Y.argmax(axis=1)


def _safe_normalize_weights(w: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Normalizes a weight vector to sum to 1.
    If the sum is too small, returns the original vector.
    """
    s = np.sum(w)
    if s <= eps:
        return w
    return w / s


# ============================================================
# Solver-side weighted losses (weighted by D)
# ============================================================

def calculate_weighted_constraint_matrix_01(
    Y: np.ndarray,
    H: np.ndarray,
    D: np.ndarray,
    num_sources: int,
    normalize_D_cols: bool = True,
) -> np.ndarray:
    """
    Computes the weighted 0-1 loss matrix L_mat[j, t] for the solver constraints:

        L_mat[j, t] = sum_i p_{i|t} * 1[pred_j(i) != y_i]

    where:
    - j = source classifier index
    - t = domain / source column used for weighting
    - p_{i|t} comes from D[:, t]

    Inputs:
    - Y: shape (N, C), one-hot true labels
    - H: shape (N, C, K), class probabilities/logits-like outputs per source
    - D: shape (N, K) or (N, C, K)
    - num_sources: K
    - normalize_D_cols: whether to normalize each D column to sum to 1

    Returns:
    - L_mat: shape (K, K)
    """
    D2 = _reduce_D_to_2d(D).astype(float)
    Y = np.asarray(Y)
    H = np.asarray(H)

    N, K = D2.shape
    if K != num_sources:
        raise ValueError(f"num_sources={num_sources}, but D has K={K}")
    if H.shape != (N, Y.shape[1], K):
        raise ValueError(f"H must have shape {(N, Y.shape[1], K)}. Got {H.shape}")

    if normalize_D_cols:
        col_sums = D2.sum(axis=0, keepdims=True)
        col_sums[col_sums == 0.0] = 1.0
        D2 = D2 / col_sums

    y_true = _extract_true_labels(Y)
    L_mat = np.zeros((K, K), dtype=float)

    for j in range(K):
        y_pred_j = H[:, :, j].argmax(axis=1)
        err_vec = (y_pred_j != y_true).astype(float)  # shape (N,)

        for t in range(K):
            p_t = D2[:, t]
            L_mat[j, t] = np.sum(p_t * err_vec)

    return L_mat


def calculate_weighted_constraint_matrix_ce(
    Y: np.ndarray,
    H: np.ndarray,
    D: np.ndarray,
    num_sources: int,
    normalize_D_cols: bool = True,
    clip_eps: float = 1e-12,
) -> np.ndarray:
    """
    Computes the weighted cross-entropy loss matrix L_mat[j, t] for the solver constraints:

        L_mat[j, t] = sum_i p_{i|t} * (-log H[i, y_i, j])

    where:
    - j = source classifier index
    - t = domain / source column used for weighting
    - p_{i|t} comes from D[:, t]

    Assumes H contains probabilities per class.
    If H already comes from softmax, this is correct.

    Inputs:
    - Y: shape (N, C), one-hot true labels
    - H: shape (N, C, K), class probabilities per source
    - D: shape (N, K) or (N, C, K)
    - num_sources: K
    - normalize_D_cols: whether to normalize each D column to sum to 1
    - clip_eps: numerical stability for log

    Returns:
    - L_mat: shape (K, K)
    """
    D2 = _reduce_D_to_2d(D).astype(float)
    Y = np.asarray(Y)
    H = np.asarray(H)

    N, K = D2.shape
    C = Y.shape[1]

    if K != num_sources:
        raise ValueError(f"num_sources={num_sources}, but D has K={K}")
    if H.shape != (N, C, K):
        raise ValueError(f"H must have shape {(N, C, K)}. Got {H.shape}")

    if normalize_D_cols:
        col_sums = D2.sum(axis=0, keepdims=True)
        col_sums[col_sums == 0.0] = 1.0
        D2 = D2 / col_sums

    y_true = _extract_true_labels(Y)
    row_idx = np.arange(N)

    L_mat = np.zeros((K, K), dtype=float)

    for j in range(K):
        p_true_j = H[row_idx, y_true, j]
        ce_vec = -np.log(np.clip(p_true_j, clip_eps, 1.0))  # shape (N,)

        for t in range(K):
            p_t = D2[:, t]
            L_mat[j, t] = np.sum(p_t * ce_vec)

    return L_mat


def calculate_weighted_constraint_matrix(
    Y: np.ndarray,
    H: np.ndarray,
    D: np.ndarray,
    num_sources: int,
    loss_type: str = "01",
    normalize_D_cols: bool = True,
    clip_eps: float = 1e-12,
) -> np.ndarray:
    """
    Wrapper for solver-side weighted loss matrices.

    loss_type:
    - "01" : weighted 0-1 loss
    - "ce" : weighted cross-entropy
    """
    loss_type = loss_type.lower()

    if loss_type == "01":
        return calculate_weighted_constraint_matrix_01(
            Y=Y,
            H=H,
            D=D,
            num_sources=num_sources,
            normalize_D_cols=normalize_D_cols,
        )

    if loss_type == "ce":
        return calculate_weighted_constraint_matrix_ce(
            Y=Y,
            H=H,
            D=D,
            num_sources=num_sources,
            normalize_D_cols=normalize_D_cols,
            clip_eps=clip_eps,
        )

    raise ValueError(f"Unsupported loss_type='{loss_type}'. Use '01' or 'ce'.")


# ============================================================
# Evaluation-side metrics (NOT weighted by D)
# ============================================================

def evaluate_predictions_accuracy(
    probs: np.ndarray,
    Y: np.ndarray,
) -> float:
    """
    Standard unweighted accuracy.

    Inputs:
    - probs: shape (N, C), predicted class scores/probabilities
    - Y: shape (N, C), one-hot labels

    Returns:
    - accuracy in [0, 100]
    """
    probs = np.asarray(probs)
    Y = np.asarray(Y)

    y_true = _extract_true_labels(Y)
    y_pred = probs.argmax(axis=1)
    return 100.0 * np.mean(y_pred == y_true)


def evaluate_predictions_error_rate(
    probs: np.ndarray,
    Y: np.ndarray,
) -> float:
    """
    Standard unweighted 0-1 error rate in [0, 1].
    """
    return 1.0 - (evaluate_predictions_accuracy(probs, Y) / 100.0)


def evaluate_predictions_cross_entropy(
    probs: np.ndarray,
    Y: np.ndarray,
    clip_eps: float = 1e-12,
) -> float:
    """
    Standard unweighted cross-entropy on the evaluated dataset.

    Inputs:
    - probs: shape (N, C), predicted class probabilities
    - Y: shape (N, C), one-hot labels

    Returns:
    - mean CE
    """
    probs = np.asarray(probs)
    Y = np.asarray(Y)

    y_true = _extract_true_labels(Y)
    row_idx = np.arange(len(y_true))
    p_true = probs[row_idx, y_true]
    ce = -np.log(np.clip(p_true, clip_eps, 1.0))
    return float(np.mean(ce))


# ============================================================
# Final solution evaluation from w / Q
# ============================================================

def combine_predictions_with_w(
    w: np.ndarray,
    D: np.ndarray,
    H: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Builds final normalized distribution-weighted predictions using global weights w:

        preds_i(c) = [sum_j w_j D_ij H_ij(c)] / [sum_j w_j D_ij]

    Inputs:
    - w: shape (K,)
    - D: shape (N, C, K) or (N, K)
         If D is (N, C, K), it is assumed replicated across classes.
    - H: shape (N, C, K)

    Returns:
    - preds: shape (N, C), normalized over classes
    """
    w = np.asarray(w, dtype=float).reshape(-1)
    H = np.asarray(H, dtype=float)

    if H.ndim != 3:
        raise ValueError(f"H must have shape (N, C, K). Got {H.shape}")

    N, C, K = H.shape
    if w.shape[0] != K:
        raise ValueError(f"w must have length {K}. Got {w.shape[0]}")

    if D.ndim == 2:
        D2 = np.asarray(D, dtype=float)
        if D2.shape != (N, K):
            raise ValueError(f"D must have shape {(N, K)}. Got {D2.shape}")
    elif D.ndim == 3:
        D3 = np.asarray(D, dtype=float)
        if D3.shape != (N, C, K):
            raise ValueError(f"D must have shape {(N, C, K)}. Got {D3.shape}")
        D2 = D3[:, 0, :]   # assume replicated across classes
    else:
        raise ValueError(f"D must have shape (N,K) or (N,C,K). Got {D.shape}")

    # numerator: (N, C)
    numerator = np.sum(H * (D2[:, None, :] * w[None, None, :]), axis=2)

    # denominator: (N, 1)
    denominator = np.sum(D2 * w[None, :], axis=1, keepdims=True)

    preds = numerator / np.clip(denominator, eps, None)

    # extra numerical safety: renormalize over classes
    preds_sum = preds.sum(axis=1, keepdims=True)
    preds = preds / np.clip(preds_sum, eps, None)

    return preds

def combine_predictions_with_q(
    Q: np.ndarray,
    H: np.ndarray,
) -> np.ndarray:
    """
    Builds final predictions from Q.

    Supported:
    - Q shape (N, K)
    - Q shape (N*C, K), reshaped to (N, C, K)

    Returns:
    - preds: shape (N, C)
    """
    Q = np.asarray(Q)
    H = np.asarray(H)

    N, C, K = H.shape

    if Q.shape == (N, K):
        return (H * Q[:, None, :]).sum(axis=2)

    if Q.shape == (N * C, K):
        Q3 = Q.reshape(N, C, K)
        return (H * Q3).sum(axis=2)

    raise ValueError(f"Unexpected Q shape {Q.shape}, expected {(N, K)} or {(N*C, K)}")


def evaluate_solution_with_w(
    w: np.ndarray,
    D: np.ndarray,
    H: np.ndarray,
    Y: np.ndarray,
    metric: str = "accuracy",
    clip_eps: float = 1e-12,
) -> float:
    """
    Evaluates final solution from w using an unweighted metric.

    metric:
    - "accuracy"
    - "error"
    - "ce"
    """
    preds = combine_predictions_with_w(w=w, D=D, H=H)
    metric = metric.lower()

    if metric == "accuracy":
        return evaluate_predictions_accuracy(preds, Y)
    if metric == "error":
        return evaluate_predictions_error_rate(preds, Y)
    if metric == "ce":
        return evaluate_predictions_cross_entropy(preds, Y, clip_eps=clip_eps)

    raise ValueError(f"Unsupported metric='{metric}'.")


def evaluate_solution_with_q(
    Q: np.ndarray,
    H: np.ndarray,
    Y: np.ndarray,
    metric: str = "accuracy",
    clip_eps: float = 1e-12,
) -> float:
    """
    Evaluates final solution from Q using an unweighted metric.

    metric:
    - "accuracy"
    - "error"
    - "ce"
    """
    preds = combine_predictions_with_q(Q=Q, H=H)
    metric = metric.lower()

    if metric == "accuracy":
        return evaluate_predictions_accuracy(preds, Y)
    if metric == "error":
        return evaluate_predictions_error_rate(preds, Y)
    if metric == "ce":
        return evaluate_predictions_cross_entropy(preds, Y, clip_eps=clip_eps)

    raise ValueError(f"Unsupported metric='{metric}'.")