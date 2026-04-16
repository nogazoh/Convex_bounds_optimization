import os
import json
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Sequence, Tuple, List

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# ============================================================
# Optional imports from your codebase
# ============================================================
try:
    from cvxpy_3_31_kkt import solve_convex_problem_smoothed_kl_331
except Exception:
    solve_convex_problem_smoothed_kl_331 = None

try:
    from cvxpy_3_33_kkt import solve_convex_problem_smoothed_original_p_333
except Exception:
    solve_convex_problem_smoothed_original_p_333 = None

try:
    from cvxpy_solver import calculate_expected_loss as external_calculate_expected_loss
except Exception:
    external_calculate_expected_loss = None


# ============================================================
# Config
# ============================================================
@dataclass
class SimpleToyConfig:
    name: str = "simple_toy_clear"

    n_points: int = 20
    n_sources: int = 3
    n_classes: int = 2
    random_state: int = 7

    # Target geometry
    class0_mean: Tuple[float, float] = (-1.35, -0.10)
    class1_mean: Tuple[float, float] = (1.25, 0.10)
    class0_scale: Tuple[float, float] = (0.38, 0.34)
    class1_scale: Tuple[float, float] = (0.38, 0.34)

    # Source distributions over target points
    special_mass: float = 0.5
    special_indices: Tuple[int, int, int] = (0, 7, 14)
    target_lambda: Tuple[float, float, float] = (0.25, 0.45, 0.30)

    # Source-specific label generation
    # Intentionally different directions so the SVMs are different
    base_ws: Tuple[Tuple[float, float], ...] = (
        (1.00, 0.05),    # almost vertical
        (0.10, 1.00),    # almost horizontal
        (-0.95, 0.85),   # diagonal
    )
    base_bs: Tuple[float, float, float] = (0.00, -0.08, 0.12)
    desired_error_vec: Tuple[float, float, float] = (0.08, 0.12, 0.10)
    svm_flip_strength: Tuple[float, float, float] = (0.10, 0.14, 0.18)

    # SVM
    svm_c: float = 20.0
    svm_kernel: str = "linear"
    svm_gamma: float = 3.5
    svm_probability: bool = True

    # Solver
    solver_type: str = "SCS"
    eta_331: float = 1e-2
    eta_333: float = 1e-2
    q_min: float = 1e-8
    w_min: float = 1e-8
    scs_eps: float = 1e-5
    scs_max_iters: int = 25000

    # Epsilon is built from diag(L), then optionally inflated if infeasible
    epsilon_slack: float = 1e-6
    epsilon_min: float = 1e-8

    # Retry logic for infeasibility
    max_epsilon_retries: int = 12
    epsilon_retry_multiplier: float = 1.10


# ============================================================
# Constants / helpers
# ============================================================
SOURCE_COLORS = ["tab:red", "tab:blue", "tab:green"]
CLASS_MARKERS = {0: "o", 1: "^"}


def safe_mkdir(path: str):
    os.makedirs(path, exist_ok=True)


def one_hot(labels: np.ndarray, n_classes: int) -> np.ndarray:
    Y = np.zeros((len(labels), n_classes), dtype=float)
    Y[np.arange(len(labels)), labels] = 1.0
    return Y


# ============================================================
# Loss calculation
# ============================================================
def calculate_expected_loss(Y, H, D, num_sources):
    """
    L_mat[j, t] = expected loss of classifier j under source t.
    Y: (N,C)
    H: (N,C,K)
    D: (N,C,K) or (N,K)
    """
    if external_calculate_expected_loss is not None and D.ndim == 2 and H.ndim != 3:
        return external_calculate_expected_loss(Y, H, D, num_sources)

    L_mat = np.zeros((num_sources, num_sources), dtype=float)
    D_used = D[:, 0, :] if D.ndim == 3 else D

    for j in range(num_sources):
        loss_vec = np.abs(H[:, :, j] - Y)
        loss_vec = np.sum(loss_vec, axis=1) / 2.0
        for t in range(num_sources):
            p_t = D_used[:, t]
            L_mat[j, t] = float(np.sum(p_t * loss_vec))

    return L_mat


# ============================================================
# Evaluation aligned with your MSA_ALL_SUMMER logic
# ============================================================
def prediction_scores_from_w(w: np.ndarray, D: np.ndarray, H: np.ndarray) -> np.ndarray:
    return ((D * H) * np.asarray(w).reshape(1, 1, -1)).sum(axis=2)


def prediction_scores_from_q(Q: np.ndarray, H: np.ndarray) -> np.ndarray:
    Q = np.asarray(Q)
    N, C, K = H.shape

    if Q.shape == (N, K):
        return (H * Q[:, None, :]).sum(axis=2)

    if Q.shape == (N * C, K):
        Q3 = Q.reshape(N, C, K)
        return (H * Q3).sum(axis=2)

    raise ValueError(f"Unexpected Q shape {Q.shape}, expected {(N, K)} or {(N * C, K)}")


def evaluate_accuracy_wd(w, D, H, Y):
    scores = prediction_scores_from_w(np.asarray(w), D, H)
    return accuracy_score(Y.argmax(axis=1), scores.argmax(axis=1)) * 100.0


def evaluate_accuracy_q(Q, H, Y):
    scores = prediction_scores_from_q(np.asarray(Q), H)
    return accuracy_score(Y.argmax(axis=1), scores.argmax(axis=1)) * 100.0


def expected_loss_under_distribution_from_scores(pred_scores: np.ndarray, Y: np.ndarray, p: np.ndarray) -> float:
    pred_labels = pred_scores.argmax(axis=1)
    true_labels = Y.argmax(axis=1)
    return float(np.sum(p * (pred_labels != true_labels)))


def expected_acc_under_distribution_from_scores(pred_scores: np.ndarray, Y: np.ndarray, p: np.ndarray) -> float:
    pred_labels = pred_scores.argmax(axis=1)
    true_labels = Y.argmax(axis=1)
    return float(np.sum(p * (pred_labels == true_labels)))


# ============================================================
# Data generation
# ============================================================
def make_target_points(cfg: SimpleToyConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(cfg.random_state)

    n0 = cfg.n_points // 2
    n1 = cfg.n_points - n0

    x0 = rng.normal(loc=cfg.class0_mean, scale=cfg.class0_scale, size=(n0, 2))
    x1 = rng.normal(loc=cfg.class1_mean, scale=cfg.class1_scale, size=(n1, 2))

    X = np.vstack([x0, x1])
    y = np.concatenate([np.zeros(n0, dtype=int), np.ones(n1, dtype=int)])

    perm = rng.permutation(cfg.n_points)
    X = X[perm]
    y = y[perm]
    Y = one_hot(y, cfg.n_classes)
    return X, y, Y


def build_discrete_source_distributions(cfg: SimpleToyConfig) -> np.ndarray:
    D_loaded = np.full(
        (cfg.n_points, cfg.n_sources),
        (1.0 - cfg.special_mass) / (cfg.n_points - 1),
        dtype=float,
    )

    for j, idx in enumerate(cfg.special_indices):
        D_loaded[:, j] = (1.0 - cfg.special_mass) / (cfg.n_points - 1)
        D_loaded[idx, j] = cfg.special_mass

    D_loaded = D_loaded / D_loaded.sum(axis=0, keepdims=True)
    return D_loaded


def expand_D_for_classes(D_loaded: np.ndarray, n_classes: int) -> np.ndarray:
    return np.tile(D_loaded[:, None, :], (1, n_classes, 1))


# ============================================================
# Source-specific label construction
# ============================================================
def decision_function_with_flips(
    X: np.ndarray,
    base_w: np.ndarray,
    bias: float,
    flip_strength: float,
) -> np.ndarray:
    base = X @ base_w + bias
    perturb = flip_strength * np.sin(1.7 * X[:, 0] - 1.3 * X[:, 1])
    return base + perturb


def select_flip_indices_by_mass(
    margins: np.ndarray,
    source_mass: np.ndarray,
    desired_error: float,
    special_idx: int,
) -> np.ndarray:
    desired_error = float(np.clip(desired_error, 0.0, 0.999))
    if desired_error <= 0:
        return np.array([], dtype=int)

    order = np.argsort(np.abs(margins))
    ranked = [special_idx] + [idx for idx in order if idx != special_idx]

    chosen = []
    cumulative = 0.0
    for idx in ranked:
        chosen.append(idx)
        cumulative += source_mass[idx]
        if cumulative >= desired_error:
            break

    return np.array(sorted(set(chosen)), dtype=int)


# ============================================================
# Train source SVMs
# ============================================================
def train_source_svms(
    X: np.ndarray,
    y: np.ndarray,
    D_loaded: np.ndarray,
    cfg: SimpleToyConfig,
) -> Tuple[List[SVC], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    models: List[SVC] = []
    H = np.zeros((cfg.n_points, cfg.n_classes, cfg.n_sources), dtype=float)
    train_labels = np.zeros((cfg.n_points, cfg.n_sources), dtype=int)
    achieved_source_error = np.zeros(cfg.n_sources, dtype=float)
    train_label_accuracy = np.zeros(cfg.n_sources, dtype=float)

    for j in range(cfg.n_sources):
        margins = decision_function_with_flips(
            X,
            np.asarray(cfg.base_ws[j], dtype=float),
            float(cfg.base_bs[j]),
            float(cfg.svm_flip_strength[j]),
        )

        flip_idx = select_flip_indices_by_mass(
            margins=margins,
            source_mass=D_loaded[:, j],
            desired_error=float(cfg.desired_error_vec[j]),
            special_idx=cfg.special_indices[j],
        )

        y_train_j = y.copy()
        if len(flip_idx) > 0:
            y_train_j[flip_idx] = 1 - y_train_j[flip_idx]

        train_labels[:, j] = y_train_j

        if cfg.svm_kernel == "linear":
            clf = SVC(
                kernel="linear",
                C=cfg.svm_c,
                probability=cfg.svm_probability,
                random_state=cfg.random_state,
            )
        else:
            clf = SVC(
                kernel=cfg.svm_kernel,
                C=cfg.svm_c,
                gamma=cfg.svm_gamma,
                probability=cfg.svm_probability,
                random_state=cfg.random_state,
            )

        clf.fit(X, y_train_j)

        if cfg.svm_probability:
            proba = clf.predict_proba(X)
        else:
            pred = clf.predict(X)
            proba = one_hot(pred, cfg.n_classes)

        H[:, :, j] = proba
        models.append(clf)

        pred_labels = proba.argmax(axis=1)
        achieved_source_error[j] = float(np.sum(D_loaded[:, j] * (pred_labels != y)))
        train_label_accuracy[j] = float(np.mean(pred_labels == y_train_j))

    return models, H, train_labels, achieved_source_error, train_label_accuracy


# ============================================================
# Epsilon from exact loss seen by solver
# ============================================================
def build_epsilon_vec_from_loss_matrix(L_mat: np.ndarray, cfg: SimpleToyConfig) -> np.ndarray:
    eps = np.diag(L_mat).astype(float) + float(cfg.epsilon_slack)
    eps = np.maximum(eps, float(cfg.epsilon_min))
    return eps


# ============================================================
# Target distribution
# ============================================================
def compute_target_distribution(D_loaded: np.ndarray, target_lambda: Sequence[float]) -> np.ndarray:
    lam = np.asarray(target_lambda, dtype=float)
    lam = lam / lam.sum()
    p_target = D_loaded @ lam
    p_target = p_target / p_target.sum()
    return p_target


# ============================================================
# Solver wrappers with retry / epsilon inflation
# ============================================================
def _solver_331_call(Y, D, H, eps_vec, cfg):
    return solve_convex_problem_smoothed_kl_331(
        Y=Y.astype(float),
        D=D.copy(),
        H=H.astype(float).copy(),
        epsilon=np.asarray(eps_vec, dtype=float),
        eta=cfg.eta_331,
        solver_type=cfg.solver_type,
        q_min=cfg.q_min,
        w_min=cfg.w_min,
        scs_eps=cfg.scs_eps,
        scs_max_iters=cfg.scs_max_iters,
        normalize_D=True,
        return_kkt_details=True,
    )


def _solver_333_call(Y, D, H, eps_vec, cfg):
    return solve_convex_problem_smoothed_original_p_333(
        Y=Y.astype(float),
        D=D.copy(),
        H=H.astype(float).copy(),
        epsilon=np.asarray(eps_vec, dtype=float),
        eta=cfg.eta_333,
        solver_type=cfg.solver_type,
        q_min=cfg.q_min,
        w_min=cfg.w_min,
        scs_eps=cfg.scs_eps,
        scs_max_iters=cfg.scs_max_iters,
        return_kkt_details=True,
    )


def run_single_solver_with_retry(
    solver_name: str,
    solver_fn,
    Y: np.ndarray,
    D: np.ndarray,
    H: np.ndarray,
    base_epsilon_vec: np.ndarray,
    cfg: SimpleToyConfig,
) -> Dict[str, object]:
    current_eps = np.asarray(base_epsilon_vec, dtype=float).copy()
    multiplier = 1.0
    attempt_logs = []
    last_error = None

    for attempt in range(cfg.max_epsilon_retries + 1):
        try:
            out = solver_fn(Y, D, H, current_eps, cfg)

            if out is not None and out[0] is not None:
                w_opt, Q_opt, kkt = out
                status = None
                if isinstance(kkt, dict):
                    status = kkt.get("status", None)

                # accept any non-infeasible successful return
                if status is None or str(status).lower() not in {"infeasible", "unbounded"}:
                    return {
                        "status": "ok",
                        "w": np.asarray(w_opt, dtype=float),
                        "Q": np.asarray(Q_opt, dtype=float),
                        "kkt": kkt,
                        "epsilon_vec_used": current_eps.tolist(),
                        "epsilon_multiplier": multiplier,
                        "retry_count": attempt,
                        "attempt_logs": attempt_logs,
                    }

                last_error = f"status={status}"
            else:
                last_error = "solver returned None / no solution"

        except Exception as e:
            last_error = str(e)

        attempt_logs.append({
            "attempt": attempt,
            "epsilon_multiplier": multiplier,
            "epsilon_vec": current_eps.tolist(),
            "error": last_error,
        })

        multiplier *= cfg.epsilon_retry_multiplier
        current_eps = np.maximum(base_epsilon_vec * multiplier, cfg.epsilon_min)

    return {
        "status": "error",
        "error": f"{solver_name} failed after retries. last_error={last_error}",
        "epsilon_vec_used": current_eps.tolist(),
        "epsilon_multiplier": multiplier,
        "retry_count": cfg.max_epsilon_retries,
        "attempt_logs": attempt_logs,
    }


def run_solvers(
    Y: np.ndarray,
    D: np.ndarray,
    H: np.ndarray,
    epsilon_vec: np.ndarray,
    cfg: SimpleToyConfig,
) -> Dict[str, Optional[Dict[str, object]]]:
    results: Dict[str, Optional[Dict[str, object]]] = {"331": None, "333": None}

    if solve_convex_problem_smoothed_kl_331 is not None:
        results["331"] = run_single_solver_with_retry(
            solver_name="331",
            solver_fn=_solver_331_call,
            Y=Y,
            D=D,
            H=H,
            base_epsilon_vec=epsilon_vec,
            cfg=cfg,
        )

    if solve_convex_problem_smoothed_original_p_333 is not None:
        results["333"] = run_single_solver_with_retry(
            solver_name="333",
            solver_fn=_solver_333_call,
            Y=Y,
            D=D,
            H=H,
            base_epsilon_vec=epsilon_vec,
            cfg=cfg,
        )

    return results


# ============================================================
# Summary
# ============================================================
def summarize_results(
    Y: np.ndarray,
    H: np.ndarray,
    D: np.ndarray,
    D_loaded: np.ndarray,
    p_target: np.ndarray,
    L_mat: np.ndarray,
    solver_results: Dict[str, Optional[Dict[str, object]]],
    target_lambda: Sequence[float],
) -> Dict[str, object]:
    summary: Dict[str, object] = {"loss_matrix": L_mat.tolist()}
    true_labels = Y.argmax(axis=1)

    source_rows = []
    for j in range(H.shape[2]):
        pred_j = H[:, :, j].argmax(axis=1)
        own_loss = float(np.sum(D_loaded[:, j] * (pred_j != true_labels)))
        own_acc = float(np.sum(D_loaded[:, j] * (pred_j == true_labels)))
        target_loss = float(np.sum(p_target * (pred_j != true_labels)))
        target_acc = float(np.sum(p_target * (pred_j == true_labels)))

        source_rows.append({
            "method": f"source_{j}",
            "weights": [1.0 if t == j else 0.0 for t in range(H.shape[2])],
            "expected_loss_on_own_source": own_loss,
            "expected_accuracy_on_own_source": own_acc,
            "expected_loss_on_target": target_loss,
            "expected_accuracy_on_target": target_acc,
            "status": "ok",
        })
    summary["per_source_classifiers"] = source_rows

    uniform_w = np.ones(H.shape[2], dtype=float) / H.shape[2]
    uniform_scores = prediction_scores_from_w(uniform_w, D, H)
    summary["uniform"] = {
        "weights": uniform_w.tolist(),
        "loss_target": expected_loss_under_distribution_from_scores(uniform_scores, Y, p_target),
        "acc_target": expected_acc_under_distribution_from_scores(uniform_scores, Y, p_target),
        "status": "ok",
    }

    oracle_w = np.asarray(target_lambda, dtype=float)
    oracle_w = oracle_w / oracle_w.sum()
    oracle_scores = prediction_scores_from_w(oracle_w, D, H)
    summary["oracle"] = {
        "weights": oracle_w.tolist(),
        "loss_target": expected_loss_under_distribution_from_scores(oracle_scores, Y, p_target),
        "acc_target": expected_acc_under_distribution_from_scores(oracle_scores, Y, p_target),
        "status": "ok",
    }

    for tag, res in solver_results.items():
        key = f"solver_{tag}"
        if not res:
            summary[key] = {"status": "not_available"}
            continue
        if res.get("status") != "ok":
            summary[key] = res
            continue

        w = np.asarray(res["w"], dtype=float)
        Q = np.asarray(res["Q"], dtype=float)
        scores_w = prediction_scores_from_w(w, D, H)
        scores_q = prediction_scores_from_q(Q, H)

        summary[key] = {
            "status": "ok",
            "weights": w.tolist(),
            "Q_shape": list(Q.shape),
            "epsilon_vec_used": res["epsilon_vec_used"],
            "epsilon_multiplier": res["epsilon_multiplier"],
            "retry_count": res["retry_count"],
            "attempt_logs": res["attempt_logs"],
            "using_w": {
                "loss_target": expected_loss_under_distribution_from_scores(scores_w, Y, p_target),
                "acc_target": expected_acc_under_distribution_from_scores(scores_w, Y, p_target),
            },
            "using_q": {
                "loss_target": expected_loss_under_distribution_from_scores(scores_q, Y, p_target),
                "acc_target": expected_acc_under_distribution_from_scores(scores_q, Y, p_target),
            },
            "kkt": res.get("kkt", None),
        }

    return summary


# ============================================================
# Save / print
# ============================================================
def save_summary_json(summary: Dict[str, object], out_dir: str) -> str:
    path = os.path.join(out_dir, "summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return path


def save_solver_summary_txt(
    summary: Dict[str, object],
    cfg: SimpleToyConfig,
    base_epsilon_vec: np.ndarray,
    out_dir: str,
) -> str:
    path = os.path.join(out_dir, "solver_summary.txt")
    lines = []

    lines.append("============== SIMPLE TOY SUMMARY ==============\n")
    lines.append(f"name               : {cfg.name}")
    lines.append(f"desired_error      : {cfg.desired_error_vec}")
    lines.append(f"base_epsilon_diagL : {tuple(np.round(base_epsilon_vec, 8))}")
    lines.append(f"target_lambda      : {cfg.target_lambda}")
    lines.append(f"solver_type        : {cfg.solver_type}")
    lines.append(f"svm_kernel         : {cfg.svm_kernel}")
    lines.append("")
    lines.append("LOSS MATRIX L[j,t]")
    lines.append(str(np.round(np.asarray(summary['loss_matrix']), 6)))
    lines.append("")

    lines.append("PER-SOURCE CLASSIFIERS")
    for row in summary["per_source_classifiers"]:
        lines.append(
            f"{row['method']}: "
            f"loss_own={row['expected_loss_on_own_source']:.6f}, "
            f"acc_own={row['expected_accuracy_on_own_source']:.6f}, "
            f"loss_target={row['expected_loss_on_target']:.6f}, "
            f"acc_target={row['expected_accuracy_on_target']:.6f}"
        )

    lines.append("")
    lines.append(
        f"uniform: loss_target={summary['uniform']['loss_target']:.6f}, "
        f"acc_target={summary['uniform']['acc_target']:.6f}, "
        f"weights={np.round(np.asarray(summary['uniform']['weights']), 6)}"
    )
    lines.append(
        f"oracle : loss_target={summary['oracle']['loss_target']:.6f}, "
        f"acc_target={summary['oracle']['acc_target']:.6f}, "
        f"weights={np.round(np.asarray(summary['oracle']['weights']), 6)}"
    )
    lines.append("")

    for key in ["solver_331", "solver_333"]:
        item = summary.get(key, {})
        lines.append(f"{key}:")
        if item.get("status") != "ok":
            lines.append(f"  status               = {item.get('status', 'missing')}")
            if "error" in item:
                lines.append(f"  error                = {item['error']}")
            if "epsilon_multiplier" in item:
                lines.append(f"  final_multiplier     = {item['epsilon_multiplier']}")
            if "attempt_logs" in item:
                lines.append("  attempts:")
                for a in item["attempt_logs"]:
                    lines.append(
                        f"    attempt={a['attempt']}, mult={a['epsilon_multiplier']:.6f}, "
                        f"eps={np.round(np.asarray(a['epsilon_vec']), 8)}, err={a['error']}"
                    )
            lines.append("")
            continue

        lines.append("  status               = ok")
        lines.append(f"  weights              = {np.round(np.asarray(item['weights']), 6)}")
        lines.append(f"  Q_shape              = {item['Q_shape']}")
        lines.append(f"  epsilon_used         = {np.round(np.asarray(item['epsilon_vec_used']), 8)}")
        lines.append(f"  epsilon_multiplier   = {item['epsilon_multiplier']:.6f}")
        lines.append(f"  retry_count          = {item['retry_count']}")
        lines.append(f"  using_w loss_target  = {item['using_w']['loss_target']:.6f}")
        lines.append(f"  using_w acc_target   = {item['using_w']['acc_target']:.6f}")
        lines.append(f"  using_q loss_target  = {item['using_q']['loss_target']:.6f}")
        lines.append(f"  using_q acc_target   = {item['using_q']['acc_target']:.6f}")
        if item.get("kkt") is not None:
            kkt = item["kkt"]
            lines.append(f"  mu                   = {np.round(np.asarray(kkt.get('mu', [])), 6)}")
            lines.append(f"  mu_zero_count        = {kkt.get('mu_zero_count', '-')}")
            lines.append(f"  gamma_zero_count     = {kkt.get('gamma_zero_count', '-')}")
            lines.append(f"  solver_status        = {kkt.get('status', '-')}")
        if item.get("attempt_logs"):
            lines.append("  attempts:")
            for a in item["attempt_logs"]:
                lines.append(
                    f"    attempt={a['attempt']}, mult={a['epsilon_multiplier']:.6f}, "
                    f"eps={np.round(np.asarray(a['epsilon_vec']), 8)}, err={a['error']}"
                )
        lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


# ============================================================
# Plotting
# ============================================================
def plot_point_labels_and_source_mass(
    ax,
    X: np.ndarray,
    y: np.ndarray,
    D_loaded: np.ndarray,
    special_indices: Tuple[int, int, int],
):
    dominant = np.argmax(D_loaded, axis=1)

    for cls in [0, 1]:
        mask = y == cls
        ax.scatter(
            X[mask, 0],
            X[mask, 1],
            s=85,
            marker=CLASS_MARKERS[cls],
            c=[SOURCE_COLORS[d] for d in dominant[mask]],
            edgecolor="black",
            linewidth=0.6,
            alpha=0.9,
            label=f"class {cls}",
        )

    for i in range(len(X)):
        d0, d1, d2 = D_loaded[i]
        ax.text(
            X[i, 0] + 0.03,
            X[i, 1] + 0.03,
            f"{i}\n[{d0:.2f},{d1:.2f},{d2:.2f}]",
            fontsize=6,
        )

    for j, idx in enumerate(special_indices):
        ax.scatter(
            X[idx, 0],
            X[idx, 1],
            s=380,
            facecolors="none",
            edgecolors=SOURCE_COLORS[j],
            linewidths=2.6,
        )
        ax.text(
            X[idx, 0],
            X[idx, 1] - 0.18,
            f"special s{j}",
            color=SOURCE_COLORS[j],
            ha="center",
            fontsize=8,
            fontweight="bold",
        )

    ax.set_title("Points colored by dominant source + exact source masses")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.legend(loc="best", fontsize=8)


def plot_svms_on_same_axes(
    ax,
    X: np.ndarray,
    y: np.ndarray,
    models: List[SVC],
    train_labels: np.ndarray,
    D_loaded: np.ndarray,
    special_indices: Tuple[int, int, int],
):
    pad = 1.2
    xmin, ymin = X.min(axis=0) - pad
    xmax, ymax = X.max(axis=0) + pad
    gx = np.linspace(xmin, xmax, 240)
    gy = np.linspace(ymin, ymax, 240)
    xx, yy = np.meshgrid(gx, gy)
    grid = np.column_stack([xx.ravel(), yy.ravel()])

    for j, clf in enumerate(models):
        zz = clf.decision_function(grid).reshape(xx.shape)
        ax.contour(
            xx, yy, zz,
            levels=[0],
            linewidths=2.2,
            colors=[SOURCE_COLORS[j]],
        )

    dominant = np.argmax(D_loaded, axis=1)
    for cls in [0, 1]:
        mask = y == cls
        ax.scatter(
            X[mask, 0],
            X[mask, 1],
            s=75,
            marker=CLASS_MARKERS[cls],
            c=[SOURCE_COLORS[d] for d in dominant[mask]],
            edgecolor="black",
            linewidth=0.5,
            alpha=0.85,
        )

    for j in range(len(models)):
        flip_mask = train_labels[:, j] != y
        if np.any(flip_mask):
            ax.scatter(
                X[flip_mask, 0],
                X[flip_mask, 1],
                s=220,
                facecolors="none",
                edgecolors=SOURCE_COLORS[j],
                linewidths=1.7,
                alpha=0.7,
            )

    for j, idx in enumerate(special_indices):
        ax.scatter(
            X[idx, 0],
            X[idx, 1],
            s=380,
            facecolors="none",
            edgecolors=SOURCE_COLORS[j],
            linewidths=2.8,
        )

    for i in range(len(X)):
        ax.text(X[i, 0] + 0.02, X[i, 1] - 0.10, f"{i}", fontsize=6)

    ax.plot([], [], color=SOURCE_COLORS[0], label="SVM source 0")
    ax.plot([], [], color=SOURCE_COLORS[1], label="SVM source 1")
    ax.plot([], [], color=SOURCE_COLORS[2], label="SVM source 2")

    ax.set_title("All three SVM decision boundaries + source-dominant points")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.legend(loc="best", fontsize=7)


def create_overview_figure(
    X: np.ndarray,
    y: np.ndarray,
    Y: np.ndarray,
    D_loaded: np.ndarray,
    H: np.ndarray,
    train_labels: np.ndarray,
    train_label_accuracy: np.ndarray,
    models: List[SVC],
    solver_results: Dict[str, Optional[Dict[str, object]]],
    summary: Dict[str, object],
    p_target: np.ndarray,
    cfg: SimpleToyConfig,
    out_dir: str,
) -> str:
    fig, axes = plt.subplots(5, 2, figsize=(15, 21), constrained_layout=True)

    plot_point_labels_and_source_mass(
        axes[0, 0], X, y, D_loaded, cfg.special_indices
    )

    plot_svms_on_same_axes(
        axes[0, 1], X, y, models, train_labels, D_loaded, cfg.special_indices
    )

    width = 0.22
    x_idx = np.arange(cfg.n_points)
    ax = axes[1, 0]
    for j in range(cfg.n_sources):
        ax.bar(x_idx + (j - 1) * width, D_loaded[:, j], width=width, label=f"source {j}")
    ax.set_title("Discrete source masses over points")
    ax.set_xlabel("point i")
    ax.set_ylabel("mass")
    ax.legend()

    ax = axes[1, 1]
    ax.bar(np.arange(cfg.n_points), p_target)
    ax.set_title("Target mixture distribution over points")
    ax.set_xlabel("point i")
    ax.set_ylabel("p_target(i)")

    ax = axes[2, 0]
    im = ax.imshow(D_loaded, aspect="auto", cmap="magma")
    ax.set_title("Heatmap of D_loaded[i,j]")
    ax.set_xlabel("source j")
    ax.set_ylabel("point i")
    for j, idx in enumerate(cfg.special_indices):
        ax.add_patch(
            plt.Rectangle((j - 0.5, idx - 0.5), 1, 1, fill=False, edgecolor="cyan", linewidth=2.0)
        )
    fig.colorbar(im, ax=ax, shrink=0.85)

    ax = axes[2, 1]
    dominant = np.argmax(D_loaded, axis=1)
    sc = ax.scatter(X[:, 0], X[:, 1], c=dominant, s=220, cmap="tab10", alpha=0.45)
    for i in range(len(X)):
        d0, d1, d2 = D_loaded[i]
        ax.text(
            X[i, 0] + 0.03,
            X[i, 1] - 0.10,
            f"{i}: s{dominant[i]} [{d0:.2f},{d1:.2f},{d2:.2f}]",
            fontsize=6,
        )
    for j, idx in enumerate(cfg.special_indices):
        ax.scatter(
            X[idx, 0],
            X[idx, 1],
            s=380,
            facecolors="none",
            edgecolors=SOURCE_COLORS[j],
            linewidths=2.6,
        )
    ax.set_title("Dominant source per point + exact masses")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    fig.colorbar(sc, ax=ax, shrink=0.85)

    ax = axes[3, 0]
    res = solver_results.get("331", None)
    if not res or res.get("status") != "ok" or "Q" not in res:
        ax.text(0.5, 0.5, "solver 331 not available / not feasible", ha="center", va="center")
        ax.set_title("Solver 331: Q")
    else:
        Q = np.asarray(res["Q"], dtype=float)
        im = ax.imshow(Q, aspect="auto", cmap="plasma")
        ax.set_title(f"Solver 331: Q shape {Q.shape}")
        ax.set_xlabel("source j")
        ax.set_ylabel("point i or flattened (i,c)")
        fig.colorbar(im, ax=ax, shrink=0.85)

    ax = axes[3, 1]
    res = solver_results.get("333", None)
    if not res or res.get("status") != "ok" or "Q" not in res:
        ax.text(0.5, 0.5, "solver 333 not available / not feasible", ha="center", va="center")
        ax.set_title("Solver 333: Q")
    else:
        Q = np.asarray(res["Q"], dtype=float)
        im = ax.imshow(Q, aspect="auto", cmap="plasma")
        ax.set_title(f"Solver 333: Q shape {Q.shape}")
        ax.set_xlabel("source j")
        ax.set_ylabel("point i or flattened (i,c)")
        fig.colorbar(im, ax=ax, shrink=0.85)

    axes[4, 0].axis("off")
    txt1 = [
        f"Experiment: {cfg.name}",
        f"desired_error_vec = {cfg.desired_error_vec}",
        f"target_lambda     = {cfg.target_lambda}",
        f"base epsilon      = {tuple(np.round(summary['base_epsilon_vec'], 8))}",
        "",
        "Achieved source 0/1 error:",
    ]
    txt1.extend([f"source_{j}: {summary['achieved_source_error'][j]:.6f}" for j in range(cfg.n_sources)])
    txt1.append("")
    txt1.append("Uniform / Oracle:")
    txt1.append(f"uniform loss={summary['uniform']['loss_target']:.6f}, acc={summary['uniform']['acc_target']:.6f}")
    txt1.append(f"oracle  loss={summary['oracle']['loss_target']:.6f}, acc={summary['oracle']['acc_target']:.6f}")
    axes[4, 0].text(0.0, 1.0, "\n".join(txt1), va="top", fontsize=10)

    axes[4, 1].axis("off")
    txt2 = []
    for key in ["solver_331", "solver_333"]:
        item = summary.get(key, {})
        txt2.append(f"{key}:")
        if item.get("status") != "ok":
            txt2.append(f"  status = {item.get('status', 'missing')}")
            if "error" in item:
                txt2.append(f"  error  = {item['error']}")
            if "epsilon_multiplier" in item:
                txt2.append(f"  final mult = {item['epsilon_multiplier']:.6f}")
        else:
            txt2.append(f"  weights = {np.round(np.asarray(item['weights']), 6)}")
            txt2.append(f"  mult    = {item['epsilon_multiplier']:.6f}")
            txt2.append(f"  w: loss={item['using_w']['loss_target']:.6f}, acc={item['using_w']['acc_target']:.6f}")
            txt2.append(f"  q: loss={item['using_q']['loss_target']:.6f}, acc={item['using_q']['acc_target']:.6f}")
        txt2.append("")
    axes[4, 1].text(0.0, 1.0, "\n".join(txt2), va="top", fontsize=10)

    path = os.path.join(out_dir, "overview.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


# ============================================================
# Main
# ============================================================
def run_simple_toy_experiment(
    output_root: str = "./synthetic_msda_outputs",
    config: Optional[SimpleToyConfig] = None,
) -> Dict[str, object]:
    cfg = config or SimpleToyConfig()
    exp_dir = os.path.join(output_root, cfg.name)
    safe_mkdir(exp_dir)

    X, y, Y = make_target_points(cfg)
    D_loaded = build_discrete_source_distributions(cfg)
    D = expand_D_for_classes(D_loaded, cfg.n_classes)

    models, H, train_labels, achieved_source_error, train_label_accuracy = train_source_svms(
        X, y, D_loaded, cfg
    )

    L_mat = calculate_expected_loss(Y.astype(float), H.astype(float), D, cfg.n_sources)
    base_epsilon_vec = build_epsilon_vec_from_loss_matrix(L_mat, cfg)
    p_target = compute_target_distribution(D_loaded, cfg.target_lambda)

    solver_results = run_solvers(Y, D, H, base_epsilon_vec, cfg)

    summary = summarize_results(
        Y=Y,
        H=H,
        D=D,
        D_loaded=D_loaded,
        p_target=p_target,
        L_mat=L_mat,
        solver_results=solver_results,
        target_lambda=cfg.target_lambda,
    )

    summary["achieved_source_error"] = achieved_source_error.tolist()
    summary["base_epsilon_vec"] = base_epsilon_vec.tolist()
    summary["config"] = asdict(cfg)
    summary["target_distribution"] = p_target.tolist()
    summary["D_loaded"] = D_loaded.tolist()
    summary["Y"] = Y.tolist()
    summary["H"] = H.tolist()
    summary["train_labels_per_source"] = train_labels.tolist()
    summary["train_label_accuracy_per_source"] = train_label_accuracy.tolist()

    saved_paths: List[str] = []
    saved_paths.append(save_summary_json(summary, exp_dir))
    saved_paths.append(save_solver_summary_txt(summary, cfg, base_epsilon_vec, exp_dir))
    saved_paths.append(
        create_overview_figure(
            X=X,
            y=y,
            Y=Y,
            D_loaded=D_loaded,
            H=H,
            train_labels=train_labels,
            train_label_accuracy=train_label_accuracy,
            models=models,
            solver_results=solver_results,
            summary=summary,
            p_target=p_target,
            cfg=cfg,
            out_dir=exp_dir,
        )
    )

    summary["saved_paths"] = saved_paths
    return summary


if __name__ == "__main__":
    cfg = SimpleToyConfig(
        name="simple_toy_clear",
        n_points=20,
        n_sources=3,
        n_classes=2,
        random_state=7,
        class0_mean=(-1.35, -0.10),
        class1_mean=(1.25, 0.10),
        class0_scale=(0.38, 0.34),
        class1_scale=(0.38, 0.34),
        special_mass=0.5,
        special_indices=(0, 7, 14),
        target_lambda=(0.25, 0.45, 0.30),
        base_ws=(
            (1.00, 0.05),
            (0.10, 1.00),
            (-0.95, 0.85),
        ),
        base_bs=(0.00, -0.08, 0.12),
        desired_error_vec=(0.08, 0.12, 0.10),
        svm_flip_strength=(0.10, 0.14, 0.18),
        svm_kernel="linear",
        solver_type="SCS",
        max_epsilon_retries=12,
        epsilon_retry_multiplier=1.10,
    )

    run_simple_toy_experiment(
        output_root="./synthetic_msda_outputs",
        config=cfg,
    )