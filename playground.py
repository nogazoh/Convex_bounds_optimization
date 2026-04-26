import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ============================================================
# CONFIG
# ============================================================

OUTDIR = Path("toy_manual_P_2d_p1_vs_p2")
OUTDIR.mkdir(parents=True, exist_ok=True)

SEED = 7
rng = np.random.default_rng(SEED)

# We keep one shared eta grid, but the core design is manual P
ETA_GRID = np.array([
    1e-3, 2e-3, 5e-3,
    1e-2, 2e-2, 5e-2,
    1e-1, 2e-1, 5e-1, 1.0
], dtype=float)

# constants used in the P1 residual proxy
C_CONST = 3.0
M_CONST = 1.0

TRUE_LABEL_COLORS = {0: "tab:blue", 1: "tab:orange"}
EXPERT_COLORS = ["black", "tab:green", "tab:cyan"]


# ============================================================
# BASIC UTILS
# ============================================================

def normalize_vector(v):
    v = np.asarray(v, dtype=float)
    s = v.sum()
    if s <= 0:
        raise ValueError("vector must have positive sum")
    return v / s


def normalize_columns(P):
    P = np.asarray(P, dtype=float)
    colsum = P.sum(axis=0, keepdims=True)
    if np.any(colsum <= 0):
        raise ValueError("each column in P must sum to a positive value")
    return P / colsum


def kl_divergence(u, v, eps=1e-12):
    u = np.clip(u, eps, 1.0)
    v = np.clip(v, eps, 1.0)
    return float(np.sum(u * np.log(u / v)))


# ============================================================
# EXPERTS
# ============================================================

def make_vertical_expert(threshold, name):
    return {
        "type": "vertical",
        "threshold": float(threshold),
        "name": name,
        "formula": f"x1 >= {threshold:.2f}",
    }


def make_horizontal_expert(threshold, name):
    return {
        "type": "horizontal",
        "threshold": float(threshold),
        "name": name,
        "formula": f"x2 >= {threshold:.2f}",
    }


def make_diagonal_sum_expert(threshold, name):
    return {
        "type": "diag_sum",
        "threshold": float(threshold),
        "name": name,
        "formula": f"x1 + x2 >= {threshold:.2f}",
    }


def expert_predict(expert, X):
    X = np.asarray(X, dtype=float)

    if expert["type"] == "vertical":
        return (X[:, 0] >= expert["threshold"]).astype(int)

    elif expert["type"] == "horizontal":
        return (X[:, 1] >= expert["threshold"]).astype(int)

    elif expert["type"] == "diag_sum":
        return (X[:, 0] + X[:, 1] >= expert["threshold"]).astype(int)

    else:
        raise ValueError(f"unknown expert type {expert['type']}")


def predict_all_experts(experts, X):
    return np.column_stack([expert_predict(e, X) for e in experts])


def draw_expert_line(ax, expert, color, linewidth=2.5, linestyle="-", label=None):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    if expert["type"] == "vertical":
        ax.axvline(expert["threshold"], color=color, linewidth=linewidth, linestyle=linestyle, label=label, zorder=8)

    elif expert["type"] == "horizontal":
        ax.axhline(expert["threshold"], color=color, linewidth=linewidth, linestyle=linestyle, label=label, zorder=8)

    elif expert["type"] == "diag_sum":
        xs = np.linspace(xlim[0], xlim[1], 240)
        ys = expert["threshold"] - xs
        ax.plot(xs, ys, color=color, linewidth=linewidth, linestyle=linestyle, label=label, zorder=8)

    else:
        raise ValueError(f"unknown expert type {expert['type']}")


# ============================================================
# CASE BUILDERS
# ============================================================

def build_case_A():
    """
    Case A: intended to be P1-friendly.
    Key idea:
      - Source 1 contains two different routing modes.
      - Points in region A are mostly associated with source 2.
      - Points in region B are mostly associated with source 3.
      - Thus within source 1, q varies substantially across samples.
      - P is manual and designed in advance.

    Geometry:
      - 8 points in 2D
      - top-left regime A
      - bottom-right regime B
    """

    # ----------------------------
    # 2D points
    # ----------------------------
    X = np.array([
        [-2.9,  2.3],
        [-2.6,  1.8],
        [-2.2,  2.2],
        [-2.0,  1.6],
        [ 1.8, -2.9],
        [ 2.2, -2.5],
        [ 2.7, -2.0],
        [ 2.9, -1.6],
    ], dtype=float)

    point_group = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])

    # ----------------------------
    # experts
    # ----------------------------
    h1 = make_diagonal_sum_expert(0.5, "h1")      # compromise expert
    h2 = make_vertical_expert(-2.4, "h2")         # region A expert
    h3 = make_horizontal_expert(-2.4, "h3")       # region B expert
    experts = [h1, h2, h3]

    # ----------------------------
    # labels
    # ----------------------------
    # In region A, labels follow h2
    # In region B, labels follow h3
    y = np.zeros(X.shape[0], dtype=int)
    idxA = point_group == "A"
    idxB = point_group == "B"
    y[idxA] = expert_predict(h2, X[idxA])
    y[idxB] = expert_predict(h3, X[idxB])

    # ----------------------------
    # manual P matrix
    # columns = sources 1,2,3
    # each column sums to 1
    #
    # source 1 spreads across both regimes
    # source 2 mostly on A
    # source 3 mostly on B
    #
    # This is the important part:
    # within source 1, points are split between
    # "A-like routing" and "B-like routing".
    # ----------------------------
    P = np.array([
        [0.18,   0.2495, 0.0005],
        [0.18,   0.2495, 0.0005],
        [0.18,   0.2495, 0.0005],
        [0.18,   0.2495, 0.0005],
        [0.07,   0.0005, 0.2495],
        [0.07,   0.0005, 0.2495],
        [0.07,   0.0005, 0.2495],
        [0.07,   0.0005, 0.2495],
    ], dtype=float)
    P = normalize_columns(P)

    # target mixture weights
    w = np.array([1/3, 1/3, 1/3], dtype=float)

    description = (
        "Case A (designed for P1): Source 1 is internally heterogeneous. "
        "Its mass is split between region A and region B. "
        "In region A, samples route toward source 2; in region B, "
        "samples route toward source 3. "
        "Hence within-source q-variation should be high."
    )

    return {
        "name": "Case A",
        "description": description,
        "X": X,
        "y": y,
        "point_group": point_group,
        "experts": experts,
        "P": P,
        "w": w,
    }


def build_case_B():
    """
    Case B: intended to be P2-friendly.
    Key idea:
      - each source is a single coherent regime
      - within each source, q_i should look very similar
      - P is manual and very concentrated per source
    """

    # ----------------------------
    # 2D points
    # ----------------------------
    X = np.array([
        [-3.6, -0.8],
        [-3.1, -0.2],
        [-2.8,  0.3],
        [-3.4,  0.9],

        [ 1.6,  1.7],
        [ 2.1,  2.2],
        [ 2.6,  2.7],
        [ 3.0,  3.2],

        [ 1.8, -3.2],
        [ 2.4, -2.9],
        [ 2.9, -2.4],
        [ 3.3, -2.0],
    ], dtype=float)

    point_group = np.array([
        "G1", "G1", "G1", "G1",
        "G2", "G2", "G2", "G2",
        "G3", "G3", "G3", "G3",
    ])

    # ----------------------------
    # experts
    # ----------------------------
    h1 = make_vertical_expert(-3.0, "h1")
    h2 = make_diagonal_sum_expert(4.6, "h2")
    h3 = make_horizontal_expert(-2.7, "h3")
    experts = [h1, h2, h3]

    # ----------------------------
    # labels
    # ----------------------------
    y = np.zeros(X.shape[0], dtype=int)
    idx1 = point_group == "G1"
    idx2 = point_group == "G2"
    idx3 = point_group == "G3"

    y[idx1] = expert_predict(h1, X[idx1])
    y[idx2] = expert_predict(h2, X[idx2])
    y[idx3] = expert_predict(h3, X[idx3])

    # ----------------------------
    # manual P matrix
    # columns sum to 1
    # each source is highly concentrated on one group
    # ----------------------------
    P = np.array([
        [0.247,  0.0015, 0.0015],
        [0.247,  0.0015, 0.0015],
        [0.247,  0.0015, 0.0015],
        [0.247,  0.0015, 0.0015],

        [0.0015, 0.247,  0.0015],
        [0.0015, 0.247,  0.0015],
        [0.0015, 0.247,  0.0015],
        [0.0015, 0.247,  0.0015],

        [0.0015, 0.0015, 0.247 ],
        [0.0015, 0.0015, 0.247 ],
        [0.0015, 0.0015, 0.247 ],
        [0.0015, 0.0015, 0.247 ],
    ], dtype=float)
    P = normalize_columns(P)

    w = np.array([1/3, 1/3, 1/3], dtype=float)

    description = (
        "Case B (designed for P2): each source is a single coherent regime. "
        "Within each source, the q-vectors should be very similar, "
        "so the domain prototype r_t should represent its samples well."
    )

    return {
        "name": "Case B",
        "description": description,
        "X": X,
        "y": y,
        "point_group": point_group,
        "experts": experts,
        "P": P,
        "w": w,
    }


# ============================================================
# CORE COMPUTATIONS
# ============================================================

def compute_alpha(P, w):
    numer = P * w[None, :]
    denom = numer.sum(axis=1, keepdims=True)
    return numer / denom


def compute_tau(P, w, eta):
    n = P.shape[0]
    local_mass = np.sum(P * w[None, :], axis=1)
    return local_mass / (local_mass + eta / n)


def compute_q_from_p1(P, w, eta):
    alpha = compute_alpha(P, w)
    tau = compute_tau(P, w, eta)[:, None]
    K = P.shape[1]
    return tau * alpha + (1.0 - tau) / K


def compute_L_matrix(P, expert_preds, y_true):
    losses = (expert_preds != y_true[:, None]).astype(float)
    L = P.T @ losses
    return L, losses


def compute_domain_prototypes(P, Q):
    """
    r_{.|t} = sum_i p_{i|t} q_{.|i}
    """
    K = P.shape[1]
    R = np.zeros((K, K), dtype=float)

    for t in range(K):
        rt = np.sum(P[:, t][:, None] * Q, axis=0)
        R[t] = normalize_vector(rt)

    return R


def compute_kappa(P, Q, R):
    """
    kappa_t = sum_i p_{i|t} KL(r_t || q_i)
    """
    n, K = Q.shape
    kappas = np.zeros(K, dtype=float)

    for t in range(K):
        rt = R[t]
        val = 0.0
        for i in range(n):
            val += P[i, t] * kl_divergence(rt, Q[i])
        kappas[t] = val

    return kappas


def compute_delta(L, R):
    """
    Delta_t = max_j |L_t^j - barL_t|
    where barL_t = sum_j r_{j|t} L_t^j
    """
    barL = np.sum(R * L, axis=1)
    Delta = np.max(np.abs(L - barL[:, None]), axis=1)
    return Delta, barL


def compute_within_source_q_variance(P, Q, R):
    """
    Weighted Euclidean dispersion:
      var_t = sum_i p_{i|t} ||q_i - r_t||^2
    """
    K = P.shape[1]
    vars_t = np.zeros(K, dtype=float)

    for t in range(K):
        diff = Q - R[t][None, :]
        sq = np.sum(diff**2, axis=1)
        vars_t[t] = np.sum(P[:, t] * sq)

    return vars_t


def compute_p1_residual(P, w, eta):
    tau = compute_tau(P, w, eta)
    source_terms = P.T @ np.sqrt(1.0 - tau)
    value = C_CONST * M_CONST * np.dot(w, source_terms)
    return float(value), source_terms, tau


def compute_p2_residual(P, Q, R, L, w):
    kappas = compute_kappa(P, Q, R)
    Delta, barL = compute_delta(L, R)
    value = float(np.dot(w, Delta * np.sqrt(2.0 * kappas)))
    return value, kappas, Delta, barL


def compute_case_metrics(case_data, eta):
    X = case_data["X"]
    y = case_data["y"]
    P = case_data["P"]
    w = case_data["w"]
    experts = case_data["experts"]

    alpha = compute_alpha(P, w)
    tau = compute_tau(P, w, eta)
    Q = compute_q_from_p1(P, w, eta)

    expert_preds = predict_all_experts(experts, X)
    L, losses = compute_L_matrix(P, expert_preds, y)
    R = compute_domain_prototypes(P, Q)

    p1_value, p1_source_terms, tau_vec = compute_p1_residual(P, w, eta)
    p2_value, kappas, Delta, barL = compute_p2_residual(P, Q, R, L, w)
    q_var = compute_within_source_q_variance(P, Q, R)

    winner = "P1" if p1_value < p2_value else ("P2" if p2_value < p1_value else "Equal")

    # diagnostics by source
    per_source_rows = []
    for t in range(P.shape[1]):
        weights_t = P[:, t]
        mean_max_alpha_t = np.sum(weights_t * np.max(alpha, axis=1))
        mean_max_q_t = np.sum(weights_t * np.max(Q, axis=1))
        per_source_rows.append({
            "source_t": t + 1,
            "kappa_t": kappas[t],
            "Delta_t": Delta[t],
            "barL_t": barL[t],
            "q_variance_t": q_var[t],
            "mean_max_alpha_t": mean_max_alpha_t,
            "mean_max_q_t": mean_max_q_t,
            "r_t1": R[t, 0],
            "r_t2": R[t, 1],
            "r_t3": R[t, 2],
            "L_t_h1": L[t, 0],
            "L_t_h2": L[t, 1],
            "L_t_h3": L[t, 2],
        })

    return {
        "eta": float(eta),
        "alpha": alpha,
        "tau": tau_vec,
        "Q": Q,
        "R": R,
        "L": L,
        "losses": losses,
        "expert_preds": expert_preds,
        "expert_target_err": losses.mean(axis=0),
        "kappa": kappas,
        "Delta": Delta,
        "barL": barL,
        "q_variance": q_var,
        "p1_residual": float(p1_value),
        "p2_residual": float(p2_value),
        "winner": winner,
        "p1_source_terms": p1_source_terms,
        "per_source_rows": per_source_rows,
    }


def sweep_eta(case_data, desired_winner):
    rows = []
    best_score = -np.inf
    best_metrics = None
    best_eta = None

    for eta in ETA_GRID:
        m = compute_case_metrics(case_data, eta)

        if desired_winner == "P1":
            score = m["p2_residual"] - m["p1_residual"]
        elif desired_winner == "P2":
            score = m["p1_residual"] - m["p2_residual"]
        else:
            raise ValueError("desired_winner must be P1 or P2")

        rows.append({
            "case": case_data["name"],
            "eta": float(eta),
            "p1_residual": m["p1_residual"],
            "p2_residual": m["p2_residual"],
            "winner": m["winner"],
            "selection_score": float(score),
            "mean_tau": float(np.mean(m["tau"])),
            "mean_max_alpha": float(np.mean(np.max(m["alpha"], axis=1))),
            "mean_max_q": float(np.mean(np.max(m["Q"], axis=1))),
        })

        if score > best_score:
            best_score = score
            best_eta = float(eta)
            best_metrics = m

    sweep_df = pd.DataFrame(rows)
    return best_eta, best_metrics, sweep_df


# ============================================================
# PLOTTING
# ============================================================

def get_plot_limits(X, pad=0.7):
    x1_min, x1_max = X[:, 0].min() - pad, X[:, 0].max() + pad
    x2_min, x2_max = X[:, 1].min() - pad, X[:, 1].max() + pad
    return x1_min, x1_max, x2_min, x2_max


def plot_case_overview(case_data, metrics, outpath):
    X = case_data["X"]
    y = case_data["y"]
    P = case_data["P"]
    experts = case_data["experts"]

    x1_min, x1_max, x2_min, x2_max = get_plot_limits(X, pad=0.9)

    fig, axes = plt.subplots(2, 2, figsize=(14, 11), sharex=True, sharey=True)
    axes = axes.ravel()

    # --------------------------------------------------
    # panel 1: true labels + experts + point ids
    # --------------------------------------------------
    ax = axes[0]

    for label in [0, 1]:
        idx = np.where(y == label)[0]
        ax.scatter(
            X[idx, 0], X[idx, 1],
            s=90,
            c=TRUE_LABEL_COLORS[label],
            edgecolors="black",
            linewidths=0.6,
            alpha=0.9,
            zorder=4
        )

    ax.set_xlim(x1_min, x1_max)
    ax.set_ylim(x2_min, x2_max)

    for j, e in enumerate(experts):
        draw_expert_line(ax, e, color=EXPERT_COLORS[j], linewidth=2.6)

    for i, (x1, x2) in enumerate(X):
        ax.text(x1 + 0.05, x2 + 0.05, str(i + 1), fontsize=9)

    ax.set_title(
        f"{case_data['name']}\n"
        f"{case_data['description']}\n"
        f"chosen eta = {metrics['eta']:.4f} | "
        f"P1 = {metrics['p1_residual']:.4f} | "
        f"P2 = {metrics['p2_residual']:.4f} | "
        f"winner = {metrics['winner']}",
        fontsize=11
    )
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.grid(True, alpha=0.25)

    legend_handles = [
        Line2D([0], [0], marker='o', color='w', label='True label 0',
               markerfacecolor=TRUE_LABEL_COLORS[0], markeredgecolor='black', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='True label 1',
               markerfacecolor=TRUE_LABEL_COLORS[1], markeredgecolor='black', markersize=10),
        Line2D([0], [0], color=EXPERT_COLORS[0], lw=2.5, label=f"{experts[0]['name']}: {experts[0]['formula']}"),
        Line2D([0], [0], color=EXPERT_COLORS[1], lw=2.5, label=f"{experts[1]['name']}: {experts[1]['formula']}"),
        Line2D([0], [0], color=EXPERT_COLORS[2], lw=2.5, label=f"{experts[2]['name']}: {experts[2]['formula']}"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=8)

    # --------------------------------------------------
    # panels 2,3,4 = P columns
    # --------------------------------------------------
    titles = [r"$P_{i\mid 1}$", r"$P_{i\mid 2}$", r"$P_{i\mid 3}$"]
    for t in range(3):
        ax = axes[t + 1]
        sc = ax.scatter(
            X[:, 0], X[:, 1],
            c=P[:, t],
            s=150,
            cmap="viridis",
            edgecolors="black",
            linewidths=0.6,
            alpha=0.95
        )
        for i, (x1, x2) in enumerate(X):
            ax.text(x1 + 0.05, x2 + 0.05, f"{i+1}", fontsize=9)
        ax.set_title(titles[t], fontsize=13)
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.grid(True, alpha=0.25)
        ax.set_xlim(x1_min, x1_max)
        ax.set_ylim(x2_min, x2_max)
        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel(titles[t], rotation=90)

    fig.tight_layout()
    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_expert_predictions(case_data, metrics, outpath):
    X = case_data["X"]
    y = case_data["y"]
    experts = case_data["experts"]
    preds = metrics["expert_preds"]

    x1_min, x1_max, x2_min, x2_max = get_plot_limits(X, pad=0.9)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8), sharex=True, sharey=True)

    for j, ax in enumerate(axes):
        ax.set_xlim(x1_min, x1_max)
        ax.set_ylim(x2_min, x2_max)

        draw_expert_line(ax, experts[j], color="black", linewidth=2.6)

        miss = preds[:, j] != y
        err = np.mean(miss)

        for label in [0, 1]:
            idx = np.where((y == label) & (~miss))[0]
            ax.scatter(
                X[idx, 0], X[idx, 1],
                s=90,
                c=TRUE_LABEL_COLORS[label],
                edgecolors="black",
                linewidths=0.6,
                alpha=0.9
            )

        idx_bad = np.where(miss)[0]
        ax.scatter(
            X[idx_bad, 0], X[idx_bad, 1],
            s=180,
            facecolors="none",
            edgecolors="red",
            linewidths=1.8,
            zorder=7
        )

        for i, (x1, x2) in enumerate(X):
            ax.text(x1 + 0.05, x2 + 0.05, str(i + 1), fontsize=9)

        ax.set_title(
            f"{experts[j]['name']}\n"
            f"{experts[j]['formula']}\n"
            f"target err = {err:.3f}",
            fontsize=10
        )
        ax.set_xlabel("$x_1$")
        if j == 0:
            ax.set_ylabel("$x_2$")
        ax.grid(True, alpha=0.25)

    handles = [
        Line2D([0], [0], marker='o', color='w', label='True label 0',
               markerfacecolor=TRUE_LABEL_COLORS[0], markeredgecolor='black', markersize=9),
        Line2D([0], [0], marker='o', color='w', label='True label 1',
               markerfacecolor=TRUE_LABEL_COLORS[1], markeredgecolor='black', markersize=9),
        Line2D([0], [0], marker='o', color='red', label='Misclassified',
               markerfacecolor='none', markeredgecolor='red', markersize=10, linewidth=0),
    ]
    axes[0].legend(handles=handles, loc="upper left", fontsize=9)

    fig.suptitle(f"{case_data['name']} — true labels and expert errors", fontsize=14)
    fig.tight_layout()
    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_q_bars(case_data, metrics, outpath):
    """
    bar chart for q_i across points
    """
    Q = metrics["Q"]
    n = Q.shape[0]
    ids = np.arange(1, n + 1)

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    labels = ["q_{1|i}", "q_{2|i}", "q_{3|i}"]

    for j, ax in enumerate(axes):
        ax.bar(ids, Q[:, j])
        ax.set_ylabel(labels[j])
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel("point index i")
    fig.suptitle(f"{case_data['name']} — q values per point", fontsize=14)
    fig.tight_layout()
    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_eta_sweep(df_A, df_B, outpath):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)

    for ax, df, title in zip(
        axes,
        [df_A, df_B],
        ["Case A eta sweep", "Case B eta sweep"]
    ):
        ax.plot(df["eta"], df["p1_residual"], marker="o", label="P1 residual")
        ax.plot(df["eta"], df["p2_residual"], marker="s", label="P2 residual")
        ax.set_xscale("log")
        ax.set_xlabel("eta")
        ax.set_ylabel("Residual")
        ax.set_title(title)
        ax.grid(True, alpha=0.30)
        ax.legend()

    fig.tight_layout()
    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# EXPORTS
# ============================================================

def build_point_df(case_data, metrics):
    X = case_data["X"]
    y = case_data["y"]
    P = case_data["P"]
    alpha = metrics["alpha"]
    Q = metrics["Q"]
    tau = metrics["tau"]
    preds = metrics["expert_preds"]
    losses = metrics["losses"]
    groups = case_data["point_group"]

    return pd.DataFrame({
        "case": case_data["name"],
        "point_id": np.arange(1, len(X) + 1),
        "group": groups,
        "x1": X[:, 0],
        "x2": X[:, 1],
        "y_true": y,
        "P_i1": P[:, 0],
        "P_i2": P[:, 1],
        "P_i3": P[:, 2],
        "alpha_1": alpha[:, 0],
        "alpha_2": alpha[:, 1],
        "alpha_3": alpha[:, 2],
        "q_1": Q[:, 0],
        "q_2": Q[:, 1],
        "q_3": Q[:, 2],
        "tau": tau,
        "pred_h1": preds[:, 0],
        "pred_h2": preds[:, 1],
        "pred_h3": preds[:, 2],
        "loss_h1": losses[:, 0],
        "loss_h2": losses[:, 1],
        "loss_h3": losses[:, 2],
    })


def build_source_df(case_data, metrics):
    rows = []
    for row in metrics["per_source_rows"]:
        rows.append({
            "case": case_data["name"],
            **row
        })
    return pd.DataFrame(rows)


def build_overall_df(case_data, metrics):
    return pd.DataFrame([{
        "case": case_data["name"],
        "description": case_data["description"],
        "chosen_eta": metrics["eta"],
        "p1_residual": metrics["p1_residual"],
        "p2_residual": metrics["p2_residual"],
        "winner": metrics["winner"],
        "mean_tau": float(np.mean(metrics["tau"])),
        "mean_max_alpha": float(np.mean(np.max(metrics["alpha"], axis=1))),
        "mean_max_q": float(np.mean(np.max(metrics["Q"], axis=1))),
        "target_err_h1": metrics["expert_target_err"][0],
        "target_err_h2": metrics["expert_target_err"][1],
        "target_err_h3": metrics["expert_target_err"][2],
    }])


def write_summary_txt(path, results):
    lines = []
    for case_data, metrics in results:
        lines.append("=" * 90)
        lines.append(case_data["name"])
        lines.append(case_data["description"])
        lines.append(f"chosen eta       = {metrics['eta']:.6f}")
        lines.append(f"P1 residual      = {metrics['p1_residual']:.6f}")
        lines.append(f"P2 residual      = {metrics['p2_residual']:.6f}")
        lines.append(f"winner           = {metrics['winner']}")
        lines.append(f"mean(tau)        = {np.mean(metrics['tau']):.6f}")
        lines.append(f"mean(max alpha)  = {np.mean(np.max(metrics['alpha'], axis=1)):.6f}")
        lines.append(f"mean(max q)      = {np.mean(np.max(metrics['Q'], axis=1)):.6f}")
        lines.append(f"kappa            = {np.round(metrics['kappa'], 6)}")
        lines.append(f"Delta            = {np.round(metrics['Delta'], 6)}")
        lines.append(f"q variance       = {np.round(metrics['q_variance'], 6)}")
        lines.append("L matrix:")
        lines.append(str(np.round(metrics["L"], 6)))
        lines.append("R matrix:")
        lines.append(str(np.round(metrics["R"], 6)))
        lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ============================================================
# MAIN
# ============================================================

def main():
    case_A = build_case_A()
    case_B = build_case_B()

    best_eta_A, best_metrics_A, sweep_df_A = sweep_eta(case_A, desired_winner="P1")
    best_eta_B, best_metrics_B, sweep_df_B = sweep_eta(case_B, desired_winner="P2")

    best_metrics_A["eta"] = best_eta_A
    best_metrics_B["eta"] = best_eta_B

    # Save CSVs
    point_df_A = build_point_df(case_A, best_metrics_A)
    point_df_B = build_point_df(case_B, best_metrics_B)
    source_df_A = build_source_df(case_A, best_metrics_A)
    source_df_B = build_source_df(case_B, best_metrics_B)
    overall_df_A = build_overall_df(case_A, best_metrics_A)
    overall_df_B = build_overall_df(case_B, best_metrics_B)

    point_df_A.to_csv(OUTDIR / "case_A_point_level.csv", index=False)
    point_df_B.to_csv(OUTDIR / "case_B_point_level.csv", index=False)
    source_df_A.to_csv(OUTDIR / "case_A_source_metrics.csv", index=False)
    source_df_B.to_csv(OUTDIR / "case_B_source_metrics.csv", index=False)

    pd.concat([overall_df_A, overall_df_B], ignore_index=True).to_csv(
        OUTDIR / "overall_summary.csv", index=False
    )
    pd.concat([sweep_df_A, sweep_df_B], ignore_index=True).to_csv(
        OUTDIR / "eta_sweep.csv", index=False
    )

    pd.DataFrame(case_A["P"], columns=["source1", "source2", "source3"]).to_csv(
        OUTDIR / "case_A_manual_P.csv", index=False
    )
    pd.DataFrame(case_B["P"], columns=["source1", "source2", "source3"]).to_csv(
        OUTDIR / "case_B_manual_P.csv", index=False
    )

    pd.DataFrame(best_metrics_A["L"], columns=["h1", "h2", "h3"], index=["source1", "source2", "source3"]).to_csv(
        OUTDIR / "case_A_L_matrix.csv"
    )
    pd.DataFrame(best_metrics_B["L"], columns=["h1", "h2", "h3"], index=["source1", "source2", "source3"]).to_csv(
        OUTDIR / "case_B_L_matrix.csv"
    )

    pd.DataFrame(best_metrics_A["R"], columns=["q1", "q2", "q3"], index=["source1", "source2", "source3"]).to_csv(
        OUTDIR / "case_A_R_matrix.csv"
    )
    pd.DataFrame(best_metrics_B["R"], columns=["q1", "q2", "q3"], index=["source1", "source2", "source3"]).to_csv(
        OUTDIR / "case_B_R_matrix.csv"
    )

    # Save plots
    plot_case_overview(case_A, best_metrics_A, OUTDIR / "case_A_overview.png")
    plot_case_overview(case_B, best_metrics_B, OUTDIR / "case_B_overview.png")

    plot_expert_predictions(case_A, best_metrics_A, OUTDIR / "case_A_experts.png")
    plot_expert_predictions(case_B, best_metrics_B, OUTDIR / "case_B_experts.png")

    plot_q_bars(case_A, best_metrics_A, OUTDIR / "case_A_q_bars.png")
    plot_q_bars(case_B, best_metrics_B, OUTDIR / "case_B_q_bars.png")

    plot_eta_sweep(sweep_df_A, sweep_df_B, OUTDIR / "eta_sweep.png")

    # metadata
    metadata = {
        "case_A": {
            "description": case_A["description"],
            "experts": case_A["experts"],
            "P_manual": case_A["P"].tolist(),
            "X": case_A["X"].tolist(),
            "y": case_A["y"].tolist(),
            "point_group": case_A["point_group"].tolist(),
            "chosen_eta": best_eta_A,
        },
        "case_B": {
            "description": case_B["description"],
            "experts": case_B["experts"],
            "P_manual": case_B["P"].tolist(),
            "X": case_B["X"].tolist(),
            "y": case_B["y"].tolist(),
            "point_group": case_B["point_group"].tolist(),
            "chosen_eta": best_eta_B,
        },
    }
    with open(OUTDIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    write_summary_txt(
        OUTDIR / "summary.txt",
        [
            (case_A, best_metrics_A),
            (case_B, best_metrics_B),
        ]
    )

    # Print concise summary
    print("\n" + "=" * 90)
    print("Finished.")
    print("=" * 90)

    for case_data, metrics in [(case_A, best_metrics_A), (case_B, best_metrics_B)]:
        print(f"\n{case_data['name']}")
        print(case_data["description"])
        print(f"chosen eta      = {metrics['eta']:.6f}")
        print(f"P1 residual     = {metrics['p1_residual']:.6f}")
        print(f"P2 residual     = {metrics['p2_residual']:.6f}")
        print(f"winner          = {metrics['winner']}")
        print(f"kappa           = {np.round(metrics['kappa'], 6)}")
        print(f"Delta           = {np.round(metrics['Delta'], 6)}")
        print(f"q variance      = {np.round(metrics['q_variance'], 6)}")
        print("L matrix:")
        print(np.round(metrics["L"], 6))
        print("R matrix:")
        print(np.round(metrics["R"], 6))

    print("\nSaved files in:")
    print(OUTDIR.resolve())
    print("\nKey files:")
    print("  summary.txt")
    print("  overall_summary.csv")
    print("  eta_sweep.csv")
    print("  case_A_manual_P.csv")
    print("  case_B_manual_P.csv")
    print("  case_A_point_level.csv")
    print("  case_B_point_level.csv")
    print("  case_A_source_metrics.csv")
    print("  case_B_source_metrics.csv")
    print("  case_A_overview.png")
    print("  case_B_overview.png")
    print("  case_A_experts.png")
    print("  case_B_experts.png")
    print("  case_A_q_bars.png")
    print("  case_B_q_bars.png")
    print("  eta_sweep.png")


if __name__ == "__main__":
    main()