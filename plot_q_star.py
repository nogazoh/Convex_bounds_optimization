import os
import re
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt


DATASET_MODE = "OFFICE31"
SEED = 1

RESULTS_DIR = f"./results_{DATASET_MODE}_mosek/seed_{SEED}"
SWEEP_FILE = os.path.join(RESULTS_DIR, f"Sweep_Results_{SEED}_3_sets_all_baselines.txt")
QSTAR_DIR = os.path.join(RESULTS_DIR, "saved_qstars")
OUTPUT_FIG = os.path.join(RESULTS_DIR, f"{DATASET_MODE}_qstar_source_only_histograms.png")

TARGET_PAIRS = [
    ("amazon", "dslr"),
    ("amazon", "webcam"),
    ("dslr", "webcam"),
]

STRATEGY_ORDER = ["CONFIG_ORIGINAL", "CONFIG_INVERSE", "UNIFORM"]
DOMAIN_ORDER = ["amazon", "dslr", "webcam"]


def parse_target_and_strategy(header_line: str):
    m = re.search(r"TARGET:\s*(\[[^\]]+\])\s*\|\s*STRATEGY:\s*([A-Z_]+)", header_line)
    if not m:
        return None, None
    target_raw = m.group(1)
    strategy = m.group(2)
    domains = re.findall(r"'([^']+)'", target_raw)
    return tuple(domains), strategy


def parse_ratios_line(line: str):
    m = re.search(r"RATIOS:\s*\[([^\]]+)\]", line)
    if not m:
        return None
    vals = [float(x) for x in m.group(1).split()]
    return vals


def parse_solver_result_line(line: str):
    if "acc_W:" not in line or "acc_Q:" not in line:
        return None

    parts = [p.strip() for p in line.split("|")]
    if len(parts) < 6:
        return None

    solver = parts[0]
    backend = parts[1]

    m_eps = re.search(r"m:\s*([0-9.]+)", parts[2])
    m_delta = re.search(r"d_m:\s*([0-9.]+)", parts[3])
    m_acc_w = re.search(r"acc_W:\s*([0-9.]+)", parts[4])
    m_acc_q = re.search(r"acc_Q:\s*([0-9.]+)", parts[5])

    if not (m_eps and m_delta and m_acc_w and m_acc_q):
        return None

    acc_w = float(m_acc_w.group(1))
    acc_q = float(m_acc_q.group(1))

    return {
        "solver": solver,
        "backend": backend,
        "eps_mult": float(m_eps.group(1)),
        "delta_mult": float(m_delta.group(1)),
        "acc_W": acc_w,
        "acc_Q": acc_q,
        "score": max(acc_w, acc_q),
    }


def parse_all_runs_from_sweep(filepath):
    runs = {}
    current_target = None
    current_strategy = None
    current_ratios = None

    with open(filepath, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()

            if line.startswith("TARGET:"):
                current_target, current_strategy = parse_target_and_strategy(line)
                current_ratios = None
                continue

            if line.startswith("RATIOS:"):
                current_ratios = parse_ratios_line(line)
                continue

            parsed = parse_solver_result_line(line)
            if parsed is None or current_target is None or current_strategy is None:
                continue

            parsed["ratios_full"] = current_ratios
            key = (current_target, current_strategy)
            runs.setdefault(key, []).append(parsed)

    return runs


def parse_qstar_filename(path):
    name = os.path.basename(path)
    pattern = (
        r"qstar_(?P<dataset>.+?)"
        r"__target_(?P<target>.+?)"
        r"__strategy_(?P<strategy>.+?)"
        r"__solver_(?P<solver>.+?)"
        r"__backend_(?P<backend>.+?)"
        r"__epsmult_(?P<eps>[0-9.]+)"
        r"__deltamult_(?P<delta>[0-9.]+)\.pkl$"
    )
    m = re.match(pattern, name)
    if not m:
        return None

    return {
        "path": path,
        "dataset": m.group("dataset"),
        "target": tuple(m.group("target").split("_")),
        "strategy": m.group("strategy"),
        "solver": m.group("solver"),
        "backend": m.group("backend"),
        "eps_mult": float(m.group("eps")),
        "delta_mult": float(m.group("delta")),
    }


def collect_saved_pickles():
    out = {}
    for path in glob.glob(os.path.join(QSTAR_DIR, "qstar_*.pkl")):
        parsed = parse_qstar_filename(path)
        if parsed is None:
            continue
        key = (parsed["target"], parsed["strategy"])
        out.setdefault(key, []).append(parsed)
    return out


def choose_best_available_run(all_runs, saved_pickles, target, strategy, tol=1e-9):
    key = (tuple(target), strategy)
    if key not in saved_pickles:
        return None

    available = saved_pickles[key]
    candidates = []

    for p in available:
        for r in all_runs.get(key, []):
            same_solver = (r["solver"] == p["solver"])
            same_backend = (r["backend"] == p["backend"])
            same_eps = abs(r["eps_mult"] - p["eps_mult"]) < tol
            same_delta = abs(r["delta_mult"] - p["delta_mult"]) < tol

            if same_solver and same_backend and same_eps and same_delta:
                merged = dict(r)
                merged["pickle_path"] = p["path"]
                candidates.append(merged)

    if not candidates:
        return None

    return max(candidates, key=lambda x: x["score"])


def load_q_from_pickle(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    Q = np.asarray(obj["Q"])
    if Q.ndim != 2:
        raise ValueError(f"Expected Q to be 2D, got shape {Q.shape}")
    return Q


def per_sample_source_prob(Q, source_idx=0):
    return Q[:, source_idx]


def format_pair_weights(target, ratios_full):
    if ratios_full is None:
        return "weights: N/A"

    ratio_map = {d: ratios_full[i] for i, d in enumerate(DOMAIN_ORDER)}
    return f"weights: {target[0]}={ratio_map.get(target[0], 0):.2f}, {target[1]}={ratio_map.get(target[1], 0):.2f}"

def within_item_l1_two_sources(Q):
    """
    For two-source targets:
        |q_{i,1} - q_{i,2}|
    Since rows of Q sum to 1, this measures how sharp the assignment is
    for each item:
        0   -> near 50/50
        1   -> near one-hot
    """
    if Q.shape[1] != 2:
        raise ValueError(f"Expected Q with 2 columns for a source pair, got {Q.shape}")
    return np.abs(Q[:, 0] - Q[:, 1])


def pairwise_item_l1_two_sources(Q, max_pairs=20000, seed=0):
    """
    For two-source targets:
        ||q_i - q_j||_1
    computed over sampled pairs (i,j).

    For two sources this equals:
        2 * |q_{i,1} - q_{j,1}|

    We sample pairs to avoid O(N^2) blowup.
    """
    if Q.shape[1] != 2:
        raise ValueError(f"Expected Q with 2 columns for a source pair, got {Q.shape}")

    n = len(Q)
    if n < 2:
        return np.array([])

    rng = np.random.default_rng(seed)

    num_possible = n * (n - 1) // 2
    if num_possible <= max_pairs:
        vals = []
        for i in range(n):
            diffs = np.abs(Q[i + 1:, 0] - Q[i, 0])
            vals.extend(2.0 * diffs)
        return np.asarray(vals)

    i_idx = rng.integers(0, n, size=max_pairs)
    j_idx = rng.integers(0, n, size=max_pairs)

    mask = i_idx != j_idx
    i_idx = i_idx[mask]
    j_idx = j_idx[mask]

    return 2.0 * np.abs(Q[i_idx, 0] - Q[j_idx, 0])


def make_variability_figure(all_runs, saved_pickles):
    """
    Creates two histograms per target/strategy cell:

    1) Top:    |q*_{i,1} - q*_{i,2}|   for each item i
               -> how sharp the within-item assignment is

    2) Bottom: |q*_{i,1} - q*_{j,1}|   across item pairs (i,j)
               -> variability across items for a fixed source/domain
                  (here: source 1 in the pair)

    This matches Yishay's request more explicitly:
    - within-item difference between the two domains
    - across-items difference for a fixed domain/source
    """
    def within_item_l1_two_sources(Q):
        if Q.shape[1] != 2:
            raise ValueError(f"Expected Q with 2 columns for a source pair, got {Q.shape}")
        return np.abs(Q[:, 0] - Q[:, 1])

    def pairwise_fixed_domain_diff(Q, domain_idx=0, max_pairs=20000, seed=0):
        """
        For a fixed source/domain t = domain_idx, compute:
            |q_{i,t} - q_{j,t}|
        across sampled pairs of items (i,j).
        """
        n = len(Q)
        if n < 2:
            return np.array([])

        rng = np.random.default_rng(seed)
        q_t = Q[:, domain_idx]

        num_possible = n * (n - 1) // 2
        if num_possible <= max_pairs:
            vals = []
            for i in range(n):
                diffs = np.abs(q_t[i + 1:] - q_t[i])
                vals.extend(diffs)
            return np.asarray(vals)

        i_idx = rng.integers(0, n, size=max_pairs)
        j_idx = rng.integers(0, n, size=max_pairs)

        mask = i_idx != j_idx
        i_idx = i_idx[mask]
        j_idx = j_idx[mask]

        return np.abs(q_t[i_idx] - q_t[j_idx])

    fig, axes = plt.subplots(
        nrows=len(TARGET_PAIRS) * 2,
        ncols=len(STRATEGY_ORDER),
        figsize=(16, 18),
        sharex=False,
        sharey=False
    )

    # Make axes always 2D-safe
    if len(TARGET_PAIRS) * 2 == 1 and len(STRATEGY_ORDER) == 1:
        axes = np.array([[axes]])
    elif len(STRATEGY_ORDER) == 1:
        axes = axes.reshape(len(TARGET_PAIRS) * 2, 1)
    elif len(TARGET_PAIRS) * 2 == 1:
        axes = axes.reshape(1, len(STRATEGY_ORDER))

    for pair_idx, target in enumerate(TARGET_PAIRS):
        for strat_idx, strategy in enumerate(STRATEGY_ORDER):
            ax_top = axes[2 * pair_idx, strat_idx]
            ax_bottom = axes[2 * pair_idx + 1, strat_idx]

            best = choose_best_available_run(all_runs, saved_pickles, target, strategy)

            if best is None:
                ax_top.set_title(f"{target[0]} vs {target[1]}\nNo matched saved run", fontsize=10)
                ax_top.axis("off")
                ax_bottom.axis("off")
                continue

            Q = load_q_from_pickle(best["pickle_path"])
            within_l1 = within_item_l1_two_sources(Q)
            pairwise_diff = pairwise_fixed_domain_diff(Q, domain_idx=0, max_pairs=20000, seed=0)

            # --- simple sanity print for the current panel ---
            q_dom = Q[:, 0]  # fixed domain shown in the bottom plot (source 1 / target[0])

            # "Nearly one-hot" by the top metric:
            # |q1 - q2| > 0.95  <=>  q_dom < 0.025 or q_dom > 0.975
            one_hot_mask = np.abs(Q[:, 0] - Q[:, 1]) > 0.95
            num_one_hot = int(one_hot_mask.sum())
            pct_one_hot = 100.0 * num_one_hot / len(Q)

            # Among all item pairs, how many have pairwise diff > 0.5 on this fixed domain?
            n = len(q_dom)
            num_pairs_total = n * (n - 1) // 2

            num_pairs_gt_half = 0
            for i in range(n):
                num_pairs_gt_half += np.sum(np.abs(q_dom[i + 1:] - q_dom[i]) > 0.5)

            pct_pairs_gt_half = 100.0 * num_pairs_gt_half / num_pairs_total if num_pairs_total > 0 else 0.0

            print(f"\n[{target[0]} vs {target[1]} | {strategy}]")
            print(f"N items = {n}")
            print(f"one-hot items (|q1-q2|>0.95): {num_one_hot}/{n} = {pct_one_hot:.2f}%")
            print(f"pairs with |q(i,{target[0]}) - q(j,{target[0]})| > 0.5: "
                  f"{num_pairs_gt_half}/{num_pairs_total} = {pct_pairs_gt_half:.2f}%")

            num_hot_amazon = int(np.sum(q_dom > 0.975))
            num_hot_other = int(np.sum(q_dom < 0.025))

            print(f"  of the one-hot items:")
            print(f"    near {target[0]} = 1 : {num_hot_amazon}")
            print(f"    near {target[0]} = 0 : {num_hot_other}")


            # --- Top histogram: |q_{i,1} - q_{i,2}| ---
            if len(within_l1) > 0:
                w1 = np.ones_like(within_l1) * 100.0 / len(within_l1)
                ax_top.hist(within_l1, bins=25, weights=w1)

            ax_top.set_xlim(0.0, 1.0)

            weights_str = format_pair_weights(target, best.get("ratios_full"))
            ax_top.set_title(
                f"{target[0]} vs {target[1]}\n"
                f"{weights_str}\n"
                f"|q*(i, source 1) - q*(i, source 2)|",
                fontsize=10
            )

            if len(within_l1) > 0:
                ax_top.text(
                    0.98, 0.95,
                    f"N={len(within_l1)}\nmean={within_l1.mean():.3f}\nstd={within_l1.std():.3f}",
                    transform=ax_top.transAxes,
                    ha="right",
                    va="top",
                    fontsize=9,
                    bbox=dict(boxstyle="round", alpha=0.15)
                )

            # --- Bottom histogram: |q_{i,t} - q_{j,t}| for fixed t ---
            if len(pairwise_diff) > 0:
                w2 = np.ones_like(pairwise_diff) * 100.0 / len(pairwise_diff)
                ax_bottom.hist(pairwise_diff, bins=25, weights=w2)

            ax_bottom.set_xlim(0.0, 1.0)
            ax_bottom.set_title(
                f"{target[0]} vs {target[1]}\n"
                f"|q*(i, {target[0]}) - q*(j, {target[0]})| across item pairs",
                fontsize=10
            )

            if len(pairwise_diff) > 0:
                ax_bottom.text(
                    0.98, 0.95,
                    f"N={len(pairwise_diff)}\nmean={pairwise_diff.mean():.3f}\nstd={pairwise_diff.std():.3f}",
                    transform=ax_bottom.transAxes,
                    ha="right",
                    va="top",
                    fontsize=9,
                    bbox=dict(boxstyle="round", alpha=0.15)
                )

            if strat_idx == 0:
                ax_top.set_ylabel("Percent")
                ax_bottom.set_ylabel("Percent")

            if pair_idx == len(TARGET_PAIRS) - 1:
                ax_top.set_xlabel(r"$|q^*_{i,1} - q^*_{i,2}|$")
                ax_bottom.set_xlabel(r"$|q^*_{i,1} - q^*_{j,1}|$")

    fig.suptitle(
        f"{DATASET_MODE}: q* variability within items and across items",
        fontsize=16
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out_path = os.path.join(RESULTS_DIR, f"{DATASET_MODE}_qstar_variability_histograms.png")
    fig.savefig(out_path, dpi=200)
    print(f"Saved figure to: {out_path}")
    plt.show()

def make_joint_figure(all_runs, saved_pickles):
    fig, axes = plt.subplots(
        nrows=len(TARGET_PAIRS),
        ncols=len(STRATEGY_ORDER),
        figsize=(16, 11),
        sharex=True,
        sharey=True
    )

    max_y = 0.0

    for pair_idx, target in enumerate(TARGET_PAIRS):
        for strat_idx, strategy in enumerate(STRATEGY_ORDER):
            ax = axes[pair_idx, strat_idx]
            best = choose_best_available_run(all_runs, saved_pickles, target, strategy)

            if best is None:
                ax.set_title(f"{target[0]} vs {target[1]}\nNo matched saved run", fontsize=10)
                ax.axis("off")
                continue

            Q = load_q_from_pickle(best["pickle_path"])
            src_probs = per_sample_source_prob(Q, source_idx=0)
            weights = np.ones_like(src_probs) * 100.0 / len(src_probs)

            ax.hist(src_probs, bins=25, weights=weights)
            ax.set_xlim(0.0, 1.0)

            weights_str = format_pair_weights(target, best.get("ratios_full"))
            ax.set_title(
                f"{target[0]} vs {target[1]}\n"
                f"{weights_str}\n"
                f"q* for {target[0]} | score={best['score']:.2f}",
                fontsize=10
            )

            ax.text(
                0.98, 0.95,
                f"N={len(Q)}\nmean={src_probs.mean():.3f}\nstd={src_probs.std():.3f}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=9,
                bbox=dict(boxstyle="round", alpha=0.15)
            )

            max_y = max(max_y, ax.get_ylim()[1])

            if pair_idx == len(TARGET_PAIRS) - 1:
                ax.set_xlabel(fr"$q^*_{{{target[0]}|r}}$")
            if strat_idx == 0:
                ax.set_ylabel("Percent")

    for row in axes:
        for ax in row:
            if ax.has_data():
                ax.set_ylim(0, max_y)

    fig.suptitle(
        f"{DATASET_MODE}: Per-sample q* assigned to the first source in each pair",
        fontsize=16
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUTPUT_FIG, dpi=200)
    print(f"Saved figure to: {OUTPUT_FIG}")
    plt.show()


def main():
    if not os.path.exists(SWEEP_FILE):
        raise FileNotFoundError(f"Sweep file not found: {SWEEP_FILE}")
    if not os.path.isdir(QSTAR_DIR):
        raise FileNotFoundError(f"QSTAR dir not found: {QSTAR_DIR}")

    all_runs = parse_all_runs_from_sweep(SWEEP_FILE)
    saved_pickles = collect_saved_pickles()
    make_joint_figure(all_runs, saved_pickles)
    make_variability_figure(all_runs, saved_pickles)

if __name__ == "__main__":
    main()