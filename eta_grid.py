# eta_eps_grid_compare_331_333_incremental.py

import os
import gc
import io
import json
import traceback
import contextlib
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
# import cvxpy_diag
import cvxpy_3_31_kkt
import cvxpy_3_33_kkt

import msa_all_summer as msa


# ============================================================
# CONFIG
# ============================================================
DATASET_MODE = "OFFICE224"
SEED = 1

PAIRS_TO_RUN = [
    ["Clipart", "Product"],
    ["Clipart", "Real World"],
]

# ============================================================
# EPSILON MODE
# ============================================================
# Options:
#   "grid_source_error"  -> old behavior: epsilon from SOURCE_ERRORS * eps_mult
#   "oracle_error"       -> new behavior: epsilon = oracle error
#
# oracle error = 1 - oracle_accuracy
# where oracle_accuracy is computed using the usual ORACLE weights
EPSILON_MODE = "oracle_error"

# Used only when EPSILON_MODE == "grid_source_error"
EPS_MULT_GRID = [1.0, 1.2, 1.5, 2.0, 3.0, 5.0]

# Used only when EPSILON_MODE == "oracle_error"
# keep this as 1.0 if you want epsilon = exact oracle error
ORACLE_EPS_MULT = 1.0

ETA_GRID = [
    1e-8, 3e-8,
    1e-7, 3e-7,
    1e-6, 3e-6,
    1e-5, 3e-5,
    1e-4, 3e-4,
    1e-3, 3e-3,
    1e-2, 3e-2,
    1e-1, 3e-1,
    1.0, 3.0, 10.0
]

SOLVERS_TO_RUN = ["3.31", "3.33"]  # or ["P1-DIAG"]
BACKENDS = ["SCS"]   # can extend to ["SCS", "MOSEK"]

Q_MIN = 1e-12
W_MIN = 1e-12
SCS_EPS = 1e-4
SCS_MAX_ITERS = 20000
NORMALIZE_D = True

DC_NUM_RESTARTS = 5
COMPARE_TO_DC_USING_W = True

OUTPUT_DIR = f"./eta_eps_compare_kkt_{DATASET_MODE}_seed_{SEED}"

P1_DIAG_M = 1.0
P1_DIAG_DELTA_MODE = "oracle_error"
P1_DIAG_DELTA_MULT = 1.0

# ============================================================
# KKT CONFIG
# ============================================================
RETURN_KKT_DETAILS = True
KKT_TOL = 1e-8


# ============================================================
# Helpers
# ============================================================
def configure_msa_dataset_mode(msa_module, dataset_mode: str):
    msa_module.DATASET_MODE = dataset_mode
    msa_module.CURRENT_CFG = msa_module.CONFIGS[dataset_mode]
    msa_module.SOURCE_ERRORS = msa_module.CURRENT_CFG["SOURCE_ERRORS"]
    msa_module.TEST_SET_SIZES = msa_module.CURRENT_CFG["TEST_SET_SIZES"]
    msa_module.ALL_DOMAINS_LIST = msa_module.CURRENT_CFG["DOMAINS"]
    msa_module.NUM_CLASSES = msa_module.CURRENT_CFG["CLASSES"]
    msa_module.INPUT_DIM = msa_module.CURRENT_CFG["INPUT_DIM"]
    msa_module.D_PRECOMP_PATH = msa_module.CURRENT_CFG["D_PRECOMP_PATH"]


def load_classifiers_for_domains(msa_module, domains):
    device = msa_module.device
    n_gpus = torch.cuda.device_count()
    classifiers = {}

    for d in domains:
        p1 = f"./classifiers/{d}_224.pt"
        p2 = f"./classifiers/{d}_classifier.pt"
        path = p1 if os.path.exists(p1) else p2

        if not os.path.exists(path):
            raise FileNotFoundError(f"Classifier for domain '{d}' not found. Tried: {p1}, {p2}")

        if msa_module.DATASET_MODE == "DIGITS":
            m = msa_module.ClSFR.Grey_32_64_128_gp()
            m.load_state_dict(torch.load(path, map_location=device))
            if n_gpus > 1:
                m = nn.DataParallel(m)
            classifiers[d] = m.to(device).eval()
        else:
            m = models.resnet50(weights=None)
            m.fc = nn.Linear(m.fc.in_features, msa_module.NUM_CLASSES)
            m.load_state_dict(torch.load(path, map_location=device))
            extractor = msa_module.FeatureExtractor(m)
            if n_gpus > 1:
                extractor = nn.DataParallel(extractor)
            classifiers[d] = extractor.to(device).eval()

        print(f"✅ Loaded classifier: {d}")

    return classifiers


def get_weight_sets_for_pair(msa_module, pair):
    key = tuple(sorted(pair))
    if key not in msa_module.TARGET_RATIOS_CONFIG:
        raise KeyError(f"Pair {key} not found in TARGET_RATIOS_CONFIG")

    set1 = msa_module.TARGET_RATIOS_CONFIG[key]

    keys = list(set1.keys())
    vals = np.array([set1[k] for k in keys], dtype=float)
    inv_vals = 1.0 / (vals + 1e-6)
    inv_vals /= inv_vals.sum()
    set2 = {keys[i]: float(inv_vals[i]) for i in range(len(keys))}

    set3 = {d: 1.0 / len(pair) for d in pair}

    return [
        ("CONFIG_ORIGINAL", set1),
        ("CONFIG_INVERSE", set2),
        ("UNIFORM", set3),
    ]


def pretty_weights_full(w, source_domains, all_source_domains):
    if w is None:
        return "None"
    w_f = msa.map_weights_to_full_source_list(np.asarray(w), source_domains, all_source_domains)
    return np.array2string(np.asarray(w_f), precision=6)


def run_dc_once(msa_module, Y, D, H, seed):
    dp = msa_module.init_problem_from_model_fast(
        Y, D, H, p=D.shape[2], C=msa_module.NUM_CLASSES
    )
    slv = msa_module.ConvexConcaveSolverFast(
        msa_module.ConvexConcaveProblemFast(dp),
        seed,
        "err"
    )
    z_dc, _, _ = slv.solve()
    return z_dc


def run_dc_best_of_restarts(msa_module, Y, D, H, base_seed, n_restarts=5):
    best = {
        "z": None,
        "acc": -np.inf,
        "all_accs": [],
        "failures": [],
    }

    for i in range(n_restarts):
        try:
            z_dc = run_dc_once(msa_module, Y, D, H, base_seed + 100 * i)
            if z_dc is None:
                best["failures"].append(f"restart {i}: returned None")
                continue

            acc_dc = msa_module.evaluate_accuracy_wd(z_dc, D, H, Y)
            best["all_accs"].append(float(acc_dc))

            if acc_dc > best["acc"]:
                best["acc"] = float(acc_dc)
                best["z"] = np.asarray(z_dc).copy()

        except Exception as e:
            best["failures"].append(f"restart {i}: {repr(e)}")

    if best["z"] is None:
        raise RuntimeError(f"DC failed in all restarts. Failures: {best['failures']}")

    best["mean_acc"] = float(np.mean(best["all_accs"]))
    best["std_acc"] = float(np.std(best["all_accs"]))
    return best


def compute_oracle_accuracy_and_error(Y, D, H, oracle_weights):
    """
    oracle_weights are the usual ORACLE weights for the current pair/strategy.
    Returns:
        oracle_acc_percent : accuracy in percent
        oracle_error_rate  : error in [0,1]
    """
    oracle_acc_percent = float(msa.evaluate_accuracy_wd(oracle_weights, D, H, Y))
    oracle_error_rate = 1.0 - (oracle_acc_percent / 100.0)
    return oracle_acc_percent, oracle_error_rate


def compute_epsilons_from_source_errors(msa_module, source_domains, eps_mult):
    base_errors = np.array([
        msa_module.SOURCE_ERRORS.get(d, 0.1) + 0.05
        for d in source_domains
    ], dtype=float)

    errors = base_errors * eps_mult
    epsilon = float(np.max(errors))
    return {
        "epsilon_mode": "grid_source_error",
        "base_errors": base_errors.tolist(),
        "errors": errors.tolist(),
        "epsilon": epsilon,
        "eps_mult": float(eps_mult),
        "oracle_acc_percent": None,
        "oracle_error_rate": None,
    }


def compute_epsilon_from_oracle(Y, D, H, oracle_weights, oracle_eps_mult=1.0):
    oracle_acc_percent, oracle_error_rate = compute_oracle_accuracy_and_error(
        Y=Y,
        D=D,
        H=H,
        oracle_weights=oracle_weights
    )
    epsilon = float(oracle_error_rate * oracle_eps_mult)

    return {
        "epsilon_mode": "oracle_error",
        "base_errors": None,
        "errors": None,
        "epsilon": epsilon,
        "eps_mult": float(oracle_eps_mult),
        "oracle_acc_percent": float(oracle_acc_percent),
        "oracle_error_rate": float(oracle_error_rate),
    }


def run_baselines_local(Y, D, H, source_domains, all_source_domains, seed, true_r_weights):
    """
    Very close to the original run_baselines in msa_all_summer.
    """
    buf = io.StringIO()

    # ORACLE and UNIFORM
    uniform_w = np.ones(len(source_domains)) / len(source_domains)
    for name, w in [("ORACLE", true_r_weights), ("UNIFORM", uniform_w)]:
        acc = msa.evaluate_accuracy_wd(w, D, H, Y)
        w_f = msa.map_weights_to_full_source_list(w, source_domains, all_source_domains)
        buf.write(f"{name:<18} | {'N/A':<15} | {'N/A':<15} | {acc:<12.2f} | {str(np.round(w_f, 4))}\n")

    # BEST_SINGLE_SRC
    y_true = Y.argmax(axis=1)
    best_single_acc = -1.0
    best_source_name = ""
    best_source_idx = -1

    for k, src_name in enumerate(source_domains):
        src_preds = H[:, :, k].argmax(axis=1)
        src_acc = (src_preds == y_true).mean() * 100.0

        if src_acc > best_single_acc:
            best_single_acc = src_acc
            best_source_name = src_name
            best_source_idx = k

    w_best_single = np.zeros(len(source_domains))
    w_best_single[best_source_idx] = 1.0
    w_f_best = msa.map_weights_to_full_source_list(w_best_single, source_domains, all_source_domains)

    buf.write(
        f"{'BEST_SINGLE_SRC':<18} | {best_source_name:<15} | {'N/A':<15} | {best_single_acc:<12.2f} | {str(np.round(w_f_best, 4))}\n"
    )

    # ORACLE_ANY_CORRECT
    pred_per_source = H.argmax(axis=1)
    any_correct = (pred_per_source == y_true[:, None]).any(axis=1)
    oracle_any_acc = any_correct.mean() * 100.0
    buf.write(f"{'ORACLE_ANY_CORRECT':<18} | {'N/A':<15} | {'N/A':<15} | {oracle_any_acc:<12.2f} | N/A\n")

    # DC (5-Seeds)
    dc_accuracies = []
    best_z_dc = None
    for i in range(5):
        try:
            z_dc = run_dc_once(msa, Y, D, H, seed + (i * 100))
            if z_dc is None:
                continue
            acc = msa.evaluate_accuracy_wd(z_dc, D, H, Y)
            dc_accuracies.append(acc)
            if best_z_dc is None or acc >= max(dc_accuracies):
                best_z_dc = z_dc
        except Exception:
            continue

    if dc_accuracies:
        avg_res = f"{np.mean(dc_accuracies):.2f}±{np.std(dc_accuracies):.2f}"
        w_f = msa.map_weights_to_full_source_list(best_z_dc, source_domains, all_source_domains)
        buf.write(f"{'DC (5-Seeds)':<18} | {'N/A':<15} | {'N/A':<15} | {avg_res:<12} | {str(np.round(w_f, 4))}\n")

    return buf.getvalue()


def solve_one(solver_name, backend, Y, D, H, epsilon, eta):
    """
    - 3.31 from cvxpy_3_31_kkt
    - 3.33 from cvxpy_3_33_kkt
    - suppress internal prints when desired
    """
    if solver_name == "3.31":
        return cvxpy_3_31_kkt.solve_convex_problem_smoothed_kl_331(
            Y=Y, D=D, H=H,
            epsilon=epsilon,
            eta=eta,
            solver_type=backend,
            q_min=Q_MIN,
            w_min=W_MIN,
            scs_eps=SCS_EPS,
            scs_max_iters=SCS_MAX_ITERS,
            normalize_D=NORMALIZE_D,
            return_kkt_details=RETURN_KKT_DETAILS,
            kkt_tol=KKT_TOL,
        )

    elif solver_name == "3.33":
        sink = io.StringIO()
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            return cvxpy_3_33_kkt.solve_convex_problem_smoothed_original_p_333(
                Y=Y, D=D, H=H,
                epsilon=epsilon,
                eta=eta,
                solver_type=backend,
                q_min=Q_MIN,
                w_min=W_MIN,
                scs_eps=SCS_EPS,
                scs_max_iters=SCS_MAX_ITERS,
                return_kkt_details=RETURN_KKT_DETAILS,
                kkt_tol=KKT_TOL,
            )

    elif solver_name == "P1-DIAG":
        delta = float(epsilon)
        sink = io.StringIO()
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            return cvxpy_diag.solve_convex_problem_p1_diag(
                D=D,
                delta=delta,
                M=P1_DIAG_M,
                eta=eta,
                solver_type=backend,
                q_min=Q_MIN,
                w_min=W_MIN,
                scs_eps=SCS_EPS,
                scs_max_iters=SCS_MAX_ITERS,
            )

    else:
        raise ValueError(f"Unsupported solver: {solver_name}")


def run_solver_grid(Y, D, H, source_domains, oracle_weights):
    results = []

    if EPSILON_MODE == "grid_source_error":
        epsilon_specs = [
            compute_epsilons_from_source_errors(msa, source_domains, eps_mult)
            for eps_mult in EPS_MULT_GRID
        ]
    elif EPSILON_MODE == "oracle_error":
        epsilon_specs = [
            compute_epsilon_from_oracle(
                Y=Y,
                D=D,
                H=H,
                oracle_weights=oracle_weights,
                oracle_eps_mult=ORACLE_EPS_MULT,
            )
        ]
    else:
        raise ValueError(f"Unsupported EPSILON_MODE: {EPSILON_MODE}")

    for solver_name in SOLVERS_TO_RUN:
        for backend in BACKENDS:
            for eps_spec in epsilon_specs:
                epsilon = eps_spec["epsilon"]

                for eta in ETA_GRID:
                    row = {
                        "solver": solver_name,
                        "backend": backend,
                        "epsilon_mode": eps_spec["epsilon_mode"],
                        "eps_mult": eps_spec["eps_mult"],
                        "base_errors": eps_spec["base_errors"],
                        "errors": eps_spec["errors"],
                        "epsilon": float(epsilon),
                        "oracle_acc_percent": eps_spec["oracle_acc_percent"],
                        "oracle_error_rate": eps_spec["oracle_error_rate"],
                        "eta": float(eta),
                        "status": None,
                        "acc_w": None,
                        "acc_q": None,
                        "w": None,
                        "exception": None,
                        "kkt_details": None,
                    }

                    try:
                        solve_out = solve_one(
                            solver_name=solver_name,
                            backend=backend,
                            Y=Y, D=D, H=H,
                            epsilon=epsilon,
                            eta=eta,
                        )

                        if RETURN_KKT_DETAILS and solver_name in ["3.31", "3.33"]:
                            w, Q, kkt_details = solve_out
                        else:
                            w, Q = solve_out
                            kkt_details = None

                        if w is None or Q is None:
                            row["status"] = "returned_none"
                        else:
                            row["status"] = "ok"
                            row["acc_w"] = float(msa.evaluate_accuracy_wd(w, D, H, Y))
                            row["acc_q"] = float(msa.evaluate_accuracy_q(Q, H, Y))
                            row["w"] = np.asarray(w).tolist()
                            row["kkt_details"] = kkt_details

                    except Exception:
                        row["status"] = "exception"
                        row["exception"] = traceback.format_exc()

                    results.append(row)

    return results


def select_summary_rows(results, dc_acc):
    key_name = "acc_w" if COMPARE_TO_DC_USING_W else "acc_q"

    by_solver = {}
    for solver_name in SOLVERS_TO_RUN:
        valid = [r for r in results if r["solver"] == solver_name and r["status"] == "ok"]

        if not valid:
            by_solver[solver_name] = {
                "closest_to_dc": None,
                "best_overall": None,
            }
        else:
            by_solver[solver_name] = {
                "closest_to_dc": min(valid, key=lambda r: abs(r[key_name] - dc_acc)),
                "best_overall": max(valid, key=lambda r: r[key_name]),
            }
    return by_solver


def format_mu_zero_lists(kkt_details):
    """
    Return explicit lists of which mu_t are zero / non-zero.
    """
    if kkt_details is None or "mu_is_zero" not in kkt_details:
        return "", "", ""

    mu_is_zero = np.array(kkt_details["mu_is_zero"], dtype=bool)
    zero_idx = np.where(mu_is_zero)[0].tolist()
    nonzero_idx = np.where(~mu_is_zero)[0].tolist()

    zero_text = f"mu_zero_idx={zero_idx}"
    nonzero_text = f"mu_nonzero_idx={nonzero_idx}"
    full_text = f"{zero_text} | {nonzero_text}"
    return zero_text, nonzero_text, full_text


def format_original_style_block(pair, strategy_name, true_r_full, baseline_block, grid_results):
    """
    Styled to be as close as possible to the original log.
    """
    buf = io.StringIO()

    buf.write(f"\n{'=' * 100}\n")
    buf.write(f"TARGET: {pair} | STRATEGY: {strategy_name}\n")
    buf.write(f"RATIOS: {np.round(true_r_full, 4)}\n")
    buf.write(f"{'=' * 100}\n")

    buf.write(baseline_block)

    for r in grid_results:
        eps_desc = (
            f"oracle_eps:{r['epsilon']:.6f}"
            if r["epsilon_mode"] == "oracle_error"
            else f"m:{r['eps_mult']}"
        )

        if r["status"] == "returned_none":
            buf.write(
                f"[{r['solver']}/{r['backend']}] Returned None (likely infeasible) | "
                f"{eps_desc} | eta:{r['eta']}\n"
            )
        elif r["status"] == "exception":
            short_exc = repr(r["exception"][:220])
            buf.write(
                f"[{r['solver']}/{r['backend']}] ERROR {eps_desc} | eta:{r['eta']} | {short_exc}\n"
            )
        else:
            extra = ""
            if r["epsilon_mode"] == "oracle_error":
                extra += (
                    f" | oracle_acc:{r['oracle_acc_percent']:.2f}"
                    f" | oracle_err:{r['oracle_error_rate']:.6f}"
                )

            if r.get("kkt_details") is not None:
                kd = r["kkt_details"]
                mu = np.array(kd["mu"], dtype=float)
                gamma = np.array(kd["gamma"], dtype=float)
                mu_zero = np.array(kd["mu_is_zero"], dtype=bool)
                gamma_zero = np.array(kd["gamma_is_zero"], dtype=bool)

                _, _, mu_zero_lists_text = format_mu_zero_lists(kd)

                extra += (
                    f" | mu={np.round(mu, 6).tolist()}"
                    f" | mu_zero={mu_zero.tolist()}"
                    f" | {mu_zero_lists_text}"
                    f" | gamma_zero_count={int(gamma_zero.sum())}/{len(gamma_zero)}"
                    f" | gamma_min={gamma.min():.6e}"
                    f" | gamma_max={gamma.max():.6e}"
                )

            buf.write(
                f"{r['solver']:<12} | {r['backend']:<5} | "
                f"{eps_desc:<18} | eta:{r['eta']:<10} | "
                f"acc_W: {r['acc_w']:>6.2f} | "
                f"acc_Q: {r['acc_q']:>6.2f} | "
                f"W: {pretty_weights_full(r['w'], pair, msa.ALL_DOMAINS_LIST)}"
                f"{extra}\n"
            )

    return buf.getvalue()


def format_incremental_summary(summary_obj):
    """
    Compact summary for one completed pair+strategy.
    """
    buf = io.StringIO()
    compare_key = "acc_w" if COMPARE_TO_DC_USING_W else "acc_q"

    s = summary_obj
    buf.write("\n" + "=" * 100 + "\n")
    buf.write(f"PAIR     : {s['pair']}\n")
    buf.write(f"STRATEGY : {s['strategy']}\n")
    buf.write(f"RATIOS   : {s['ratios']}\n")
    buf.write(f"DC BEST  : {s['dc_best_acc']:.4f}\n")
    buf.write(f"DC Z     : {s['dc_best_z']}\n")
    buf.write(f"DC MEAN±STD: {s['dc_mean_acc']:.4f} ± {s['dc_std_acc']:.4f}\n")

    for solver_name in SOLVERS_TO_RUN:
        solver_rows = [r for r in s["grid_results"] if r["solver"] == solver_name]
        counts = {}
        for r in solver_rows:
            counts[r["status"]] = counts.get(r["status"], 0) + 1
        buf.write(f"{solver_name} STATUS COUNTS: {counts}\n")

        summary = s["solver_summaries"][solver_name]
        if summary["closest_to_dc"] is None:
            buf.write(f"{solver_name}: No feasible combination found.\n")
        else:
            c = summary["closest_to_dc"]
            b = summary["best_overall"]

            eps_desc_c = (
                f"oracle_eps={c['epsilon']:.6f}"
                if c["epsilon_mode"] == "oracle_error"
                else f"eps_mult={c['eps_mult']}"
            )
            eps_desc_b = (
                f"oracle_eps={b['epsilon']:.6f}"
                if b["epsilon_mode"] == "oracle_error"
                else f"eps_mult={b['eps_mult']}"
            )

            if c["epsilon_mode"] == "oracle_error":
                eps_desc_c += (
                    f" | oracle_acc={c['oracle_acc_percent']:.2f}"
                    f" | oracle_err={c['oracle_error_rate']:.6f}"
                )
            if b["epsilon_mode"] == "oracle_error":
                eps_desc_b += (
                    f" | oracle_acc={b['oracle_acc_percent']:.2f}"
                    f" | oracle_err={b['oracle_error_rate']:.6f}"
                )

            c_kkt = c.get("kkt_details")
            b_kkt = b.get("kkt_details")

            c_mu_extra = ""
            b_mu_extra = ""

            if c_kkt is not None:
                _, _, c_mu_lists_text = format_mu_zero_lists(c_kkt)
                c_mu_extra = f" | {c_mu_lists_text}"
            if b_kkt is not None:
                _, _, b_mu_lists_text = format_mu_zero_lists(b_kkt)
                b_mu_extra = f" | {b_mu_lists_text}"

            buf.write(
                f"{solver_name} CLOSEST TO DC: {eps_desc_c} | eta={c['eta']} | "
                f"acc_w={c['acc_w']:.4f} | acc_q={c['acc_q']:.4f} | "
                f"gap_to_dc={abs(c[compare_key] - s['dc_best_acc']):.4f} | w={c['w']}"
                f"{c_mu_extra}\n"
            )
            buf.write(
                f"{solver_name} BEST OVERALL : {eps_desc_b} | eta={b['eta']} | "
                f"acc_w={b['acc_w']:.4f} | acc_q={b['acc_q']:.4f} | w={b['w']}"
                f"{b_mu_extra}\n"
            )

    return buf.getvalue()


def append_jsonl(path, obj):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")
        f.flush()


# ============================================================
# Main
# ============================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    configure_msa_dataset_mode(msa, DATASET_MODE)

    needed_domains = sorted(set(sum(PAIRS_TO_RUN, [])))
    classifiers = load_classifiers_for_domains(msa, needed_domains)

    if msa.USE_PRECOMPUTED_D and os.path.exists(msa.D_PRECOMP_PATH):
        Global_D = msa.load_global_D_matrix(msa.D_PRECOMP_PATH)
        domain_lengths = msa.compute_domain_lengths(msa.ALL_DOMAINS_LIST)
        print("✅ Loaded precomputed Global_D")
    else:
        raise RuntimeError("This script expects precomputed D to exist.")

    full_log_path = os.path.join(OUTPUT_DIR, "full_results_original_style.txt")
    summary_log_path = os.path.join(OUTPUT_DIR, "summary_results_incremental.txt")
    jsonl_path = os.path.join(OUTPUT_DIR, "results_incremental.jsonl")

    # header for summary file
    with open(summary_log_path, "w", encoding="utf-8") as f:
        f.write(f"DATASET_MODE = {DATASET_MODE}\n")
        f.write(f"SEED = {SEED}\n")
        f.write(f"ETA_GRID = {ETA_GRID}\n")
        f.write(f"EPSILON_MODE = {EPSILON_MODE}\n")
        if EPSILON_MODE == "grid_source_error":
            f.write(f"EPS_MULT_GRID = {EPS_MULT_GRID}\n")
        elif EPSILON_MODE == "oracle_error":
            f.write(f"ORACLE_EPS_MULT = {ORACLE_EPS_MULT}\n")
        f.write(f"RETURN_KKT_DETAILS = {RETURN_KKT_DETAILS}\n")
        f.write(f"KKT_TOL = {KKT_TOL}\n")
        f.write(f"COMPARE_TO_DC_USING = {'acc_w' if COMPARE_TO_DC_USING_W else 'acc_q'}\n")
        f.flush()

    # clear/create full log
    with open(full_log_path, "w", encoding="utf-8") as f:
        f.write("")
        f.flush()

    # clear/create jsonl
    with open(jsonl_path, "w", encoding="utf-8") as f:
        f.write("")
        f.flush()

    for pair in PAIRS_TO_RUN:
        print("\n" + "=" * 100)
        print(f"PAIR: {pair}")
        print("=" * 100)

        Y_full, D_full, H_full = msa.build_YDH_with_precomputed_D(
            target_domains=pair,
            seed=SEED,
            classifiers=classifiers,
            Global_D=Global_D,
            domain_lengths=domain_lengths,
        )

        weight_sets = get_weight_sets_for_pair(msa, pair)

        for strategy_name, custom_ratios in weight_sets:
            print(f"\n--- {pair} | {strategy_name} | ratios={custom_ratios}")

            Y, D, H = msa.apply_custom_ratios(Y_full, D_full, H_full, pair, custom_ratios)

            true_r_weights = np.array([custom_ratios[d] for d in pair])
            true_r_full = msa.map_weights_to_full_source_list(
                true_r_weights, pair, msa.ALL_DOMAINS_LIST
            )

            # baselines block in original style
            baseline_block = run_baselines_local(
                Y=Y,
                D=D,
                H=H,
                source_domains=pair,
                all_source_domains=msa.ALL_DOMAINS_LIST,
                seed=SEED,
                true_r_weights=true_r_weights,
            )

            dc_best = run_dc_best_of_restarts(
                msa_module=msa,
                Y=Y,
                D=D,
                H=H,
                base_seed=SEED,
                n_restarts=DC_NUM_RESTARTS,
            )

            grid_results = run_solver_grid(
                Y=Y,
                D=D,
                H=H,
                source_domains=pair,
                oracle_weights=true_r_weights,
            )
            solver_summaries = select_summary_rows(grid_results, dc_best["acc"])

            strategy_result = {
                "pair": pair,
                "strategy": strategy_name,
                "ratios": custom_ratios,
                "dc_best_acc": dc_best["acc"],
                "dc_best_z": dc_best["z"].tolist(),
                "dc_mean_acc": dc_best["mean_acc"],
                "dc_std_acc": dc_best["std_acc"],
                "solver_summaries": solver_summaries,
                "grid_results": grid_results,
            }

            original_style_block = format_original_style_block(
                pair=pair,
                strategy_name=strategy_name,
                true_r_full=true_r_full,
                baseline_block=baseline_block,
                grid_results=grid_results,
            )

            summary_block = format_incremental_summary(strategy_result)

            # write immediately after each finished strategy
            with open(full_log_path, "a", encoding="utf-8") as f:
                f.write(original_style_block)
                f.flush()

            with open(summary_log_path, "a", encoding="utf-8") as f:
                f.write(summary_block)
                f.flush()

            append_jsonl(jsonl_path, strategy_result)

            print(f"✅ Saved logs for pair={pair}, strategy={strategy_name}")

            del Y, D, H
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        del Y_full, D_full, H_full
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n✅ Done.")
    print(f"Saved full original-style log to: {full_log_path}")
    print(f"Saved incremental summary     to: {summary_log_path}")
    print(f"Saved incremental JSONL      to: {jsonl_path}")


if __name__ == "__main__":
    main()