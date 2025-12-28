from __future__ import print_function

import time
import glob
import logging
import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity

import torch
import torch.utils.data
import torch.nn.functional as F
import matplotlib.pyplot as plt

# --- YOUR LOCAL MODULES ---
from dc import *
import classifier as ClSFR
from vae import *
import data as Data

# --- IMPORTS FOR SOLVERS ---
from cvxpy_solver import solve_convex_problem_mosek
from cvxpy_solver_per_domain import solve_convex_problem_per_domain


# =========================================================
# Global Setup
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device = ", device)

os.environ.setdefault("OMP_NUM_THREADS", "1")
torch.set_num_threads(1)

# Cache to prevent re-running same solver configs multiple times
Z_MEMORY_CACHE = {}


# =========================================================
# Helpers: target keys <-> domains
# =========================================================
LETTER2DOMAIN = {"M": "MNIST", "U": "USPS", "S": "SVHN"}
DOMAIN2LETTER = {v: k for k, v in LETTER2DOMAIN.items()}

def target_key_to_domains(target_key: str):
    """
    target_key examples: 'MU', 'MSU', 'U', ...
    Returns a sorted list of domains: ['MNIST','USPS'] etc.
    """
    letters = list(target_key.strip().upper())
    domains = [LETTER2DOMAIN[ch] for ch in letters if ch in LETTER2DOMAIN]
    # Keep deterministic order M,U,S
    order = ["MNIST", "USPS", "SVHN"]
    return [d for d in order if d in domains]

def domains_to_target_key(domains):
    order = ["MNIST", "USPS", "SVHN"]
    letters = [DOMAIN2LETTER[d] for d in order if d in domains]
    return "".join(letters).lower()  # keep your old style: m,u,s,mu,...


# =========================================================
# Data/DP builders
# =========================================================
def build_DP_model_Classes(
    data_loaders, data_size, source_domains, models, classifiers, test_path, normalize_factors, estimate_prob_type
):
    C = 10
    Y = np.zeros((data_size, C))
    D = np.zeros((data_size, C, len(source_domains)))
    H = np.zeros((data_size, C, len(source_domains)))

    i = 0
    for _domain_name, data_loader in data_loaders:
        for data, label in data_loader:
            data = data.to(device)
            N = len(data)

            y_vals = label.cpu().detach().numpy()
            one_hot = np.zeros((y_vals.size, C))
            one_hot[np.arange(y_vals.size), y_vals] = 1
            Y[i : i + N] = one_hot

            for k, source_domain in enumerate(source_domains):
                with torch.no_grad():
                    output = classifiers[source_domain](data)
                    norm_output = F.softmax(output, dim=1)
                    H[i : i + N, :, k] = norm_output.cpu().detach().numpy()

                    if estimate_prob_type == "GMSA":
                        D[i : i + N, :, k] = output.cpu().detach().numpy()

                    elif estimate_prob_type in ["OURS-STD-SCORE", "OURS-KDE", "OURS-STD-SCORE-WITH-KDE"]:
                        x_hat, _, _ = models[source_domain](data)
                        log_p = models[source_domain].compute_log_probabitility_gaussian(
                            x_hat, data.view(data.shape[0], -1), torch.zeros_like(x_hat)
                        )
                        log_p_tile = torch.tile(log_p[:, None], (1, C))
                        D[i : i + N, :, k] = log_p_tile.cpu().detach().numpy()

            i += N

    # Density post-processing per source
    for k, source_domain in enumerate(source_domains):
        if estimate_prob_type in ["GMSA", "OURS-KDE"]:
            params = {"bandwidth": np.logspace(-2, 2, 40)}
            grid = GridSearchCV(KernelDensity(), params, n_jobs=1)

            data_k = D[:, :, k]
            shuffled_indices = np.random.permutation(len(data_k))
            data_shuffle = data_k[shuffled_indices]

            grid.fit(data_shuffle[:2000])
            kde = grid.best_estimator_

            log_density = kde.score_samples(data_k)
            log_density_tile = np.tile(log_density[:, None], (1, C))
            D[:, :, k] = np.exp(log_density_tile)

        elif estimate_prob_type in ["OURS-STD-SCORE", "OURS-STD-SCORE-WITH-KDE"]:
            log_p_mean, log_p_std = normalize_factors[source_domain]
            if log_p_std == 0:
                standard_score = D[:, :, k] - log_p_mean
            else:
                standard_score = (D[:, :, k] - log_p_mean) / log_p_std

            D[:, :, k] = np.exp(-np.abs(standard_score))

            if estimate_prob_type == "OURS-STD-SCORE-WITH-KDE":
                params = {"bandwidth": np.logspace(-2, 2, 40)}
                grid = GridSearchCV(KernelDensity(), params, n_jobs=1)

                data_k = standard_score
                shuffled_indices = np.random.permutation(len(data_k))
                data_shuffle = data_k[shuffled_indices]

                grid.fit(data_shuffle[:2000])
                kde = grid.best_estimator_

                log_density = kde.score_samples(data_k)
                log_density_tile = np.tile(log_density[:, None], (1, C))
                D[:, :, k] = np.exp(log_density_tile)

        # Normalize per-sample across sources/classes in your original style
        denom = D[:, :, k].sum()
        if denom != 0:
            D[:, :, k] = D[:, :, k] / denom

    return Y, D, H


def build_DP_model(
    data_loaders, data_size, source_domains, models, classifiers, test_path, normalize_factors, estimate_prob_type
):
    C = 10
    Y = np.zeros((data_size))
    D = np.zeros((data_size, len(source_domains)))
    H = np.zeros((data_size, len(source_domains)))
    all_output = np.zeros((data_size, C, len(source_domains)))

    i = 0
    for _domain_name, data_loader in data_loaders:
        for data, label in data_loader:
            data = data.to(device)
            N = len(data)

            y_vals = label.cpu().detach().numpy()
            Y[i : i + N] = y_vals

            for k, source_domain in enumerate(source_domains):
                with torch.no_grad():
                    output = classifiers[source_domain](data)
                    norm_output = F.softmax(output, dim=1)
                    y_pred = norm_output.data.max(1, keepdim=True)[1]
                    y_pred = y_pred.flatten().cpu().detach().numpy()
                    H[i : i + N, k] = y_pred

                    if estimate_prob_type == "GMSA":
                        all_output[i : i + N, :, k] = output.cpu().detach().numpy()

                    elif estimate_prob_type in ["OURS-STD-SCORE", "OURS-KDE", "OURS-STD-SCORE-WITH-KDE"]:
                        x_hat, _, _ = models[source_domain](data)
                        log_p = models[source_domain].compute_log_probabitility_gaussian(
                            x_hat, data.view(data.shape[0], -1), torch.zeros_like(x_hat)
                        )
                        D[i : i + N, k] = log_p.cpu().detach().numpy()

            i += N

    for k, source_domain in enumerate(source_domains):
        if estimate_prob_type in ["GMSA", "OURS-KDE"]:
            if estimate_prob_type == "GMSA":
                data_k = all_output[:, :, k]
            else:
                data_k = D[:, k][:, None]

            params = {"bandwidth": np.logspace(-2, 2, 40)}
            grid = GridSearchCV(KernelDensity(), params, n_jobs=1)

            shuffled_indices = np.random.permutation(len(data_k))
            data_shuffle = data_k[shuffled_indices]

            grid.fit(data_shuffle[:2000])
            kde = grid.best_estimator_

            log_density = kde.score_samples(data_k)
            D[:, k] = np.exp(log_density)

        elif estimate_prob_type == "OURS-STD-SCORE":
            log_p_mean, log_p_std = normalize_factors[source_domain]
            if log_p_std == 0:
                standard_score = (D[:, k] - log_p_mean)
            else:
                standard_score = (D[:, k] - log_p_mean) / log_p_std
            D[:, k] = np.exp(-np.abs(standard_score))

        # normalize per source column (as you had)
        denom = D[:, k].sum()
        if denom != 0:
            D[:, k] = D[:, k] / denom

    return Y, D, H


# =========================================================
# Solver run
# =========================================================
def run_single_optimization(
    solver_name, Y, D, H, source_domains, seed, init_z_method, test_path,
    initial_delta_multiplier, fixed_epsilon_dict
):
    """
    Runs solver with AUTO-RELAXATION for DELTA (KL) ONLY.
    EPSILON (Risk) is FIXED from the CSV file and does not change.
    """
    global Z_MEMORY_CACHE

    sources_key = tuple(sorted(source_domains))
    # Cache key includes fixed eps vector to avoid stale reuse
    fixed_eps_vec_cache = tuple([fixed_epsilon_dict[d] for d in source_domains])
    cache_key = (solver_name, sources_key, seed, initial_delta_multiplier, fixed_eps_vec_cache)

    if cache_key in Z_MEMORY_CACHE:
        return Z_MEMORY_CACHE[cache_key]

    k = len(source_domains)

    # Fixed eps vector aligned to source_domains
    base_eps_vec = np.array([fixed_epsilon_dict[domain] for domain in source_domains])

    current_delta_mult = initial_delta_multiplier
    max_retries = 5
    learned_z = None
    BASE_DELTA = 0.1

    for attempt in range(max_retries):
        EPSILONS_VEC = base_eps_vec
        DELTAS_VEC = [BASE_DELTA * current_delta_mult] * k

        # For "CVXPY" general-case you used one epsilon and one delta
        SINGLE_EPS_BOUND = float(np.max(EPSILONS_VEC))
        SINGLE_DELTA_BOUND = float(np.mean(DELTAS_VEC))

        CHOSEN_BACKEND = "SCS"

        print(f"    >>> [Solver: {solver_name}] Attempt {attempt + 1}/{max_retries}")
        print(f"        Sources: {source_domains}")
        print(f"        Fixed Eps Vector: {np.round(EPSILONS_VEC, 6)}")
        print(f"        Delta Mult: {current_delta_mult:.3f} (Bound: {SINGLE_DELTA_BOUND:.6f})")

        if solver_name == "DC":
            DP = init_problem_from_model(Y, D, H, p=k, C=10)
            prob = ConvexConcaveProblem(DP)
            solver = ConvexConcaveSolver(prob, seed, init_z_method)
            z_res, _, _ = solver.solve()
            learned_z = z_res
            break

        elif solver_name == "CVXPY":
            try:
                learned_z = solve_convex_problem_mosek(
                    Y, D, H, delta=SINGLE_DELTA_BOUND, epsilon=SINGLE_EPS_BOUND, solver_type=CHOSEN_BACKEND
                )
                if not np.allclose(learned_z, np.ones(k) / k, atol=1e-3):
                    print("        -> Success!")
                    break
            except Exception as e:
                print(f"        -> Failed: {e}")

        elif solver_name == "CVXPY_PER_DOMAIN":
            try:
                learned_z = solve_convex_problem_per_domain(
                    Y, D, H, delta=DELTAS_VEC, epsilon=EPSILONS_VEC, solver_type=CHOSEN_BACKEND
                )
                if not np.allclose(learned_z, np.ones(k) / k, atol=1e-3):
                    print("        -> Success!")
                    break
            except Exception as e:
                print(f"        -> Failed: {e}")

        print("        -> Solver returned uniform/failed. Relaxing DELTA constraint...")
        current_delta_mult *= 1.20

    if learned_z is None:
        learned_z = np.ones(k) / k

    result_tuple = (learned_z, 1.0, current_delta_mult)
    Z_MEMORY_CACHE[cache_key] = result_tuple
    return result_tuple


def evaluate_z_on_data(Y, D, H, learned_z, multi_dim, source_domains):
    DP = init_problem_from_model(Y, D, H, p=len(source_domains), C=10)
    prob = ConvexConcaveProblem(DP)
    _, _, _, hz = prob.compute_DzJzKzhz(learned_z)

    if multi_dim:
        Y_true = Y.argmax(axis=1)
        hz_pred = hz.argmax(axis=1)
    else:
        Y_true = Y
        hz_pred = hz

    correct = int(np.sum(Y_true == hz_pred))
    total = int(len(Y_true))
    return correct, total


# =========================================================
# Main experiment runner: PER TARGET (different sources & weights)
# =========================================================
def run_domain_adaptation_for_target(
    target_key,
    alpha_pos, alpha_neg, vr_model_type, seed, test_path,
    estimate_prob_type, init_z_method, multi_dim,
    classifiers_all, fixed_epsilon_dict_all, delta_multiplier,
    num_train_points_per_source=1000
):
    """
    Key change:
      - For EACH target_key (e.g. 'MU'), we set source_domains = ['MNIST','USPS'] (ONLY)
      - We learn z ONLY on those sources (so z has length 2)
      - We evaluate on target mixture defined by the same target_key (union of those test sets)
      - Repeat per model and per solver => everything ends in one final table.
    """
    torch.manual_seed(seed)
    np.random.seed(seed=seed)

    logging_filename = "domain_adaptation.log"
    logging.basicConfig(filename=logging_filename, level=logging.DEBUG)

    source_domains = target_key_to_domains(target_key)  # <-- main change
    if len(source_domains) == 0:
        raise ValueError(f"Bad target_key={target_key}")

    # subset epsilons and classifiers to active sources
    fixed_epsilon_dict = {d: fixed_epsilon_dict_all[d] for d in source_domains}
    classifiers = {d: classifiers_all[d] for d in source_domains}

    # Evaluate on exactly the domains in the target_key (same list here)
    target_domains = list(source_domains)

    # Models per active source
    models = {}
    normalize_factors = {d: (0, 0) for d in source_domains}

    print(f"\n>>>> [TARGET {target_key}] [INIT] ({vr_model_type} {alpha_pos}, {alpha_neg}) sources={source_domains} <<<<")

    for domain in source_domains:
        model = vr_model(alpha_pos, alpha_neg).to(device)
        filename = f"{vr_model_type}_{alpha_pos}_{alpha_neg}_{domain}_seed1_model.pt"
        pattern = f"./models_{domain}_seed1*/{filename}"
        matches = glob.glob(pattern)
        if not matches:
            print(f"[WARN] Model weights not found for domain={domain} pattern={pattern}")
            continue

        model.load_state_dict(torch.load(max(matches, key=os.path.getmtime), map_location=torch.device(device)))
        models[domain] = model

        if estimate_prob_type in ["OURS-STD-SCORE", "OURS-STD-SCORE-WITH-KDE"]:
            train_loader, _ = Data.get_data_loaders(domain, seed=seed)
            domain_probs = torch.tensor([]).to(device)

            for data, _ in train_loader:
                with torch.no_grad():
                    data = data.to(device)
                    x_hat, mu, logstd = models[domain](data)
                    log_p = models[domain].compute_log_probabitility_gaussian(
                        x_hat, data.view(data.shape[0], -1), torch.zeros_like(x_hat)
                    )
                    domain_probs = torch.cat((log_p, domain_probs), 0)

            mean_v = domain_probs.mean().item() if domain_probs.numel() else 0.0
            std_v = domain_probs.std().item() if domain_probs.numel() else 0.0
            normalize_factors[domain] = (mean_v, std_v)

    # --------------------------
    # 1) Build optimization matrices from TRAIN of active sources
    # --------------------------
    print("  > Preparing Optimization Data Matrices...")
    data_size = 0
    data_loaders = []
    for domain in source_domains:
        train_loader, _ = Data.get_data_loaders(domain, seed=seed, num_datapoints=num_train_points_per_source)
        data_size += len(train_loader.dataset)
        data_loaders.append((domain, train_loader))

    if multi_dim:
        Y_opt, D_opt, H_opt = build_DP_model_Classes(
            data_loaders, data_size, source_domains, models, classifiers, test_path, normalize_factors, estimate_prob_type
        )
    else:
        Y_opt, D_opt, H_opt = build_DP_model(
            data_loaders, data_size, source_domains, models, classifiers, test_path, normalize_factors, estimate_prob_type
        )

    # --------------------------
    # 2) Precompute TEST matrices for each domain in target_domains
    # --------------------------
    print("  > Pre-computing Test Matrices...")
    test_cache = {}
    for domain in target_domains:
        _, test_loader = Data.get_data_loaders(domain, seed=seed)
        d_size = len(test_loader.dataset)
        d_loaders = [(domain, test_loader)]

        if multi_dim:
            Y_t, D_t, H_t = build_DP_model_Classes(
                d_loaders, d_size, source_domains, models, classifiers, test_path, normalize_factors, estimate_prob_type
            )
        else:
            Y_t, D_t, H_t = build_DP_model(
                d_loaders, d_size, source_domains, models, classifiers, test_path, normalize_factors, estimate_prob_type
            )
        test_cache[domain] = (Y_t, D_t, H_t)

    # --------------------------
    # 3) Solve z for this target_key, per solver
    # --------------------------
    solvers_to_run = ["DC", "CVXPY", "CVXPY_PER_DOMAIN"]
    results_for_this_config = []

    for solver_name in solvers_to_run:
        learned_z, _final_eps_dummy, final_delta_mult = run_single_optimization(
            solver_name=solver_name,
            Y=Y_opt, D=D_opt, H=H_opt,
            source_domains=source_domains,
            seed=seed,
            init_z_method=init_z_method,
            test_path=test_path,
            initial_delta_multiplier=delta_multiplier,
            fixed_epsilon_dict=fixed_epsilon_dict,
        )

        print(f"   > Final Z ({solver_name}) for target {target_key}: {learned_z}")

        z_str = "[" + ", ".join([f"{z:.4f}" for z in learned_z]) + "]"
        sources_str = ",".join([domains_to_target_key([d]).upper() for d in source_domains])  # e.g. "M,U"
        model_name_str = f"{vr_model_type.upper()}_{alpha_pos}_{alpha_neg}_{solver_name}_T{target_key.lower()}"

        # Evaluate on union of target domains in target_key
        total_correct = 0
        total_samples = 0
        for d in target_domains:
            Y_c, D_c, H_c = test_cache[d]
            c, t = evaluate_z_on_data(Y_c, D_c, H_c, learned_z, multi_dim, source_domains)
            total_correct += c
            total_samples += t

        score = (total_correct / total_samples) if total_samples > 0 else 0.0

        # Save small text result
        z_filename = f"Res_{solver_name}_T{target_key.lower()}_D{final_delta_mult:.2f}.txt"
        with open(os.path.join(test_path, z_filename), "w") as fp:
            fp.write(f"# TargetKey: {target_key}\n")
            fp.write(f"# Sources: {source_domains}\n")
            fp.write(f"# Solver: {solver_name}\n")
            fp.write(f"# Fixed Epsilon Dict: {fixed_epsilon_dict}\n")
            fp.write(f"# Final Delta Multiplier: {final_delta_mult}\n")
            fp.write(f"z = {z_str}\n")
            fp.write(f"score(%) = {score * 100:.4f}\n")

        results_for_this_config.append(
            {
                "Model": f"{vr_model_type.upper()}_{alpha_pos}_{alpha_neg}",
                "Solver": solver_name,
                "Target": target_key.lower(),     # columns in final pivot
                "Sources": sources_str,           # helpful debug
                "Weights": z_str,                 # varies per target
                "Score": score * 100,
                "Final Eps Mult": "Fixed",
                "Final Delta Mult": f"{final_delta_mult:.2f}",
                "Seed": seed,
                "ProbType": estimate_prob_type,
                "InitZ": init_z_method,
                "MultiDim": int(multi_dim),
            }
        )

    return results_for_this_config


# =========================================================
# Fixed epsilons loader
# =========================================================
def load_fixed_epsilons(csv_path):
    """
    Loads the cross-domain accuracy CSV and extracts the self-error (1 - acc/100)
    for each domain. Expects rows: Source, Target, Accuracy (no header).
    """
    try:
        df = pd.read_csv(csv_path, header=None, names=["Source", "Target", "Accuracy"])
        self_performance = df[df["Source"] == df["Target"]]

        epsilon_dict = {}
        for _, row in self_performance.iterrows():
            domain = row["Source"]
            acc = float(row["Accuracy"])
            error = 1.0 - (acc / 100.0)
            epsilon_dict[domain] = error

        print("Loaded Fixed Epsilons (Error Rates):", epsilon_dict)
        return epsilon_dict
    except Exception as e:
        print(f"Error loading epsilon CSV: {e}")
        return {}


# =========================================================
# Main
# =========================================================
def task_run(
    date, seed, estimate_prob_type, init_z_method, multi_dim,
    model_type, pos_alpha, neg_alpha, target_key,
    classifiers_all, fixed_epsilon_dict_all, delta_multiplier
):
    # Shorten folder names for Windows
    prob_type_short = {
        "OURS-STD-SCORE-WITH-KDE": "KDE",
        "OURS-STD-SCORE": "STD",
        "GMSA": "GMSA",
        "OURS-KDE": "OKDE",
    }.get(estimate_prob_type, "Unk")

    relative_path = (
        f"./Res_{date}_FixedEps_d{delta_multiplier}/"
        f"{prob_type_short}/"
        f"T{target_key.lower()}/"
        f"z{init_z_method}_md{int(multi_dim)}_s{seed}/"
        f"{model_type}_p{pos_alpha}_n{neg_alpha}"
    )

    abs_path = os.path.abspath(relative_path)
    if os.name == "nt" and len(abs_path) > 200 and not abs_path.startswith("\\\\?\\"):
        abs_path = "\\\\?\\" + abs_path

    os.makedirs(abs_path, exist_ok=True)

    return run_domain_adaptation_for_target(
        target_key=target_key,
        alpha_pos=pos_alpha,
        alpha_neg=neg_alpha,
        vr_model_type=model_type,
        seed=seed,
        test_path=abs_path,
        estimate_prob_type=estimate_prob_type,
        init_z_method=init_z_method,
        multi_dim=multi_dim,
        classifiers_all=classifiers_all,
        fixed_epsilon_dict_all=fixed_epsilon_dict_all,
        delta_multiplier=delta_multiplier,
    )


def main():
    print("device = ", device)

    # --- CONFIGURATION ---
    DELTA_MULTIPLIER = 1.05

    # Load fixed epsilons from CSV
    csv_path = "source_models_cross_domain_accuracy.csv"
    fixed_epsilon_dict_all = load_fixed_epsilons(csv_path)

    if not fixed_epsilon_dict_all:
        print("CRITICAL WARNING: Could not load fixed epsilons. Using fallback.")
        fixed_epsilon_dict_all = {"MNIST": 0.0051, "USPS": 0.0309, "SVHN": 0.0566}

    # Load classifiers ONCE (all three), then we will subset per target_key
    classifiers_all = {}
    all_domains = ["MNIST", "USPS", "SVHN"]
    for domain in all_domains:
        classifier = ClSFR.Grey_32_64_128_gp().to(device)
        try:
            classifier.load_state_dict(torch.load(f"./classifiers_new/{domain}_classifier.pt",
                                                  map_location=torch.device(device)))
            classifiers_all[domain] = classifier
        except FileNotFoundError:
            print(f"[WARN] Missing classifier for domain={domain}")

    # Sanity: make sure we have required epsilons + classifiers
    for d in all_domains:
        if d not in fixed_epsilon_dict_all:
            raise ValueError(f"Missing epsilon for domain={d}")
        if d not in classifiers_all:
            raise ValueError(f"Missing classifier for domain={d}")

    # Experiment grid
    estimate_prob_types = ["OURS-STD-SCORE-WITH-KDE"]
    init_z_methods = ["err"]
    multi_dim_vals = [True]
    date = "15_12"
    seeds = [10]

    alphas_by_model = {
        "vrs": [(2, -2), (2, -0.5), (0.5, -2), (0.5, -0.5)],
        "vr": [(2, -1), (0.5, -1)],
        "vae": [(1, -1)],
    }

    # TARGETS: each one runs a separate optimization with its own sources and its own z
    # (this is the core requirement you asked for)
    target_keys = [
        "MU", "MS", "US", "MSU",
        # Optional singles:
        # "M", "U", "S"
    ]

    tasks = []
    for seed in seeds:
        for estimate_prob_type in estimate_prob_types:
            for init_z_method in init_z_methods:
                for multi_dim in multi_dim_vals:
                    for model_type, pairs in alphas_by_model.items():
                        for (pos_alpha, neg_alpha) in pairs:
                            for target_key in target_keys:
                                tasks.append(
                                    (
                                        date, seed, estimate_prob_type, init_z_method, multi_dim,
                                        model_type, pos_alpha, neg_alpha, target_key,
                                        classifiers_all, fixed_epsilon_dict_all, DELTA_MULTIPLIER
                                    )
                                )

    print("\n🚀 Starting Parallel Execution (n_jobs=-1)...")
    results_lists = Parallel(n_jobs=-1, verbose=10)(
        delayed(task_run)(*t) for t in tasks
    )

    flat_results = [item for sublist in results_lists for item in sublist]
    df = pd.DataFrame(flat_results)

    if df.empty:
        print("No results collected.")
        return

    # ---- ONE final summary table that includes BOTH scores and weights per target ----
    # This produces a wide table with a MultiIndex column:
    #   ('Score', 'mu'), ('Weights','mu'), ('Score','msu'), ('Weights','msu'), ...
    final_table = df.pivot_table(
        index=["Model", "Solver", "Final Eps Mult", "Final Delta Mult", "Seed", "ProbType", "InitZ", "MultiDim"],
        columns="Target",
        values=["Score", "Weights"],
        aggfunc="first"
    )

    # Add mean score across available targets
    if ("Score" in final_table.columns.get_level_values(0)):
        score_block = final_table["Score"]
        final_table[("Score", "mean")] = score_block.mean(axis=1)

    # Round only the Score columns
    for col in final_table.columns:
        if col[0] == "Score":
            final_table[col] = final_table[col].astype(float).round(2)

    base_name = f"final_summary_results_{date}_PerTargetZ_FixedEps_delta_{DELTA_MULTIPLIER}"
    extension = ".csv"
    version = 1
    while True:
        output_csv_name = f"{base_name}_v{version}{extension}"
        if not os.path.exists(output_csv_name):
            break
        version += 1

    final_table.to_csv(output_csv_name)
    print(f"\n✅ Saved to: {output_csv_name}")
    print(final_table)


if __name__ == "__main__":
    main()
