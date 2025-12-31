from __future__ import print_function
import time
import torch.utils.data
import logging
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
import os
from joblib import Parallel, delayed
import torch
import torch.nn.functional as F
import glob
import sys

# --- YOUR MODULES ---
from dc import *
import classifier as ClSFR
import matplotlib.pyplot as plt
from vae import *
import data as Data

# --- SOLVERS IMPORTS ---
try:
    from cvxpy_solver import solve_convex_problem_mosek
    from cvxpy_solver_per_domain import solve_convex_problem_per_domain
except ImportError:
    print("[WARNING] CVXPY solvers not found. Make sure cvxpy_solver.py exists.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device = ", device)

# Limit to one THREAD per process to prevent collisions in Parallel execution
os.environ.setdefault("OMP_NUM_THREADS", "1")
torch.set_num_threads(1)

# --- GLOBAL CONFIG FOR ERRORS (EPSILONS) ---
SOURCE_ERRORS = {
    'MNIST': 1.0 - 0.9948,
    'USPS': 1.0 - 0.972596,
    'SVHN': 1.0 - 0.949716
}

# --- TRUE TEST SET SIZES (For Oracle Calculation) ---
TEST_SET_SIZES = {
    'MNIST': 10000,
    'USPS': 2007,
    'SVHN': 26032
}


# --- HELPER FOR PRINTING WITH PID ---
def log_print(*args, **kwargs):
    """Prints messages prefixed with the Process ID to track parallel execution."""
    pid = os.getpid()
    print(f"[PID {pid}]", *args, **kwargs)
    sys.stdout.flush()


# --- HELPER FOR MAPPING WEIGHTS ---
def map_weights_to_full_source_list(subset_weights, subset_sources, full_source_list):
    """
    Helper function to map a smaller weight vector (e.g. size 2) back to the full list (size 3).
    Example: subset_weights=[0.8, 0.2], subset_sources=['M', 'U'], full=['M', 'U', 'S']
    Result -> [0.8, 0.2, 0.0]
    """
    full_weights = np.zeros(len(full_source_list))
    for i, source in enumerate(full_source_list):
        if source in subset_sources:
            subset_idx = subset_sources.index(source)
            full_weights[i] = subset_weights[subset_idx]
        else:
            full_weights[i] = 0.0
    return full_weights


def get_oracle_weights(target_domains, source_domains, mode="real_ratio"):
    """
    Calculates the TRUE ratio.
    """
    if mode == "balanced":
        weights = []
        active_domains = len(target_domains)
        for source in source_domains:
            if source in target_domains:
                weights.append(1.0 / active_domains)
            else:
                weights.append(0.0)
        return np.array(weights)

    # Default: Calculate based on Test Set Sizes
    weights = []
    total_samples = 0
    for domain in target_domains:
        if domain in TEST_SET_SIZES:
            total_samples += TEST_SET_SIZES[domain]

    for source in source_domains:
        if source in target_domains:
            w = TEST_SET_SIZES[source] / total_samples
        else:
            w = 0.0
        weights.append(w)

    return np.array(weights)


def create_loaders(domains, seed, strategy):
    """
    Creates data loaders based on the strategy.
    """
    loaders = []
    total_size = 0

    for domain in domains:
        # Always fetch fresh loader
        _, full_loader = Data.get_data_loaders(domain, seed=seed)

        if strategy == "balanced_1000":
            # Subset to 1000 samples
            indices = list(range(min(1000, len(full_loader.dataset))))
            subset = torch.utils.data.Subset(full_loader.dataset, indices)
            loader = torch.utils.data.DataLoader(subset, batch_size=128, shuffle=False)
            loaders.append((domain, loader))
            total_size += len(indices)
        else:
            # Use full data
            loaders.append((domain, full_loader))
            total_size += len(full_loader.dataset)

    return loaders, total_size


def build_DP_model_Classes(data_loaders, data_size, source_domains, models, classifiers,
                           normalize_factors, estimate_prob_type):
    """
    Target-Optimized Matrix Construction
    """
    log_print(f"--- Starting build_DP_model_Classes (Size: {data_size}) ---")

    C = 10
    Y = np.zeros((data_size, C))
    D = np.zeros((data_size, C, len(source_domains)))
    H = np.zeros((data_size, C, len(source_domains)))
    i = 0

    # PART 1: Data Loading & Raw Score Calculation
    for target_domain, data_loader in data_loaders:
        for batch_idx, (data, label) in enumerate(data_loader):
            data = data.to(device)
            N = len(data)

            # Ground Truth
            y_vals = label.cpu().detach().numpy()
            one_hot = np.zeros((y_vals.size, C))
            one_hot[np.arange(y_vals.size), y_vals] = 1
            Y[i:i + N] = one_hot

            for k, source_domain in enumerate(source_domains):
                with torch.no_grad():
                    # H(x)
                    output = classifiers[source_domain](data)
                    norm_output = F.softmax(output, dim=1)
                    H[i:i + N, :, k] = norm_output.cpu().detach().numpy()

                    # D(x)
                    if estimate_prob_type == "GMSA":
                        D[i:i + N, :, k] = output.cpu().detach().numpy()

                    elif estimate_prob_type in ["OURS-STD-SCORE", "OURS-KDE", "OURS-STD-SCORE-WITH-KDE"]:
                        x_hat, _, _ = models[source_domain](data)
                        log_p = models[source_domain].compute_log_probabitility_gaussian(
                            x_hat,
                            data.view(data.shape[0], -1),
                            torch.zeros_like(x_hat)
                        )
                        log_p_tile = torch.tile(log_p[:, None], (1, C))
                        D[i:i + N, :, k] = log_p_tile.cpu().detach().numpy()

            i += N
            if i % 5000 == 0:
                log_print(f"[{i}/{data_size}] Processing Data...")

    log_print(f"Finished Loading. Applying Density Estimation...")

    # PART 2: Optimized Density Estimation (KDE)
    for k, source_domain in enumerate(source_domains):
        if estimate_prob_type in ["OURS-KDE", "OURS-STD-SCORE", "OURS-STD-SCORE-WITH-KDE"]:

            raw_scores_1d = D[:, 0, k].reshape(-1, 1)
            scores_to_process = raw_scores_1d

            # Standardization
            if "STD-SCORE" in estimate_prob_type:
                log_p_mean, log_p_std = normalize_factors[source_domain]
                std_val = log_p_std.item()
                mean_val = log_p_mean.item()
                if std_val < 1e-6:
                    scores_to_process = (raw_scores_1d - mean_val)
                else:
                    scores_to_process = (raw_scores_1d - mean_val) / std_val

            # KDE Optimization
            if "KDE" in estimate_prob_type:
                score_std = np.std(scores_to_process)

                # Check variance to decide bandwidth range
                if score_std < 0.05:
                    params = {"bandwidth": np.linspace(0.04, 0.5, 25)}
                else:
                    params = {"bandwidth": np.logspace(-2, 2, 40)}

                grid = GridSearchCV(KernelDensity(), params, n_jobs=1, cv=3)
                shuffled_indices = np.random.permutation(len(scores_to_process))
                grid.fit(scores_to_process[shuffled_indices][:3000])

                best_bw = grid.best_estimator_.bandwidth
                log_print(f"Optimal Bandwidth for {source_domain}: {best_bw:.4f}")

                kde = grid.best_estimator_
                log_density = kde.score_samples(scores_to_process)
                final_1d_values = np.exp(log_density).reshape(-1, 1)
            else:
                final_1d_values = np.exp(-np.abs(scores_to_process))

            D[:, :, k] = np.tile(final_1d_values, (1, C))

    # PART 3: Strict Global Noise Gate (20%)
    noise_gate = 0.3
    log_print("Applying STRICT Noise Gate (20% Threshold)...")
    max_probs = D.max(axis=2, keepdims=True)
    mask_weak = D < (max_probs * noise_gate)
    D[mask_weak] = 0.0

    # PART 4: Normalization
    for k in range(len(source_domains)):
        total_sum = D[:, :, k].sum()
        if total_sum > 0:
            D[:, :, k] = D[:, :, k] / total_sum
        else:
            D[:, :, k] = 1.0 / len(D)

    return Y, D, H


def solve_and_evaluate(Y_train, D_train, H_train, Y_test, D_test, H_test,
                       source_domains, seed, init_z_method, multi_dim, fp,
                       target_name_str, target_domains_list,
                       precomputed_weights=None, oracle_mode="real_ratio"):
    """
    Runs solvers + ORACLE + UNIFORM Baseline.
    Detects 'Fake Success' (Fallback) only if weights are IDENTICAL to uniform.
    """
    solvers = ["ORACLE", "UNIFORM", "DC", "CVXPY_GLOBAL", "CVXPY_PER_DOMAIN"]

    # --- OPTIMIZATION SUBSET (From Train Data) ---
    SUBSET_SIZE = 2000
    N_train = len(Y_train)
    if N_train > SUBSET_SIZE:
        np.random.seed(seed)
        opt_indices = np.random.choice(N_train, SUBSET_SIZE, replace=False)
    else:
        opt_indices = np.arange(N_train)

    Y_opt = Y_train[opt_indices]
    D_opt = D_train[opt_indices]
    H_opt = H_train[opt_indices]

    # --- Prepare Constraints ---
    errors_list = [SOURCE_ERRORS[d] + 0.01 for d in source_domains]
    eps_global = max(errors_list)
    eps_vector = np.array(errors_list)

    results = {}

    # Pre-calculate Oracle Weights
    oracle_z_test = get_oracle_weights(target_domains_list, source_domains, mode=oracle_mode)

    # Pre-calculate Uniform Weights
    uniform_weights = np.ones(len(source_domains)) / len(source_domains)

    for solver_name in solvers:
        learned_z = None
        solver_status = "Unknown"

        if solver_name == "ORACLE":
            learned_z = oracle_z_test
            solver_status = "Oracle (Target Truth)"

        elif solver_name == "UNIFORM":
            learned_z = uniform_weights
            solver_status = "Fixed (1/N)"

        elif precomputed_weights is not None and solver_name in precomputed_weights:
            # STATIC MODE
            learned_z = precomputed_weights[solver_name]
            solver_status = "Static (Precomputed)"

        else:
            # DYNAMIC OPTIMIZATION MODE
            log_print(f">>> Running Optimization: {solver_name}")

            # Try deltas until one works
            deltas_to_try = [1.0, 2, 3]

            for delta_val in deltas_to_try:
                try:
                    delta_vector = np.array([delta_val] * len(source_domains))

                    if solver_name == "DC":
                        DP = init_problem_from_model(Y_opt, D_opt, H_opt, p=len(source_domains), C=10)
                        prob = ConvexConcaveProblem(DP)
                        solver = ConvexConcaveSolver(prob, seed, init_z_method)
                        learned_z, _, _ = solver.solve()
                        if learned_z is not None:
                            solver_status = f"Converged (DC)"
                            break

                    elif solver_name == "CVXPY_GLOBAL":
                        learned_z = solve_convex_problem_mosek(
                            Y_opt, D_opt, H_opt, delta=delta_val, epsilon=eps_global, solver_type='SCS'
                        )

                    elif solver_name == "CVXPY_PER_DOMAIN":
                        learned_z = solve_convex_problem_per_domain(
                            Y_opt, D_opt, H_opt, delta=delta_vector, epsilon=eps_vector, solver_type='SCS'
                        )

                    # --- CHECK FOR FALLBACK ---
                    if learned_z is not None:
                        # FIX: Using 1e-9 to ensure we only catch the exact fallback,
                        # while allowing valid optimization results that are close to uniform.
                        is_exact_uniform = np.allclose(learned_z, uniform_weights, atol=1e-9)

                        # If it is EXACTLY uniform (and delta isn't huge), assume it's the fallback
                        if is_exact_uniform:
                            learned_z = None  # Treat as failure
                            continue  # Try next delta
                        else:
                            solver_status = f"Converged (d={delta_val})"
                            break

                except Exception as e:
                    continue

            if learned_z is None:
                log_print(f"   [WARNING] All deltas failed for {solver_name}. Using Uniform.")
                learned_z = uniform_weights
                solver_status = "FAILED (Fallback)"

        # --- EVALUATION (On Test Data) ---
        z_tensor = learned_z.reshape(1, 1, -1)
        weighted_preds = (D_test * H_test) * z_tensor
        final_probs = weighted_preds.sum(axis=2)

        if multi_dim:
            Y_true = Y_test.argmax(axis=1)
            hz_pred = final_probs.argmax(axis=1)
        else:
            Y_true = Y_test
            hz_pred = final_probs.argmax(axis=1)

        score = accuracy_score(y_true=Y_true, y_pred=hz_pred)
        l1_dist = np.sum(np.abs(learned_z - oracle_z_test))

        log_print(f"   [Target: {target_name_str}] Solver: {solver_name} | Status: {solver_status}")
        log_print(f"   Weights: {learned_z} | Score: {score * 100:.2f}%")

        results[solver_name] = {
            "weights": learned_z,
            "score": score * 100,
            "l1_dist": l1_dist,
            "status": solver_status
        }

    return results, oracle_z_test


def run_domain_adaptation(alpha_pos, alpha_neg, vr_model_type, seed, test_path,
                          estimate_prob_type, init_z_method, multi_dim,
                          classifiers, all_source_domains,
                          optimization_scope, sampling_strategy,
                          restrict_sources_to_target=False):
    torch.manual_seed(seed)
    np.random.seed(seed=seed)

    # --- STRING HELPERS FOR FILENAMES ---
    source_mode_str = "Sources_RESTRICTED" if restrict_sources_to_target else "Sources_ALL"

    target_domains_sets = [
        ['MNIST', 'USPS', 'SVHN'],
        ['MNIST', 'USPS'], ['MNIST', 'SVHN'], ['USPS', 'SVHN']
        # ,
        # ['MNIST'], ['USPS'], ['SVHN']
    ]

    models = {}
    normalize_factors = {'MNIST': (0, 0), 'USPS': (0, 0), 'SVHN': (0, 0)}

    # --- Load Source Models ---
    for domain in all_source_domains:
        model = vr_model(alpha_pos, alpha_neg).to(device)
        pattern = f"./models_{domain}_seed{seed}_*/{vr_model_type}_{alpha_pos}_{alpha_neg}_{domain}_seed{seed}_model.pt"
        files = glob.glob(pattern)
        model_path = files[
            0] if files else f"./models_{domain}_seed{seed}_1/{vr_model_type}_{alpha_pos}_{alpha_neg}_{domain}_seed{seed}_model.pt"

        log_print(f"[LOAD] {model_path}")
        try:
            model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
        except FileNotFoundError:
            continue
        models[domain] = model

        # Factors
        if estimate_prob_type in ["OURS-STD-SCORE", "OURS-STD-SCORE-WITH-KDE"]:
            train_loader, _ = Data.get_data_loaders(domain, seed=seed)
            temp_probs = []
            for batch_idx, (data, _) in enumerate(train_loader):
                if batch_idx > 50: break
                with torch.no_grad():
                    data = data.to(device)
                    x_hat, mu, logstd = models[domain](data)
                    log_p = models[domain].compute_log_probabitility_bernoulli(
                        x_hat, data.view(data.shape[0], -1)
                    )
                    temp_probs.append(log_p)
            all_probs = torch.cat(temp_probs, 0)
            normalize_factors[domain] = (all_probs.mean(), all_probs.std())
            log_print(f"Factors {domain}: {normalize_factors[domain]}")

    # --- GLOBAL OPTIMIZATION (If selected) ---
    global_weights = None
    all_summary_data = []

    if optimization_scope == "global":
        log_print(f"\n=== MODE: GLOBAL OPTIMIZATION (Sampling: {sampling_strategy}) ===")

        train_loaders, train_size = create_loaders(all_source_domains, seed, sampling_strategy)
        Y_train, D_train, H_train = build_DP_model_Classes(
            train_loaders, train_size, all_source_domains, models, classifiers,
            normalize_factors, estimate_prob_type
        )

        global_results, global_oracle = solve_and_evaluate(
            Y_train, D_train, H_train, Y_train, D_train, H_train,
            all_source_domains, seed, init_z_method, multi_dim, open(os.devnull, 'w'),
            "GLOBAL_TRAIN", all_source_domains, oracle_mode="balanced"
        )

        global_weights = {k: v['weights'] for k, v in global_results.items() if k != "ORACLE"}
        log_print(f"Global Weights Learned: {global_weights}")

        for solver, res in global_results.items():
            all_summary_data.append({
                "Target Mix": "GLOBAL_TRAIN",
                "Solver": solver,
                "Status": res['status'],
                "Score (%)": round(res['score'], 2),
                "L1 Dist": round(res['l1_dist'], 4),
                "Target Data Ratio": np.round(global_oracle, 2),
                "Learned Weights": np.round(res['weights'], 2),
            })

    # --- FILE NAME CONSTRUCTION ---
    filename = f'Results_{optimization_scope}_{sampling_strategy}_{source_mode_str}_seed{seed}.txt'
    full_file_path = os.path.join(test_path, filename)

    # --- MAIN TARGET LOOP ---
    with open(full_file_path, 'w') as fp:
        log_print(f"--- SAVING RESULTS TO: {filename} ---")

        for target_domains in target_domains_sets:
            target_str = str(target_domains)
            log_print(f"\n=== Processing Target Mix: {target_str} ===")

            # --- DETERMINE SOURCES TO USE ---
            if restrict_sources_to_target:
                current_source_domains = [d for d in all_source_domains if d in target_domains]
            else:
                current_source_domains = all_source_domains

            log_print(f"   --> Using Sources: {current_source_domains}")

            # 1. Prepare EVALUATION Data
            eval_loaders, eval_size = create_loaders(target_domains, seed, "all_data")
            Y_eval, D_eval, H_eval = build_DP_model_Classes(
                eval_loaders, eval_size, current_source_domains, models, classifiers,
                normalize_factors, estimate_prob_type
            )

            # 2. Prepare TRAINING Data
            if optimization_scope == "per_target":
                train_loaders, train_size = create_loaders(target_domains, seed, sampling_strategy)
                if sampling_strategy == "all_data":
                    Y_train, D_train, H_train = Y_eval, D_eval, H_eval
                else:
                    Y_train, D_train, H_train = build_DP_model_Classes(
                        train_loaders, train_size, current_source_domains, models, classifiers,
                        normalize_factors, estimate_prob_type
                    )
            else:
                Y_train, D_train, H_train = Y_eval, D_eval, H_eval  # Dummy

            # 3. Solve & Evaluate
            iter_results, oracle_z = solve_and_evaluate(
                Y_train, D_train, H_train,
                Y_eval, D_eval, H_eval,
                current_source_domains, seed, init_z_method, multi_dim, fp,
                target_str, target_domains,
                precomputed_weights=global_weights
            )

            # Log results
            # NOTE: Calculate full oracle just for the table L1 distance correctness against the full world view
            full_oracle = get_oracle_weights(target_domains, all_source_domains, mode="real_ratio")

            for solver, res in iter_results.items():
                # Map the learned weights (which might be size 2) back to size 3 for the table
                full_weights_display = map_weights_to_full_source_list(
                    res['weights'], current_source_domains, all_source_domains
                )

                fp.write(f"Target: {target_str} | Solver: {solver}\n")
                fp.write(f"Status: {res['status']}\n")
                fp.write(f"Weights (Active): {res['weights']}\n")
                fp.write(f"Weights (Full): {full_weights_display}\n")
                fp.write(f"Score: {res['score']:.2f}\n")

                # Recalculate L1 Dist on the full vectors to be safe and consistent
                real_l1_dist = np.sum(np.abs(full_weights_display - full_oracle))
                fp.write(f"L1 Dist from Oracle: {real_l1_dist:.4f}\n\n")

                all_summary_data.append({
                    "Target Mix": target_str,
                    "Solver": solver,
                    "Status": res['status'],
                    "Score (%)": round(res['score'], 2),
                    "L1 Dist": round(real_l1_dist, 4),
                    "Target Data Ratio": np.round(full_oracle, 2),
                    "Learned Weights": np.round(full_weights_display, 2),
                })

        # Summary
        fp.write("\n\n" + "=" * 50 + "\nSUMMARY TABLE\n" + "=" * 50 + "\n")
        df_summary = pd.DataFrame(all_summary_data)
        fp.write(df_summary.to_string())
        log_print("\nSUMMARY TABLE:\n")
        log_print(df_summary.to_string())


def task_run(date, seed, estimate_prob_type, init_z_method, multi_dim,
             model_type, pos_alpha, neg_alpha,
             classifiers, source_domains, opt_scope, sample_strat, restrict_sources):
    # --- NAME CONSTRUCTION: Includes Scope, Strategy, and Source Restriction ---
    source_mode_str = "Sources_RESTRICTED" if restrict_sources else "Sources_ALL"

    test_path = (
        f'./{estimate_prob_type}_results_{date}/'
        f'init_z_{init_z_method}/use_multi_dim_{multi_dim}/seed_{seed}/'
        f'{opt_scope}_{sample_strat}_{source_mode_str}'  # <--- UNIQUE FOLDER NAME
    )
    os.makedirs(test_path, exist_ok=True)

    run_domain_adaptation(
        pos_alpha, neg_alpha, model_type, seed, test_path,
        estimate_prob_type, init_z_method, multi_dim,
        classifiers, source_domains, opt_scope, sample_strat,
        restrict_sources_to_target=restrict_sources
    )


def main():
    print("device = ", device)
    classifiers = {}
    source_domains = ['MNIST', 'USPS', 'SVHN']

    for domain in source_domains:
        _, test_loader = Data.get_data_loaders(domain, seed=1)
        classifier = ClSFR.Grey_32_64_128_gp().to(device)
        try:
            classifier.load_state_dict(
                torch.load(f"./classifiers_new/{domain}_classifier.pt",
                           map_location=torch.device(device))
            )
            classifiers[domain] = classifier
        except FileNotFoundError:
            print(f"Warning: Classifier for {domain} not found.")

    # Using the best config found
    estimate_prob_types = ["OURS-STD-SCORE-WITH-KDE"]
    init_z_methods = ["err"]
    multi_dim_vals = [True]
    date = '30_12_EXPERIMENTS'
    seeds = [1]

    alphas_by_model = {
        "vrs": [(2, -2)],
    }

    # --- CONFIGURATION ---

    # 1. Choose Mode:
    OPTIMIZATION_SCOPE = "per_target"  # 'global' or 'per_target'
    SAMPLING_STRATEGY = "all_data"  # 'balanced_1000' or 'all_data'

    # 2. Choose Source Strategy:
    # True = Use ONLY sources present in target (e.g., Target=M+U -> Sources=M+U)
    # False = Use ALL sources (e.g., Target=M+U -> Sources=M+U+S)
    RESTRICT_SOURCES = True

    tasks = []
    for seed in seeds:
        for estimate_prob_type in estimate_prob_types:
            for init_z_method in init_z_methods:
                for multi_dim in multi_dim_vals:
                    for model_type, pairs in alphas_by_model.items():
                        for (pos_alpha, neg_alpha) in pairs:
                            tasks.append(
                                (date, seed, estimate_prob_type,
                                 init_z_method, multi_dim,
                                 model_type, pos_alpha, neg_alpha,
                                 classifiers, source_domains,
                                 OPTIMIZATION_SCOPE, SAMPLING_STRATEGY, RESTRICT_SOURCES)
                            )

    mode_desc = "RESTRICTED SOURCES" if RESTRICT_SOURCES else "ALL SOURCES"
    print(f"Starting execution... Mode: {OPTIMIZATION_SCOPE} | Sampling: {SAMPLING_STRATEGY} | {mode_desc}")

    Parallel(n_jobs=-1, backend="loky")(
        delayed(task_run)(*t) for t in tasks
    )


if __name__ == "__main__":
    main()