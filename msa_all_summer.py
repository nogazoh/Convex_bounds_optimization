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


def get_oracle_weights(target_domains, source_domains):
    """
    Calculates the TRUE ratio of the source domains within the current target mix.
    If a source domain is NOT in the target mix, its weight is 0.
    """
    weights = []
    total_samples = 0

    # Calculate total relevant samples
    for domain in target_domains:
        if domain in TEST_SET_SIZES:
            total_samples += TEST_SET_SIZES[domain]

    # Calculate weight for each source domain
    for source in source_domains:
        if source in target_domains:
            w = TEST_SET_SIZES[source] / total_samples
        else:
            w = 0.0
        weights.append(w)

    return np.array(weights)


def build_DP_model_Classes(data_loaders, data_size, source_domains, models, classifiers, test_path,
                           normalize_factors, estimate_prob_type):
    """
    VERSION 5: Target-Optimized Matrix Construction
    """
    log_print(f"--- Starting build_DP_model_Classes (VERSION 5 - TARGET OPTIMIZED) ---")

    C = 10
    Y = np.zeros((data_size, C))
    D = np.zeros((data_size, C, len(source_domains)))
    H = np.zeros((data_size, C, len(source_domains)))
    i = 0

    # ---------------------------------------------------------
    # PART 1: Data Loading & Raw Score Calculation
    # ---------------------------------------------------------
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

    # ---------------------------------------------------------
    # PART 2: Optimized Density Estimation (KDE)
    # ---------------------------------------------------------
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

                grid = GridSearchCV(KernelDensity(), params, n_jobs=1, cv=3)  # n_jobs=1 inside parallel
                shuffled_indices = np.random.permutation(len(scores_to_process))
                # Fit on subset to speed up
                grid.fit(scores_to_process[shuffled_indices][:3000])

                best_bw = grid.best_estimator_.bandwidth
                log_print(f"Optimal Bandwidth for {source_domain}: {best_bw:.4f}")

                kde = grid.best_estimator_
                log_density = kde.score_samples(scores_to_process)
                final_1d_values = np.exp(log_density).reshape(-1, 1)
            else:
                final_1d_values = np.exp(-np.abs(scores_to_process))

            D[:, :, k] = np.tile(final_1d_values, (1, C))

    # ---------------------------------------------------------
    # PART 3: Strict Global Noise Gate (20%)
    # ---------------------------------------------------------
    noise_gate = 0.3
    log_print("Applying STRICT Noise Gate (20% Threshold)...")
    max_probs = D.max(axis=2, keepdims=True)
    mask_weak = D < (max_probs * noise_gate)  # Kill if less than 20% of winner
    D[mask_weak] = 0.0

    # ---------------------------------------------------------
    # PART 4: Normalization
    # ---------------------------------------------------------
    for k in range(len(source_domains)):
        total_sum = D[:, :, k].sum()
        if total_sum > 0:
            D[:, :, k] = D[:, :, k] / total_sum
        else:
            D[:, :, k] = 1.0 / len(D)

    return Y, D, H


def solve_and_evaluate(Y, D, H, source_domains, seed, init_z_method, multi_dim, fp, target_name_str,
                       target_domains_list):
    """
    Runs all 3 solvers + ORACLE using SUBSAMPLING optimization with RELAXATION LOOP.
    """
    solvers = ["ORACLE", "DC", "CVXPY_GLOBAL", "CVXPY_PER_DOMAIN"]

    # --- CONFIG ---
    SUBSET_SIZE = 2000
    N_total = len(Y)

    # Create Indices for Optimization
    if N_total > SUBSET_SIZE:
        np.random.seed(seed)  # Ensure reproducibility
        opt_indices = np.random.choice(N_total, SUBSET_SIZE, replace=False)
    else:
        opt_indices = np.arange(N_total)

    # Create Subsets for the Solver
    Y_opt = Y[opt_indices]
    D_opt = D[opt_indices]
    H_opt = H[opt_indices]

    # --- Prepare Constraints ---
    errors_list = [SOURCE_ERRORS[d] + 0.01 for d in source_domains]
    eps_global = max(errors_list)
    eps_vector = np.array(errors_list)

    results = {}

    # Pre-calculate Oracle Weights once
    oracle_z = get_oracle_weights(target_domains_list, source_domains)

    for solver_name in solvers:
        log_print(f">>> Running Solver: {solver_name} (on {len(opt_indices)} samples) for Target: {target_name_str}")
        learned_z = None

        if solver_name == "ORACLE":
            learned_z = oracle_z
        else:
            # --- RELAXATION LOOP ---
            deltas_to_try = [0.01, 0.05, 0.1, 0.2]

            for delta_val in deltas_to_try:
                try:
                    delta_vector = np.array([delta_val] * len(source_domains))

                    # --- STEP A: OPTIMIZE Z on SUBSET ---
                    if solver_name == "DC":
                        # DC doesn't use explicit delta in this implementation
                        DP = init_problem_from_model(Y_opt, D_opt, H_opt, p=len(source_domains), C=10)
                        prob = ConvexConcaveProblem(DP)
                        solver = ConvexConcaveSolver(prob, seed, init_z_method)
                        learned_z, _, _ = solver.solve()
                        break  # Success

                    elif solver_name == "CVXPY_GLOBAL":
                        learned_z = solve_convex_problem_mosek(
                            Y_opt, D_opt, H_opt, delta=delta_val, epsilon=eps_global, solver_type='SCS'
                        )
                        if learned_z is not None: break

                    elif solver_name == "CVXPY_PER_DOMAIN":
                        learned_z = solve_convex_problem_per_domain(
                            Y_opt, D_opt, H_opt, delta=delta_vector, epsilon=eps_vector, solver_type='SCS'
                        )
                        if learned_z is not None: break

                except Exception as e:
                    continue

            if learned_z is None:
                log_print(f"   [WARNING] All deltas failed for {solver_name}. Using Uniform.")
                learned_z = np.ones(len(source_domains)) / len(source_domains)

        # --- STEP B: EVALUATE on FULL DATASET ---
        z_tensor = learned_z.reshape(1, 1, -1)
        weighted_preds = (D * H) * z_tensor
        final_probs = weighted_preds.sum(axis=2)

        if multi_dim:
            Y_true = Y.argmax(axis=1)
            hz_pred = final_probs.argmax(axis=1)
        else:
            Y_true = Y
            hz_pred = final_probs.argmax(axis=1)

        score = accuracy_score(y_true=Y_true, y_pred=hz_pred)

        # Calculate L1 Distance from Oracle
        l1_dist = np.sum(np.abs(learned_z - oracle_z))

        log_print(f"   [Target: {target_name_str}] Weights ({solver_name}): {learned_z}")
        log_print(f"   [Target: {target_name_str}] Score ({solver_name}): {score * 100:.2f}% | L1 Dist: {l1_dist:.4f}")

        # Store for summary table
        results[solver_name] = {
            "weights": learned_z,
            "score": score * 100,
            "l1_dist": l1_dist
        }

    return results, oracle_z


def run_domain_adaptation(alpha_pos, alpha_neg, vr_model_type, seed, test_path,
                          estimate_prob_type, init_z_method, multi_dim,
                          classifiers, source_domains):
    # Re-seed inside process for safety
    torch.manual_seed(seed)
    np.random.seed(seed=seed)

    target_domains_sets = [
        ['MNIST', 'USPS', 'SVHN'],
        ['MNIST', 'USPS'], ['MNIST', 'SVHN'], ['USPS', 'SVHN'],
        ['MNIST'], ['USPS'], ['SVHN']
    ]

    models = {}
    normalize_factors = {'MNIST': (0, 0), 'USPS': (0, 0), 'SVHN': (0, 0)}

    # --- Load Source Models & Stats ---
    for domain in source_domains:
        model = vr_model(alpha_pos, alpha_neg).to(device)
        pattern = f"./models_{domain}_seed{seed}_*/{vr_model_type}_{alpha_pos}_{alpha_neg}_{domain}_seed{seed}_model.pt"
        files = glob.glob(pattern)
        model_path = files[
            0] if files else f"./models_{domain}_seed{seed}_1/{vr_model_type}_{alpha_pos}_{alpha_neg}_{domain}_seed{seed}_model.pt"

        log_print(f"[LOAD] {model_path}")
        try:
            model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
        except FileNotFoundError:
            log_print(f"CRITICAL: Model file not found for {domain}")
            continue
        models[domain] = model

        # Calc factors on source
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

    # --- MAIN TARGET LOOP ---
    all_summary_data = []

    with open(test_path + r'/Results_ALL_SOLVERS_{}.txt'.format(seed), 'w') as fp:
        for target_domains in target_domains_sets:
            target_str = str(target_domains)
            log_print(f"\n=== Processing Target Mix: {target_str} ===")

            target_data_size = 0
            target_data_loaders = []
            for domain in target_domains:
                _, test_loader = Data.get_data_loaders(domain, seed=seed)
                target_data_size += len(test_loader.dataset)
                target_data_loaders.append((domain, test_loader))

            Y_tgt, D_tgt, H_tgt = build_DP_model_Classes(
                target_data_loaders, target_data_size, source_domains,
                models, classifiers, test_path,
                normalize_factors, estimate_prob_type
            )

            # Solve & Evaluate
            iter_results, oracle_z = solve_and_evaluate(Y_tgt, D_tgt, H_tgt, source_domains, seed,
                                                        init_z_method, multi_dim, fp, target_str, target_domains)

            # Write detailed logs
            for solver, res in iter_results.items():
                fp.write(f"Target: {target_str} | Solver: {solver}\n")
                fp.write(f"Weights: {res['weights']}\n")
                fp.write(f"Score: {res['score']:.2f}\n")
                fp.write(f"L1 Dist from Oracle: {res['l1_dist']:.4f}\n\n")

                # Collect for summary table
                all_summary_data.append({
                    "Target Mix": target_str,
                    "True Ratio (Oracle)": np.round(oracle_z, 2),
                    "Solver": solver,
                    "Learned Weights": np.round(res['weights'], 2),
                    "Score (%)": round(res['score'], 2),
                    "L1 Dist": round(res['l1_dist'], 4)
                })

        # Write Summary Table
        fp.write("\n\n" + "=" * 50 + "\nSUMMARY TABLE\n" + "=" * 50 + "\n")
        df_summary = pd.DataFrame(all_summary_data)
        fp.write(df_summary.to_string())
        log_print("\nSUMMARY TABLE:\n")
        log_print(df_summary.to_string())


def task_run(date, seed, estimate_prob_type, init_z_method, multi_dim,
             model_type, pos_alpha, neg_alpha,
             classifiers, source_domains):
    test_path = (
        f'./{estimate_prob_type}_results_{date}/'
        f'init_z_{init_z_method}/use_multi_dim_{multi_dim}/seed_{seed}/'
        f'model_type_{model_type}___pos_alpha_{pos_alpha}___neg_alpha_{neg_alpha}'
    )
    os.makedirs(test_path, exist_ok=True)
    run_domain_adaptation(
        pos_alpha, neg_alpha, model_type, seed, test_path,
        estimate_prob_type, init_z_method, multi_dim,
        classifiers, source_domains
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
    date = '29_12_v5_all_solvers_with_oracle'
    seeds = [1]

    alphas_by_model = {
        "vrs": [(2, -2)],
        "vr": [(2, -1)],
    }

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
                                 model_type, pos_alpha, neg_alpha)
                            )

    print(f"Starting execution of {len(tasks)} tasks in PARALLEL...")

    # ENABLE PARALLEL EXECUTION
    # Using n_jobs=-1 to use all cores
    Parallel(n_jobs=-1, backend="loky")(
        delayed(task_run)(*t, classifiers, source_domains) for t in tasks
    )


if __name__ == "__main__":
    main()