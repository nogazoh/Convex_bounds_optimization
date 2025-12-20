from __future__ import print_function
import time
import torch.utils.data
import glob
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

# --- YOUR LOCAL MODULES ---
from dc import *
import classifier as ClSFR
import matplotlib.pyplot as plt
from vae import *
import data as Data

# --- IMPORTS FOR SOLVERS ---
from cvxpy_solver import solve_convex_problem_mosek
from cvxpy_solver_per_domain import solve_convex_problem_per_domain

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device = ", device)
os.environ.setdefault("OMP_NUM_THREADS", "1")
torch.set_num_threads(1)

Z_MEMORY_CACHE = {}


def calculate_expected_loss_local(Y, H, D, num_sources):
    L_mat = np.zeros((num_sources, num_sources))
    for j in range(num_sources):
        if H.ndim == 3:
            pred_j = H[:, :, j]
        else:
            pred_j = H[:, j]
        loss_vec = np.abs(pred_j - Y)
        for t in range(num_sources):
            if D.ndim == 3:
                p_t = D[:, :, t]
            else:
                p_t = D[:, t]
            L_mat[j, t] = np.sum(p_t * loss_vec)
    return L_mat


def build_DP_model_Classes(data_loaders, data_size, source_domains, models, classifiers, test_path, normalize_factors,
                           estimate_prob_type):
    C = 10
    Y = np.zeros((data_size, C))
    D = np.zeros((data_size, C, len(source_domains)))
    H = np.zeros((data_size, C, len(source_domains)))

    i = 0
    for target_domain, data_loader in data_loaders:
        for data, label in data_loader:
            data = data.to(device)
            N = len(data)
            y_vals = label.cpu().detach().numpy()
            one_hot = np.zeros((y_vals.size, C))
            one_hot[np.arange(y_vals.size), y_vals] = 1
            Y[i:i + N] = one_hot

            for k, source_domain in enumerate(source_domains):
                with torch.no_grad():
                    output = classifiers[source_domain](data)
                    norm_output = F.softmax(output, dim=1)
                    H[i:i + N, :, k] = norm_output.cpu().detach().numpy()

                    if estimate_prob_type == "GMSA":
                        D[i:i + N, :, k] = output.cpu().detach().numpy()
                    elif estimate_prob_type in ["OURS-STD-SCORE", "OURS-KDE", "OURS-STD-SCORE-WITH-KDE"]:
                        x_hat, _, _ = models[source_domain](data)
                        log_p = models[source_domain].compute_log_probabitility_gaussian(
                            x_hat, data.view(data.shape[0], -1), torch.zeros_like(x_hat)
                        )
                        log_p_tile = torch.tile(log_p[:, None], (1, C))
                        D[i:i + N, :, k] = log_p_tile.cpu().detach().numpy()
            i += N

    for k, source_domain in enumerate(source_domains):
        if estimate_prob_type == "GMSA" or estimate_prob_type == "OURS-KDE":
            params = {"bandwidth": np.logspace(-2, 2, 40)}
            grid = GridSearchCV(KernelDensity(), params, n_jobs=1)
            data = D[:, :, k]
            shuffled_indices = np.random.permutation(len(data))
            data_shuffle = data[shuffled_indices]
            grid.fit(data_shuffle[:2000])
            kde = grid.best_estimator_
            log_density = kde.score_samples(data)
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
                data = standard_score
                shuffled_indices = np.random.permutation(len(data))
                data_shuffle = data[shuffled_indices]
                grid.fit(data_shuffle[:2000])
                kde = grid.best_estimator_
                log_density = kde.score_samples(data)
                log_density_tile = np.tile(log_density[:, None], (1, C))
                D[:, :, k] = np.exp(log_density_tile)
        D[:, :, k] = D[:, :, k] / D[:, :, k].sum()
    return Y, D, H


def build_DP_model(data_loaders, data_size, source_domains, models, classifiers, test_path, normalize_factors,
                   estimate_prob_type):
    C = 10
    Y = np.zeros((data_size))
    D = np.zeros((data_size, len(source_domains)))
    H = np.zeros((data_size, len(source_domains)))
    all_output = np.zeros((data_size, C, len(source_domains)))

    i = 0
    for target_domain, data_loader in data_loaders:
        for data, label in data_loader:
            data = data.to(device)
            N = len(data)
            y_vals = label.cpu().detach().numpy()
            Y[i:i + N] = y_vals
            for k, source_domain in enumerate(source_domains):
                with torch.no_grad():
                    output = classifiers[source_domain](data)
                    norm_output = F.softmax(output, dim=1)
                    y_pred = norm_output.data.max(1, keepdim=True)[1]
                    y_pred = y_pred.flatten().cpu().detach().numpy()
                    H[i:i + N, k] = y_pred
                    if estimate_prob_type == "GMSA":
                        all_output[i:i + N, :, k] = output.cpu().detach().numpy()
                    elif estimate_prob_type in ["OURS-STD-SCORE", "OURS-KDE", "OURS-STD-SCORE-WITH-KDE"]:
                        x_hat, _, _ = models[source_domain](data)
                        log_p = models[source_domain].compute_log_probabitility_gaussian(
                            x_hat, data.view(data.shape[0], -1), torch.zeros_like(x_hat)
                        )
                        D[i:i + N, k] = log_p.cpu().detach().numpy()
            i += N

    for k, source_domain in enumerate(source_domains):
        if estimate_prob_type == "GMSA" or estimate_prob_type == "OURS-KDE":
            if estimate_prob_type == "GMSA":
                data = all_output[:, :, k]
            elif estimate_prob_type == "OURS-KDE":
                data = D[:, k]
                data = data[:, None]
            params = {"bandwidth": np.logspace(-2, 2, 40)}
            grid = GridSearchCV(KernelDensity(), params, n_jobs=1)
            shuffled_indices = np.random.permutation(len(data))
            data_shuffle = data[shuffled_indices]
            grid.fit(data_shuffle[:2000])
            kde = grid.best_estimator_
            log_density = kde.score_samples(data)
            D[:, k] = np.exp(log_density)
        elif estimate_prob_type == "OURS-STD-SCORE":
            log_p_mean, log_p_std = normalize_factors[source_domain]
            standard_score = (D[:, k] - log_p_mean) / log_p_std
            D[:, k] = np.exp(-np.abs(standard_score))
        D[:, k] = D[:, k] / D[:, k].sum()
    return Y, D, H


def run_single_optimization(solver_name, Y, D, H, source_domains, seed, init_z_method, test_path,
                            initial_epsilon_multiplier):
    """
    Runs solver with AUTO-RELAXATION.
    If 'infeasible', it increases epsilon multiplier and retries.
    """
    global Z_MEMORY_CACHE
    sources_key = tuple(sorted(source_domains))
    # NOTE: Using initial_epsilon_multiplier in key to distinguish different starting points
    cache_key = (solver_name, sources_key, seed, initial_epsilon_multiplier)

    if cache_key in Z_MEMORY_CACHE:
        return Z_MEMORY_CACHE[cache_key]

    z_filename = f'DC_accuracy_score_{seed}_{solver_name}.txt'
    full_path = os.path.join(test_path, z_filename)
    should_recalculate = True

    if os.path.exists(full_path):
        try:
            with open(full_path, 'r') as fp:
                first_line = fp.readline()
                parts = first_line.strip().split('\t')
                loaded_z_dict = {}
                for part in parts:
                    if '=' in part:
                        key, val = part.split('=')
                        domain_name = key.replace('z_', '').strip()
                        loaded_z_dict[domain_name] = float(val)
                z_vector = []
                for domain in source_domains:
                    z_vector.append(loaded_z_dict.get(domain, 0.0))
                learned_z = np.array(z_vector)

                # Check if it's the uniform fallback (0.333...) which we want to improve upon
                is_uniform = np.allclose(learned_z, np.ones(len(learned_z)) / len(learned_z), atol=1e-3)

                if np.sum(learned_z) == 0:
                    print(f"   [CACHE] Found {z_filename} (invalid zeros). Recalculating...")
                    should_recalculate = True
                elif is_uniform and solver_name != "DC":
                    # If we loaded a failed (uniform) result from disk, let's try to improve it!
                    print(f"   [CACHE] Found {z_filename} (uniform fallback). Retrying with relaxation...")
                    should_recalculate = True
                else:
                    print(f"   [CACHE] Loaded valid Z from {z_filename}: {learned_z}")
                    Z_MEMORY_CACHE[cache_key] = learned_z
                    should_recalculate = False
                    return learned_z
        except Exception:
            should_recalculate = True

    if should_recalculate:
        learned_z = None
        k = len(source_domains)
        L_mat_temp = calculate_expected_loss_local(Y, H, D, k)
        min_losses = np.min(L_mat_temp, axis=0)

        # --- AUTO-RELAXATION LOOP ---
        current_multiplier = initial_epsilon_multiplier
        # Try up to 5 times, increasing by 20% each time (1.05 -> 1.26 -> 1.51...)
        max_retries = 5

        for attempt in range(max_retries):
            EPSILONS_VEC = min_losses * current_multiplier
            DELTAS_VEC = [0.1] * k  # Keep delta constant or relax it too if needed
            AVG_EPS = np.mean(EPSILONS_VEC)
            AVG_DELTA = 0.1
            CHOSEN_BACKEND = 'SCS'

            print(
                f"   >>> [Solver: {solver_name}] Attempt {attempt + 1}/{max_retries} with Mult={current_multiplier:.3f}")
            print(f"       Epsilons: {np.round(EPSILONS_VEC, 4)}")

            if solver_name == "DC":
                DP = init_problem_from_model(Y, D, H, p=k, C=10)
                prob = ConvexConcaveProblem(DP)
                solver = ConvexConcaveSolver(prob, seed, init_z_method)
                z_res, _, _ = solver.solve()
                learned_z = z_res
                break  # DC always returns something

            elif solver_name == "CVXPY":
                try:
                    learned_z = solve_convex_problem_mosek(
                        Y, D, H, delta=AVG_DELTA, epsilon=AVG_EPS, solver_type=CHOSEN_BACKEND
                    )
                    # Check if returned uniform (fallback inside solver)
                    if not np.allclose(learned_z, np.ones(k) / k, atol=1e-3):
                        print("       -> Success!")
                        break  # Found a real solution!
                except Exception as e:
                    print(f"       -> Failed: {e}")

            elif solver_name == "CVXPY_PER_DOMAIN":
                try:
                    learned_z = solve_convex_problem_per_domain(
                        Y, D, H, delta=DELTAS_VEC, epsilon=EPSILONS_VEC, solver_type=CHOSEN_BACKEND
                    )
                    # Check if returned uniform (fallback inside solver)
                    if not np.allclose(learned_z, np.ones(k) / k, atol=1e-3):
                        print("       -> Success!")
                        break  # Found a real solution!
                except Exception as e:
                    print(f"       -> Failed: {e}")

            # If we are here, solver failed or returned uniform. Relax and retry.
            print("       -> Solver returned uniform/failed. Relaxing constraints...")
            current_multiplier *= 1.2  # Relax by 20%

        # If loop finishes and we still have None or Uniform, fallback is inevitable
        if learned_z is None:
            learned_z = np.ones(k) / k

        # Update memory cache
        Z_MEMORY_CACHE[cache_key] = learned_z

        return learned_z


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
    correct = np.sum(Y_true == hz_pred)
    total = len(Y_true)
    return correct, total


def task_run(date, seed, estimate_prob_type, init_z_method, multi_dim,
             model_type, pos_alpha, neg_alpha,
             classifiers, source_domains, epsilon_multiplier):
    test_path = (
        f'./{estimate_prob_type}_results_{date}_eps_{epsilon_multiplier}/'
        f'init_z_{init_z_method}/use_multi_dim_{multi_dim}/seed_{seed}/'
        f'model_type_{model_type}___pos_alpha_{pos_alpha}___neg_alpha_{neg_alpha}'
    )
    os.makedirs(test_path, exist_ok=True)

    return run_domain_adaptation(
        pos_alpha, neg_alpha, model_type, seed, test_path,
        estimate_prob_type, init_z_method, multi_dim,
        classifiers, source_domains, epsilon_multiplier
    )


def run_domain_adaptation(alpha_pos, alpha_neg, vr_model_type, seed, test_path, estimate_prob_type, init_z_method,
                          multi_dim, classifiers, source_domains, epsilon_multiplier):
    torch.manual_seed(seed)
    np.random.seed(seed=seed)
    logging_filename = "domain_adaptation.log"
    logging.basicConfig(filename=logging_filename, level=logging.DEBUG)

    target_domains_sets = [
        ['MNIST', 'USPS', 'SVHN'],
        ['MNIST', 'USPS'], ['MNIST', 'SVHN'], ['USPS', 'SVHN'],
        ['MNIST'], ['USPS'], ['SVHN']
    ]
    unique_targets = ['MNIST', 'USPS', 'SVHN']

    models = {}
    normalize_factors = {'MNIST': (0, 0), 'USPS': (0, 0), 'SVHN': (0, 0)}
    print(f"\n>>>> [INIT] ({vr_model_type} {alpha_pos}, {alpha_neg}) <<<<")

    for domain in source_domains:
        model = vr_model(alpha_pos, alpha_neg).to(device)
        filename = f"{vr_model_type}_{alpha_pos}_{alpha_neg}_{domain}_seed1_model.pt"
        pattern = f"./models_{domain}_seed1*/{filename}"
        matches = glob.glob(pattern)
        if not matches: continue
        model.load_state_dict(torch.load(max(matches, key=os.path.getmtime), map_location=torch.device(device)))
        models[domain] = model

        if estimate_prob_type in ["OURS-STD-SCORE", "OURS-STD-SCORE-WITH-KDE"]:
            train_loader, test_loader = Data.get_data_loaders(domain, seed=seed)
            domain_probs = torch.tensor([]).to(device)
            for data, _ in train_loader:
                with torch.no_grad():
                    data = data.to(device)
                    x_hat, mu, logstd = models[domain](data)
                    log_p = models[domain].compute_log_probabitility_gaussian(x_hat, data.view(data.shape[0], -1),
                                                                              torch.zeros_like(x_hat))
                    domain_probs = torch.cat((log_p, domain_probs), 0)
            normalize_factors[domain] = (domain_probs.mean().item(), domain_probs.std().item())

    # 1. Train Matrices
    print(f"  > Preparing Data Matrices...")
    data_size = 0
    data_loaders = []
    for k, domain in enumerate(source_domains):
        train_loader, _ = Data.get_data_loaders(domain, seed=seed, num_datapoints=1000)
        data_size += len(train_loader.dataset)
        data_loaders.append((domain, train_loader))

    if multi_dim:
        Y_opt, D_opt, H_opt = build_DP_model_Classes(data_loaders, data_size, source_domains, models, classifiers,
                                                     test_path, normalize_factors, estimate_prob_type)
    else:
        Y_opt, D_opt, H_opt = build_DP_model(data_loaders, data_size, source_domains, models, classifiers, test_path,
                                             normalize_factors, estimate_prob_type)

    # 2. Test Cache
    print(f"  > Pre-computing Test Matrices...")
    test_cache = {}
    for domain in unique_targets:
        _, test_loader = Data.get_data_loaders(domain, seed=seed)
        d_size = len(test_loader.dataset)
        d_loaders = [(domain, test_loader)]
        if multi_dim:
            Y_t, D_t, H_t = build_DP_model_Classes(d_loaders, d_size, source_domains, models, classifiers, test_path,
                                                   normalize_factors, estimate_prob_type)
        else:
            Y_t, D_t, H_t = build_DP_model(d_loaders, d_size, source_domains, models, classifiers, test_path,
                                           normalize_factors, estimate_prob_type)
        test_cache[domain] = (Y_t, D_t, H_t)

    # 3. Solvers
    solvers_to_run = ["DC", "CVXPY", "CVXPY_PER_DOMAIN"]
    results_for_this_config = []
    weights_col_name = f"Weights ({', '.join(source_domains)})"

    for solver_name in solvers_to_run:
        learned_z = run_single_optimization(solver_name, Y_opt, D_opt, H_opt, source_domains, seed, init_z_method,
                                            test_path, epsilon_multiplier)
        print(f"   > Final Z ({solver_name}): {learned_z}")

        z_str = "[" + ", ".join([f"{z:.3f}" for z in learned_z]) + "]"
        z_filename = f'DC_accuracy_score_{seed}_{solver_name}.txt'

        with open(os.path.join(test_path, z_filename), 'w') as fp:
            fp.write(f'\nz_MNIST = {learned_z[0]}\tz_USPS = {learned_z[1]}\tz_SVHN = {learned_z[2]}\n')
            model_name_str = f"{vr_model_type.upper()}_{alpha_pos}_{alpha_neg}_{solver_name}"

            domain_stats = {}
            for domain in unique_targets:
                Y_c, D_c, H_c = test_cache[domain]
                correct, total = evaluate_z_on_data(Y_c, D_c, H_c, learned_z, multi_dim, source_domains)
                domain_stats[domain] = (correct, total)

            for target_domains in target_domains_sets:
                total_correct = 0
                total_samples = 0
                for d in target_domains:
                    c, t = domain_stats[d]
                    total_correct += c
                    total_samples += t
                score = total_correct / total_samples if total_samples > 0 else 0

                short_names = []
                if 'MNIST' in target_domains: short_names.append('m')
                if 'SVHN' in target_domains: short_names.append('s')
                if 'USPS' in target_domains: short_names.append('u')
                col_key = "".join(sorted(short_names))

                results_for_this_config.append({
                    "Model": model_name_str,
                    "Target": col_key,
                    "Score": score * 100,
                    weights_col_name: z_str
                })
                fp.write(f"{target_domains}\t{score * 100}\n")

    return results_for_this_config


def main():
    print("device = ", device)

    # --- START WITH TIGHT EPSILON (1.05) ---
    # The code will automatically relax it if needed
    EPSILON_MULTIPLIER = 1.05

    classifiers = {}
    source_domains = ['MNIST', 'USPS', 'SVHN']
    for domain in source_domains:
        classifier = ClSFR.Grey_32_64_128_gp().to(device)
        try:
            classifier.load_state_dict(
                torch.load(f"./classifiers_new/{domain}_classifier.pt", map_location=torch.device(device)))
            classifiers[domain] = classifier
        except FileNotFoundError:
            pass

    estimate_prob_types = ["OURS-STD-SCORE-WITH-KDE"]
    init_z_methods = ["err"]
    multi_dim_vals = [True]
    date = '15_12'
    seeds = [10]
    alphas_by_model = {
        "vrs": [(2, -2), (2, -0.5), (0.5, -2), (0.5, -0.5)],
        "vr": [(2, -1), (0.5, -1)],
        "vae": [(1, -1)],
    }

    tasks = []
    for seed in seeds:
        for estimate_prob_type in estimate_prob_types:
            for init_z_method in init_z_methods:
                for multi_dim in multi_dim_vals:
                    for model_type, pairs in alphas_by_model.items():
                        for (pos_alpha, neg_alpha) in pairs:
                            tasks.append(
                                (date, seed, estimate_prob_type, init_z_method, multi_dim, model_type, pos_alpha,
                                 neg_alpha))

    print("\n🚀 Starting Parallel Execution (n_jobs=-1)...")
    results_lists = Parallel(n_jobs=-1, verbose=10)(
        delayed(task_run)(*t, classifiers, source_domains, EPSILON_MULTIPLIER) for t in tasks
    )

    flat_results = [item for sublist in results_lists for item in sublist]
    df = pd.DataFrame(flat_results)

    if not df.empty:
        weights_col = f"Weights ({', '.join(source_domains)})"
        final_table = df.pivot_table(index=['Model', weights_col], columns='Target', values='Score')
        final_table['mean'] = final_table.mean(axis=1)
        final_table = final_table.round(1)

        base_name = f'final_summary_results_{date}_eps_{EPSILON_MULTIPLIER}'
        extension = '_COMPARISON.csv'
        version = 1
        while True:
            output_csv_name = f'{base_name}_v{version}{extension}'
            if not os.path.exists(output_csv_name): break
            version += 1

        final_table.to_csv(output_csv_name)
        print(f"\n✅ Saved to: {output_csv_name}")
        print(final_table)
    else:
        print("No results collected.")


if __name__ == "__main__":
    main()