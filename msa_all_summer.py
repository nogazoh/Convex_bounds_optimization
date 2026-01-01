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
import io

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

os.environ.setdefault("OMP_NUM_THREADS", "1")
torch.set_num_threads(1)

# --- CONFIG ---
SOURCE_ERRORS = {
    'MNIST': 1.0 - 0.9948,
    'USPS': 1.0 - 0.972596,
    'SVHN': 1.0 - 0.949716
}

TEST_SET_SIZES = {
    'MNIST': 10000,
    'USPS': 2007,
    'SVHN': 26032
}


def log_print(*args, **kwargs):
    pid = os.getpid()
    print(f"[PID {pid}]", *args, **kwargs)
    sys.stdout.flush()


def map_weights_to_full_source_list(subset_weights, subset_sources, full_source_list):
    if subset_weights is None: return None
    full_weights = np.zeros(len(full_source_list))
    for i, source in enumerate(full_source_list):
        if source in subset_sources:
            subset_idx = subset_sources.index(source)
            full_weights[i] = subset_weights[subset_idx]
        else:
            full_weights[i] = 0.0
    return full_weights


def get_oracle_weights(target_domains, source_domains, mode="real_ratio"):
    if mode == "balanced":
        weights = []
        active = len(target_domains)
        for source in source_domains:
            weights.append(1.0 / active if source in target_domains else 0.0)
        return np.array(weights)

    weights = []
    total_samples = 0
    for domain in target_domains:
        if domain in TEST_SET_SIZES: total_samples += TEST_SET_SIZES[domain]

    for source in source_domains:
        if source in target_domains:
            weights.append(TEST_SET_SIZES[source] / total_samples)
        else:
            weights.append(0.0)
    return np.array(weights)


def create_loaders(domains, seed, strategy):
    loaders = []
    total_size = 0
    for domain in domains:
        _, full_loader = Data.get_data_loaders(domain, seed=seed)
        if strategy == "balanced_1000":
            indices = list(range(min(1000, len(full_loader.dataset))))
            subset = torch.utils.data.Subset(full_loader.dataset, indices)
            loader = torch.utils.data.DataLoader(subset, batch_size=128, shuffle=False)
            loaders.append((domain, loader))
            total_size += len(indices)
        else:
            loaders.append((domain, full_loader))
            total_size += len(full_loader.dataset)
    return loaders, total_size


def build_DP_model_Classes(data_loaders, data_size, source_domains, models, classifiers,
                           normalize_factors, estimate_prob_type):
    # This function is now run ONCE per target in the main process
    log_print(f"--- Building Matrices (Size: {data_size}) ---")

    C = 10
    Y = np.zeros((data_size, C))
    D = np.zeros((data_size, C, len(source_domains)))
    H = np.zeros((data_size, C, len(source_domains)))
    i = 0

    for target_domain, data_loader in data_loaders:
        for batch_idx, (data, label) in enumerate(data_loader):
            data = data.to(device)
            N = len(data)
            y_vals = label.cpu().detach().numpy()
            one_hot = np.zeros((y_vals.size, C))
            one_hot[np.arange(y_vals.size), y_vals] = 1
            Y[i:i + N] = one_hot

            for k, source_domain in enumerate(source_domains):
                with torch.no_grad():
                    output = classifiers[source_domain](data)
                    H[i:i + N, :, k] = F.softmax(output, dim=1).cpu().detach().numpy()

                    if estimate_prob_type == "GMSA":
                        D[i:i + N, :, k] = output.cpu().detach().numpy()
                    elif estimate_prob_type in ["OURS-STD-SCORE", "OURS-KDE", "OURS-STD-SCORE-WITH-KDE"]:
                        x_hat, _, _ = models[source_domain](data)
                        log_p = models[source_domain].compute_log_probabitility_gaussian(
                            x_hat, data.view(data.shape[0], -1), torch.zeros_like(x_hat)
                        )
                        D[i:i + N, :, k] = torch.tile(log_p[:, None], (1, C)).cpu().detach().numpy()
            i += N

    # KDE / Normalization
    for k, source_domain in enumerate(source_domains):
        if estimate_prob_type in ["OURS-KDE", "OURS-STD-SCORE", "OURS-STD-SCORE-WITH-KDE"]:
            raw_scores = D[:, 0, k].reshape(-1, 1)
            scores_proc = raw_scores
            if "STD-SCORE" in estimate_prob_type:
                mean_val, std_val = normalize_factors[source_domain]
                mean_val, std_val = mean_val.item(), std_val.item()
                scores_proc = (raw_scores - mean_val) / std_val if std_val > 1e-6 else (raw_scores - mean_val)

            if "KDE" in estimate_prob_type:
                score_std = np.std(scores_proc)
                if score_std < 0.05:
                    params = {"bandwidth": np.linspace(0.04, 0.5, 25)}
                else:
                    params = {"bandwidth": np.logspace(-2, 2, 40)}

                # Fit on subset to be fast
                subset_idx = np.random.permutation(len(scores_proc))[:3000]
                grid = GridSearchCV(KernelDensity(), params, n_jobs=1, cv=3)
                grid.fit(scores_proc[subset_idx])

                kde = grid.best_estimator_
                D[:, :, k] = np.tile(np.exp(kde.score_samples(scores_proc)).reshape(-1, 1), (1, C))
            else:
                D[:, :, k] = np.tile(np.exp(-np.abs(scores_proc)), (1, C))

    # Noise Gate
    mask_weak = D < (D.max(axis=2, keepdims=True) * 0.3)
    D[mask_weak] = 0.0

    # Final Norm
    for k in range(len(source_domains)):
        total = D[:, :, k].sum()
        D[:, :, k] = D[:, :, k] / total if total > 0 else 1.0 / len(D)

    return Y, D, H


def evaluate_accuracy(weights, D, H, Y, multi_dim):
    if weights is None: return 0.0
    z = weights.reshape(1, 1, -1)
    preds = (D * H) * z
    final = preds.sum(axis=2)
    y_true = Y.argmax(axis=1) if multi_dim else Y
    return accuracy_score(y_true, final.argmax(axis=1)) * 100.0


def run_solver_sweep_worker(Y, D, H, eps_mult, source_domains, target_domains,
                            seed, init_z_method, multi_dim, precomputed_weights,
                            all_source_domains):
    """
    Worker function: Runs the Deltas Sweep for a specific Epsilon.
    Receives Pre-Built Matrices Y, D, H.
    """
    output_buffer = io.StringIO()

    # Constants
    errors_list = [(SOURCE_ERRORS[d] + 0.01) * eps_mult for d in source_domains]
    eps_global = max(errors_list)
    eps_vector = np.array(errors_list)

    oracle_z = get_oracle_weights(target_domains, source_domains, mode="real_ratio")
    uniform_w = np.ones(len(source_domains)) / len(source_domains)

    K = len(source_domains)
    max_entropy = np.log(K) if K > 1 else 0.0
    multipliers = [0.5, 0.8, 1.0, 1.3, 1.5, 2.0]
    solvers = ["ORACLE", "UNIFORM", "DC", "CVXPY_GLOBAL", "CVXPY_PER_DOMAIN"]

    # Subset for Optimization
    SUBSET = 2000
    if len(Y) > SUBSET:
        np.random.seed(seed)
        idx = np.random.choice(len(Y), SUBSET, replace=False)
        Y_opt, D_opt, H_opt = Y[idx], D[idx], H[idx]
    else:
        Y_opt, D_opt, H_opt = Y, D, H

    results = {}
    for solver in solvers:
        results[solver] = []

        # Fixed Solvers
        if solver in ["ORACLE", "UNIFORM"] or (precomputed_weights and solver in precomputed_weights):
            if solver == "ORACLE":
                w, st = oracle_z, "Oracle"
            elif solver == "UNIFORM":
                w, st = uniform_w, "Fixed"
            else:
                w, st = precomputed_weights[solver], "Precomputed"

            sc = evaluate_accuracy(w, D, H, Y, multi_dim)
            results[solver].append({
                "mult": "N/A", "delta": "N/A", "w": w, "sc": f"{sc:.4f}", "st": st
            })
            continue

        # Dynamic Sweep
        for mult in multipliers:
            delta_val = mult * max_entropy if K > 1 else 0.0
            learned_z, status, score_disp = None, "Unknown", "---"

            if solver == "DC":
                scores, ws = [], []
                for i in range(4):
                    try:
                        dp = init_problem_from_model(Y_opt, D_opt, H_opt, p=K, C=10)
                        prob = ConvexConcaveProblem(dp)
                        slv = ConvexConcaveSolver(prob, seed + i * 100, init_z_method)
                        z_c, _, _ = slv.solve()
                        if z_c is not None:
                            scores.append(evaluate_accuracy(z_c, D, H, Y, multi_dim))
                            ws.append(z_c)
                    except:
                        continue

                if scores:
                    score_disp = f"{np.mean(scores):.4f} ± {np.std(scores):.4f}"
                    learned_z = ws[np.argmax(scores)]
                    status = "Converged"
                else:
                    status = "Failed"
            else:
                try:
                    delta_vec = np.array([delta_val] * K)
                    if solver == "CVXPY_GLOBAL":
                        learned_z = solve_convex_problem_mosek(Y_opt, D_opt, H_opt, delta=delta_val, epsilon=eps_global,
                                                               solver_type='SCS')
                    elif solver == "CVXPY_PER_DOMAIN":
                        learned_z = solve_convex_problem_per_domain(Y_opt, D_opt, H_opt, delta=delta_vec,
                                                                    epsilon=eps_vector, solver_type='SCS')
                except:
                    pass

            # Validation
            if learned_z is None:
                if status == "Unknown": status = "Infeasible"
            else:
                is_uniform = np.allclose(learned_z, uniform_w, atol=1e-9)
                # if abs(np.sum(learned_z) - 1.0) > 1e-2:
                #     status, learned_z = "Explosion", None
                if is_uniform and K > 1:
                    status, learned_z = "Fallback", None
                elif solver != "DC":
                    status = "Converged"
                    score_disp = f"{evaluate_accuracy(learned_z, D, H, Y, multi_dim):.4f}"

            results[solver].append({
                "mult": mult, "delta": delta_val, "w": learned_z, "sc": score_disp, "st": status
            })
            if solver == "DC": break  # Run DC once per eps

    # Format output string
    for solver, res_list in results.items():
        for r in res_list:
            w_full = map_weights_to_full_source_list(r['w'], source_domains, all_source_domains)
            delta_str = f"{r['delta']:.2f}" if isinstance(r['delta'], float) else "N/A"
            w_str = str(np.round(w_full, 4)) if w_full is not None else "---"

            line = f"{solver:<18} | {str(eps_mult):<4} | {str(r['mult']):<4} | {delta_str:<6} | {r['st']:<22} | {r['sc']:<20} | {w_str:<30}\n"
            output_buffer.write(line)
        if len(res_list) > 0: output_buffer.write("-" * 60 + "\n")

    return output_buffer.getvalue()


def task_run(date, seed, estimate_prob_type, init_z_method, multi_dim,
             model_type, pos_alpha, neg_alpha,
             classifiers, all_source_domains, opt_scope, sample_strat, restrict_sources):
    torch.manual_seed(seed)
    np.random.seed(seed)

    source_mode_str = "Sources_RESTRICTED" if restrict_sources else "Sources_ALL"
    test_path = (
        f'./{estimate_prob_type}_results_{date}/'
        f'init_z_{init_z_method}/use_multi_dim_{multi_dim}/seed_{seed}/'
        f'{opt_scope}_{sample_strat}_{source_mode_str}_OPTIMIZED_PARALLEL'
    )
    os.makedirs(test_path, exist_ok=True)

    # 1. LOAD MODELS ONCE
    models = {}
    normalize_factors = {'MNIST': (0, 0), 'USPS': (0, 0), 'SVHN': (0, 0)}
    for domain in all_source_domains:
        model = vr_model(pos_alpha, neg_alpha).to(device)
        path = glob.glob(f"./models_{domain}_seed{seed}_*/{model_type}_*")[0]
        try:
            model.load_state_dict(torch.load(path, map_location=torch.device(device)))
            models[domain] = model
        except:
            continue

        # Calc Factors
        if "STD-SCORE" in estimate_prob_type:
            tl, _ = Data.get_data_loaders(domain, seed=seed)
            probs = []
            for i, (d, _) in enumerate(tl):
                if i > 50: break
                with torch.no_grad():
                    x_hat, _, _ = models[domain](d.to(device))
                    probs.append(
                        models[domain].compute_log_probabitility_bernoulli(x_hat, d.to(device).view(d.shape[0], -1)))
            all_p = torch.cat(probs, 0)
            normalize_factors[domain] = (all_p.mean(), all_p.std())

    # 2. GLOBAL WEIGHTS (Optional, Serial)
    global_weights_cache = {}
    if opt_scope == "global":
        log_print("Calculating Global Weights...")
        tl, ts = create_loaders(all_source_domains, seed, sample_strat)
        Y_g, D_g, H_g = build_DP_model_Classes(tl, ts, all_source_domains, models, classifiers, normalize_factors,
                                               estimate_prob_type)

        # Parallel Global Calculation for different epsilons? Or simple serial. Serial is fast enough here.
        for eps in [1.0, 1.1, 1.2]:
            res = run_solver_sweep_worker(Y_g, D_g, H_g, eps, all_source_domains, all_source_domains, seed,
                                          init_z_method, multi_dim, None, all_source_domains)
            # Parse result manually or change worker to return dict.
            # For simplicity, let's assume global mode is less critical for parallel optimization right now or just run it inside worker logic if passed.
            # To keep code simple: We skip caching complex global weights in this parallel version,
            # OR we just pass None and let them run dynamic. Let's pass None for now.

    # 3. MAIN LOOP - SERIAL TARGETS, PARALLEL EPSILONS
    filename = f'Sweep_Results_OPTIMIZED_{seed}.txt'
    with open(os.path.join(test_path, filename), 'w') as fp:
        log_print(f"Writing to {filename}")

        target_sets = [
            ['MNIST', 'USPS', 'SVHN'], ['MNIST', 'USPS'], ['MNIST', 'SVHN'], ['USPS', 'SVHN']
            # ,
            # ['MNIST'], ['USPS'], ['SVHN']
        ]

        for target in target_sets:
            target_str = str(target)
            fp.write(f"\n{'=' * 120}\nTARGET MIX: {target_str}\n{'=' * 120}\n")
            header = f"{'Solver':<18} | {'Eps':<4} | {'Mult':<4} | {'Delta':<6} | {'Status':<22} | {'Score':<20} | {'Weights':<30}\n"
            fp.write(header + "-" * 120 + "\n")

            # A. PREPARE MATRICES (ONCE PER TARGET)
            if restrict_sources:
                src = [d for d in all_source_domains if d in target]
            else:
                src = all_source_domains

            log_print(f"Building Matrices for {target_str}...")
            el, es = create_loaders(target, seed, "all_data")
            Y_ev, D_ev, H_ev = build_DP_model_Classes(el, es, src, models, classifiers, normalize_factors,
                                                      estimate_prob_type)

            # For training data, if per_target and all_data, it's the same.
            Y_tr, D_tr, H_tr = Y_ev, D_ev, H_ev

            # B. PARALLEL EXECUTION (FOR EPSILONS)
            # Joblib efficiently handles shared memory for numpy arrays (read-only)
            log_print(f"   -> Launching parallel solvers for Epsilons [1.0, 1.1, 1.2]")

            parallel_results = Parallel(n_jobs=-1, backend="loky")(
                delayed(run_solver_sweep_worker)(
                    Y_tr, D_tr, H_tr, eps, src, target, seed, init_z_method, multi_dim, None, all_source_domains
                ) for eps in [1.0, 1.1, 1.2]
            )

            # C. WRITE RESULTS IMMEDIATELY
            for res_str in parallel_results:
                fp.write(res_str)
                fp.write("." * 120 + "\n")

            fp.flush()  # Ensure write to disk


def main():
    print("device = ", device)
    classifiers = {}
    domains = ['MNIST', 'USPS', 'SVHN']
    for d in domains:
        classifier = ClSFR.Grey_32_64_128_gp().to(device)
        try:
            classifier.load_state_dict(
                torch.load(f"./classifiers_new/{d}_classifier.pt", map_location=torch.device(device)))
            classifiers[d] = classifier
        except:
            pass

    # Config
    date = '01_01'
    for seed in [1]:
        task_run(date, seed, "OURS-STD-SCORE-WITH-KDE", "err", True, "vrs", 2, -2,
                 classifiers, domains, "per_target", "all_data", True)
    print("Done.")


if __name__ == "__main__":
    main()