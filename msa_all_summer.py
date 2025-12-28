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

# ייבוא המודולים שלך
from dc import *
import classifier as ClSFR
import matplotlib.pyplot as plt
from vae import *
import data as Data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device = ", device)
os.environ.setdefault("OMP_NUM_THREADS", "1")
torch.set_num_threads(1)


def build_DP_model_Classes(data_loaders, data_size, source_domains, models, classifiers, test_path,
                           normalize_factors, estimate_prob_type):
    """
    גרסה סופית ואופטימלית:
    1. זיהוי אוטומטי של שונות נמוכה (ללא Hardcoding של שמות דומיינים).
    2. אכיפת טווח Bandwidth דינמי.
    3. סינון רעשים (Noise Gate) כדי להבטיח משקולות חדות (Z).
    """
    print(f"\n[DEBUG] --- Starting build_DP_model_Classes (AUTO-OPTIMIZED) ---")

    C = 10  # num_classes
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

            # Ground Truth Formatting
            y_vals = label.cpu().detach().numpy()
            one_hot = np.zeros((y_vals.size, C))
            one_hot[np.arange(y_vals.size), y_vals] = 1
            Y[i:i + N] = one_hot

            for k, source_domain in enumerate(source_domains):
                with torch.no_grad():
                    # --- H(x): Classifier Predictions ---
                    output = classifiers[source_domain](data)
                    norm_output = F.softmax(output, dim=1)
                    H[i:i + N, :, k] = norm_output.cpu().detach().numpy()

                    # --- D(x): Density / Anomaly Scores ---
                    if estimate_prob_type == "GMSA":
                        D[i:i + N, :, k] = output.cpu().detach().numpy()

                    elif estimate_prob_type in ["OURS-STD-SCORE", "OURS-KDE", "OURS-STD-SCORE-WITH-KDE"]:
                        # Calculate Log Probability via VAE
                        x_hat, _, _ = models[source_domain](data)
                        log_p = models[source_domain].compute_log_probabitility_gaussian(
                            x_hat,
                            data.view(data.shape[0], -1),
                            torch.zeros_like(x_hat)
                        )
                        # Tile to (N, C) because score is class-agnostic
                        log_p_tile = torch.tile(log_p[:, None], (1, C))
                        D[i:i + N, :, k] = log_p_tile.cpu().detach().numpy()

            i += N
            if i % 2000 == 0:
                print(f"[{i}/{data_size}] Processing...")

    print(f"[DEBUG] Finished Loading. Applying Density Estimation...")

    # ---------------------------------------------------------
    # PART 2: Optimized Density Estimation (KDE)
    # ---------------------------------------------------------
    for k, source_domain in enumerate(source_domains):

        if estimate_prob_type == "GMSA":
            # GMSA default logic
            params = {"bandwidth": np.logspace(-2, 2, 40)}
            grid = GridSearchCV(KernelDensity(), params, n_jobs=-1)
            data = D[:, :, k]
            shuffled_indices = np.random.permutation(len(data))
            grid.fit(data[shuffled_indices][:2000])
            kde = grid.best_estimator_
            log_density = kde.score_samples(data)
            D[:, :, k] = np.exp(np.tile(log_density[:, None], (1, C)))

        elif estimate_prob_type in ["OURS-KDE", "OURS-STD-SCORE", "OURS-STD-SCORE-WITH-KDE"]:

            # 1. Extract Raw 1D Scores
            raw_scores_1d = D[:, 0, k].reshape(-1, 1)
            scores_to_process = raw_scores_1d

            # 2. Standardization (Safe Mode)
            if "STD-SCORE" in estimate_prob_type:
                log_p_mean, log_p_std = normalize_factors[source_domain]
                std_val = log_p_std.item()
                mean_val = log_p_mean.item()

                if std_val < 1e-6:  # Avoid division by zero
                    scores_to_process = (raw_scores_1d - mean_val)
                else:
                    scores_to_process = (raw_scores_1d - mean_val) / std_val

            # 3. KDE with AUTOMATIC OPTIMIZATION
            if "KDE" in estimate_prob_type:
                print(f"--- Optimizing Bandwidth for {source_domain} ---")

                # A. Check Score Statistics
                score_std = np.std(scores_to_process)
                print(f"    > Score Std Dev: {score_std:.5f}")

                # B. Define Search Space based on Variance (NO LEAKAGE here, purely Source stats)
                # If variance is tiny (< 0.05), we are in a "Needle" scenario (like MNIST).
                # We MUST force a minimum bandwidth to allow generalization.
                if score_std < 0.05:
                    print("    > DETECTED LOW VARIANCE. Forcing Smooth Fit (Safe Mode).")
                    # Search range: 0.05 to 0.5 (Prevents 0.001 overfitting)
                    params = {"bandwidth": np.linspace(0.05, 0.5, 20)}
                else:
                    print("    > Variance is Healthy. Using Standard Precision Search.")
                    # Search range: 0.01 to 100 (Standard)
                    params = {"bandwidth": np.logspace(-2, 2, 40)}

                # C. Run Grid Search (Cross-Validation on Source)
                grid = GridSearchCV(KernelDensity(), params, n_jobs=-1, cv=3)

                # Fit on a subset to save time, but enough to be representative
                shuffled_indices = np.random.permutation(len(scores_to_process))
                data_subset = scores_to_process[shuffled_indices][:3000]

                grid.fit(data_subset)
                best_bw = grid.best_estimator_.bandwidth
                print(f"    > Selected Optimal Bandwidth: {best_bw:.4f}")

                # D. Apply Best Model
                kde = grid.best_estimator_
                log_density = kde.score_samples(scores_to_process)

                # E. Convert back to Probability
                final_1d_values = np.exp(log_density).reshape(-1, 1)

            else:
                # Fallback without KDE
                final_1d_values = np.exp(-np.abs(scores_to_process))

            # Tile back to (N, C)
            D[:, :, k] = np.tile(final_1d_values, (1, C))

    # ---------------------------------------------------------
    # PART 3: Global Noise Gate (The "Winner Takes All" Fix)
    # ---------------------------------------------------------
    # This prevents the solver from averaging between sources (e.g. 33%/33%/33%).
    # We zero out any source probability that is negligible compared to the leader.
    print("[DEBUG] Applying Noise Gate (Thresholding)...")

    # Find max probability per image across all sources
    # D shape: (N, 10, K)
    max_probs_per_pixel = D.max(axis=2, keepdims=True)  # Shape (N, 10, 1)

    # Threshold: If probability is less than 1% of the winner, kill it.
    mask_weak = D < (max_probs_per_pixel * 0.01)
    D[mask_weak] = 0.0

    # ---------------------------------------------------------
    # PART 4: Final Normalization
    # ---------------------------------------------------------
    for k in range(len(source_domains)):
        total_sum = D[:, :, k].sum()
        if total_sum > 0:
            D[:, :, k] = D[:, :, k] / total_sum
        else:
            print(f"[WARNING] Source {source_domain} has 0 sum. Setting uniform.")
            D[:, :, k] = 1.0 / len(D)

    return Y, D, H


def DC_programming(seed, models, classifiers, source_domains, test_path, normalize_factors,
                   estimate_prob_type, init_z_method, multi_dim):
    ''' Calculate the distribution and hypothesis of the data (over the target data) '''
    logging.info("============== Build domain adaptation model ===================")
    data_size = 0
    data_loaders = []
    for k, domain in enumerate(source_domains):
        train_loader, _ = Data.get_data_loaders(domain, seed=seed, num_datapoints=1000)
        data_size += len(train_loader.dataset)
        data_loaders.append((domain, train_loader))

    # Use the OPTIMIZED function
    Y, D, H = build_DP_model_Classes(
        data_loaders, data_size, source_domains,
        models, classifiers, test_path,
        normalize_factors, estimate_prob_type
    )

    DP = init_problem_from_model(Y, D, H, p=len(source_domains), C=10)
    prob = ConvexConcaveProblem(DP)
    solver = ConvexConcaveSolver(prob, seed, init_z_method)
    z_iter, o_iter, err_iter = solver.solve()
    return z_iter


def test_DC_model(seed, models, classifiers, source_domains, target_domains,
                  learned_z, test_path, normalize_factors,
                  estimate_prob_type, multi_dim):
    data_size = 0
    data_loaders = []
    for domain in target_domains:
        _, test_loader = Data.get_data_loaders(domain, seed=seed)
        data_size += len(test_loader.dataset)
        data_loaders.append((domain, test_loader))

    # Use the OPTIMIZED function
    Y, D, H = build_DP_model_Classes(
        data_loaders, data_size, source_domains,
        models, classifiers, test_path,
        normalize_factors, estimate_prob_type
    )

    DP = init_problem_from_model(Y, D, H, p=len(source_domains), C=10)
    prob = ConvexConcaveProblem(DP)
    _, _, _, hz = prob.compute_DzJzKzhz(learned_z)

    if multi_dim:
        Y = Y.argmax(axis=1)
        hz = hz.argmax(axis=1)

    print("Hz samples: ", hz[:20])
    print("Y samples : ", Y[:20])
    print("\n============== Score ===================")
    score = accuracy_score(y_true=Y, y_pred=hz)
    print(score)
    logging.info("score = {}".format(score))
    return score


def run_domain_adaptation(alpha_pos, alpha_neg, vr_model_type, seed, test_path,
                          estimate_prob_type, init_z_method, multi_dim,
                          classifiers, source_domains):
    torch.manual_seed(seed)
    np.random.seed(seed=seed)
    logging_filename = "domain_adaptation.log"
    logging.basicConfig(filename=logging_filename, level=logging.DEBUG)

    target_domains_sets = [
        ['MNIST', 'USPS', 'SVHN'],
        ['MNIST', 'USPS'], ['MNIST', 'SVHN'], ['USPS', 'SVHN'],
        ['MNIST'], ['USPS'], ['SVHN']
    ]

    models = {}
    probabilities = torch.tensor([]).to(device)
    normalize_factors = {'MNIST': (0, 0), 'USPS': (0, 0), 'SVHN': (0, 0)}

    for domain in source_domains:
        model = vr_model(alpha_pos, alpha_neg).to(device)
        # תיקון נתיבים גמיש יותר
        import glob
        pattern = f"./models_{domain}_seed{seed}_*/{vr_model_type}_{alpha_pos}_{alpha_neg}_{domain}_seed{seed}_model.pt"
        files = glob.glob(pattern)
        if not files:
            # Fallback to fixed path if glob fails
            model_path = f"./models_{domain}_seed{seed}_1/{vr_model_type}_{alpha_pos}_{alpha_neg}_{domain}_seed{seed}_model.pt"
        else:
            model_path = files[0]

        print(f"[LOAD] {model_path}")
        try:
            model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
        except FileNotFoundError:
            print(f"CRITICAL: Model file not found for {domain}")
            continue

        models[domain] = model

        # Calculate Normalize Factors (Mean/Std of Source)
        if estimate_prob_type in ["OURS-STD-SCORE", "OURS-STD-SCORE-WITH-KDE"]:
            train_loader, _ = Data.get_data_loaders(domain, seed=seed)
            # Use a subset to save time
            temp_probs = []
            for batch_idx, (data, _) in enumerate(train_loader):
                if batch_idx > 50: break  # Optimization: Don't run full epoch just for stats
                with torch.no_grad():
                    data = data.to(device)
                    x_hat, mu, logstd = models[domain](data)
                    log_p = models[domain].compute_log_probabitility_bernoulli(
                        x_hat, data.view(data.shape[0], -1)
                    )
                    temp_probs.append(log_p)

            all_probs = torch.cat(temp_probs, 0)
            normalize_factors[domain] = (all_probs.mean(), all_probs.std())
            print(f"Factors for {domain}: {normalize_factors[domain]}")

    learned_z = DC_programming(
        seed, models, classifiers, source_domains, test_path,
        normalize_factors, estimate_prob_type, init_z_method, multi_dim
    )

    with open(test_path + r'/DC_accuracy_score_{}.txt'.format(seed), 'w') as fp:
        fp.write(
            '\nz_MNIST = {}\tz_USPS = {}\tz_SVHN = {}\n'
            .format(learned_z[0], learned_z[1], learned_z[2])
        )
        for target_domains in target_domains_sets:
            print(f"--- Testing on Targets: {target_domains} ---")
            score = test_DC_model(
                seed, models, classifiers, source_domains,
                target_domains, learned_z, test_path,
                normalize_factors, estimate_prob_type, multi_dim
            )
            fp.write('\t'.join(target_domains + [str(score * 100), "\n"]))


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

    # Load Classifiers
    for domain in source_domains:
        _, test_loader = Data.get_data_loaders(domain, seed=1)
        classifier = ClSFR.Grey_32_64_128_gp().to(device)
        try:
            classifier.load_state_dict(
                torch.load(f"./classifiers_new/{domain}_classifier.pt",
                           map_location=torch.device(device))
            )
            # accuracy = ClSFR.test(classifier, test_loader) # Optional: Verify accuracy
            classifiers[domain] = classifier
        except FileNotFoundError:
            print(f"Warning: Classifier for {domain} not found.")

    estimate_prob_types = ["OURS-STD-SCORE-WITH-KDE"]
    init_z_methods = ["err"]
    multi_dim_vals = [True]
    date = 'summer_optimized'
    seeds = [1]

    alphas_by_model = {
        "vrs": [(2, -2)],
        "vr": [(2, -1)],
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
                                (date, seed, estimate_prob_type,
                                 init_z_method, multi_dim,
                                 model_type, pos_alpha, neg_alpha)
                            )

    # Run sequentially for debugging, or Parallel for speed
    # Parallel(n_jobs=os.cpu_count(), backend="loky")(
    #     delayed(task_run)(*t, classifiers, source_domains) for t in tasks
    # )

    # Running single task for immediate output verification
    for t in tasks:
        task_run(*t, classifiers, source_domains)


if __name__ == "__main__":
    main()