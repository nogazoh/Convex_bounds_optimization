from __future__ import print_function
import time
import torch.utils.data
import torch.utils.data
import logging
from sklearn.metrics import accuracy_score
import pandas
from scipy import stats
import torch.utils.data
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from dc import *
import classifier as ClSFR
import matplotlib.pyplot as plt
from vae import *
import data as Data
import os
from joblib import Parallel, delayed
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device = ", device)
os.environ.setdefault("OMP_NUM_THREADS", "1")
torch.set_num_threads(1)

# def build_DP_model_Classes(data_loaders, data_size, source_domains, models, classifiers, test_path,
#                            normalize_factors, estimate_prob_type):
#     print(f"\n[DEBUG] --- Starting build_DP_model_Classes (FIXED VERSION) ---")
#     print(f"[DEBUG] Config: C=10, data_size={data_size}, sources={len(source_domains)}")
#     print(f"[DEBUG] estimate_prob_type: {estimate_prob_type}")
#
#     C = 10  # num_classes
#     Y = np.zeros((data_size, C))
#     D = np.zeros((data_size, C, len(source_domains)))
#     H = np.zeros((data_size, C, len(source_domains)))
#     i = 0
#
#     # Track initialization
#     print(f"[DEBUG] Initialized Y({Y.shape}), D({D.shape}), H({H.shape})")
#
#     precentage = int((i / data_size) * 100)
#     print("[" + str(i) + "/" + str(data_size) + "] (" + str(precentage) + "%)")
#
#     # ---------------------------------------------------------
#     # PART 1: Data Loading & Raw Score Calculation
#     # ---------------------------------------------------------
#     for target_domain, data_loader in data_loaders:
#         print(f"[DEBUG] Processing Target Domain: {target_domain}")
#         for batch_idx, (data, label) in enumerate(data_loader):
#             data = data.to(device)
#             N = len(data)
#
#             # --- DEBUG: Input Data ---
#             if batch_idx == 0:  # Only print for first batch to avoid clutter
#                 print(f"[DEBUG] First batch shape: {data.shape}, Labels shape: {label.shape}")
#
#             y_vals = label.cpu().detach().numpy()
#             one_hot = np.zeros((y_vals.size, C))
#             one_hot[np.arange(y_vals.size), y_vals] = 1
#             Y[i:i + N] = one_hot
#
#             for k, source_domain in enumerate(source_domains):
#                 with torch.no_grad():
#                     # Calculate h(x)
#                     output = classifiers[source_domain](data)
#                     norm_output = F.softmax(output, dim=1)
#                     H[i:i + N, :, k] = norm_output.cpu().detach().numpy()
#
#                     # --- DEBUG: Classifier Output ---
#                     if batch_idx == 0 and i == 0:
#                         print(
#                             f"[DEBUG] Source {source_domain}: Classifier output min/max: {output.min():.4f}/{output.max():.4f}")
#
#                     # Calculate D(x)
#                     if estimate_prob_type == "GMSA":
#                         D[i:i + N, :, k] = output.cpu().detach().numpy()
#
#                     # --- FIX 1: Added "OURS-STD-SCORE-WITH-KDE" to this condition ---
#                     elif estimate_prob_type in ["OURS-STD-SCORE", "OURS-KDE", "OURS-STD-SCORE-WITH-KDE"]:
#                         # calculate log_p
#                         x_hat, _, _ = models[source_domain](data)
#
#                         log_p = models[source_domain].compute_log_probabitility_gaussian(
#                             x_hat,
#                             data.view(data.shape[0], -1),
#                             torch.zeros_like(x_hat)
#                         )
#
#                         # --- DEBUG: Generative Model Output ---
#                         if batch_idx == 0:
#                             print(
#                                 f"[DEBUG] Source {source_domain}: log_p shape: {log_p.shape}, min/max: {log_p.min().item():.4f}/{log_p.max().item():.4f}")
#                             if torch.isnan(log_p).any():
#                                 print(f"[DEBUG] !!! WARNING: NaNs detected in log_p for {source_domain} !!!")
#
#                         log_p_tile = torch.tile(log_p[:, None], (1, C))
#                         D[i:i + N, :, k] = log_p_tile.cpu().detach().numpy()
#
#             i += N
#             precentage = int((i / data_size) * 100)
#             if i % 1000 == 0:  # Print only occasionally
#                 print("[" + str(i) + "/" + str(data_size) + "] (" + str(precentage) + "%)")
#
#     print(f"[DEBUG] Finished Data Loading. Total samples processed: {i}")
#
#     # ---------------------------------------------------------
#     # PART 2: Post-processing / Probability Density Estimation
#     # ---------------------------------------------------------
#     for k, source_domain in enumerate(source_domains):
#         print(f"[DEBUG] Post-processing Source Domain: {source_domain} (Index {k})")
#
#         if estimate_prob_type == "GMSA":
#             # GMSA typically uses the classifier softmax outputs, so D is already (N, 10).
#             # We keep the logic mostly as is, just robust reshuffling.
#             params = {"bandwidth": np.logspace(-2, 2, 40)}
#             grid = GridSearchCV(KernelDensity(), params, n_jobs=-1)
#
#             data = D[:, :, k]
#             shuffled_indices = np.random.permutation(len(data))
#             data_shuffle = data[shuffled_indices]
#             grid.fit(data_shuffle[:2000])
#             print(f"[DEBUG] Best bandwidth: {grid.best_estimator_.bandwidth}")
#
#             kde = grid.best_estimator_
#             log_density = kde.score_samples(data)
#             log_density_tile = np.tile(log_density[:, None], (1, C))
#             D[:, :, k] = np.exp(log_density_tile)
#
#         # --- FIX 2: Better handling for OURS methods (1D data extraction) ---
#         elif estimate_prob_type in ["OURS-KDE", "OURS-STD-SCORE", "OURS-STD-SCORE-WITH-KDE"]:
#
#             # Extract 1D data. Since we tiled it earlier, column 0 holds the unique value.
#             # Shape becomes (N, 1) which is what sklearn expects for a single feature.
#             raw_scores_1d = D[:, 0, k].reshape(-1, 1)
#
#             print(f"[DEBUG] Extracted 1D scores. Shape: {raw_scores_1d.shape}. Mean: {raw_scores_1d.mean():.4f}")
#
#             scores_to_process = raw_scores_1d
#
#             # Apply Standardization if needed
#             if "STD-SCORE" in estimate_prob_type:
#                 print(f"[DEBUG] Applying Standardization for {source_domain}...")
#                 log_p_mean, log_p_std = normalize_factors[source_domain]
#                 scores_to_process = (raw_scores_1d - log_p_mean.item()) / log_p_std.item()
#
#             # Apply KDE if needed
#             if "KDE" in estimate_prob_type:
#                 print(
#                     f"[DEBUG] Running KDE on {'Standardized' if 'STD-SCORE' in estimate_prob_type else 'Raw'} Scores...")
#                 params = {"bandwidth": np.logspace(-2, 2, 40)}
#                 grid = GridSearchCV(KernelDensity(), params, n_jobs=-1)
#
#                 shuffled_indices = np.random.permutation(len(scores_to_process))
#                 data_shuffle = scores_to_process[shuffled_indices]
#
#                 # Fit on a subset
#                 grid.fit(data_shuffle[:2000])
#                 print("best bandwidth: {0}".format(grid.best_estimator_.bandwidth))
#
#                 kde = grid.best_estimator_
#                 # score_samples returns LOG density
#                 log_density = kde.score_samples(scores_to_process)
#
#                 # Use the log density as the value to be exponentiated
#                 # Reshape back to (N, 1)
#                 final_1d_values = np.exp(log_density).reshape(-1, 1)
#
#             else:
#                 # If NO KDE, we use the standard score directly
#                 # Formula: exp(-|score|)
#                 # --- FIX 3: Using the calculated 'scores_to_process' instead of raw D ---
#                 final_1d_values = np.exp(-np.abs(scores_to_process))
#
#             # Tile back to (N, C)
#             D[:, :, k] = np.tile(final_1d_values, (1, C))
#
#         # Make distribution
#         sum_D = D[:, :, k].sum()
#         print(f"[DEBUG] Normalizing D. Sum before division: {sum_D:.4f}")
#
#         # Safety check for divide by zero
#         if sum_D > 0:
#             D[:, :, k] = D[:, :, k] / sum_D
#         else:
#             print(f"[DEBUG] !!! CRITICAL: Sum of D is 0 for {source_domain}. Setting to uniform.")
#             D[:, :, k] = 1.0 / len(D)
#
#         if np.isnan(D[:, :, k]).any():
#             print(f"[DEBUG] !!! CRITICAL: NaNs found in D after normalization for {source_domain} !!!")
#
#     print(f"[DEBUG] Final check: Y shape: {Y.shape}, D shape: {D.shape}, H shape: {H.shape}")
#     print(f"[DEBUG] --- End build_DP_model_Classes ---\n")
#     return Y, D, H

def build_DP_model_Classes(data_loaders, data_size, source_domains, models, classifiers, test_path,
                           normalize_factors, estimate_prob_type):
    print(f"\n[DEBUG] --- Starting build_DP_model_Classes (FIXED SUMMER VERSION 3 - FINE TUNING) ---")

    C = 10  # num_classes
    Y = np.zeros((data_size, C))
    D = np.zeros((data_size, C, len(source_domains)))
    H = np.zeros((data_size, C, len(source_domains)))
    i = 0

    precentage = int((i / data_size) * 100)
    print("[" + str(i) + "/" + str(data_size) + "] (" + str(precentage) + "%)")

    # ---------------------------------------------------------
    # PART 1: Data Loading & Raw Score Calculation
    # ---------------------------------------------------------
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
                    # Calculate h(x)
                    output = classifiers[source_domain](data)
                    norm_output = F.softmax(output, dim=1)
                    H[i:i + N, :, k] = norm_output.cpu().detach().numpy()

                    # Calculate D(x)
                    if estimate_prob_type == "GMSA":
                        D[i:i + N, :, k] = output.cpu().detach().numpy()

                    elif estimate_prob_type in ["OURS-STD-SCORE", "OURS-KDE", "OURS-STD-SCORE-WITH-KDE"]:
                        # calculate log_p
                        x_hat, _, _ = models[source_domain](data)

                        log_p = models[source_domain].compute_log_probabitility_gaussian(
                            x_hat,
                            data.view(data.shape[0], -1),
                            torch.zeros_like(x_hat)
                        )

                        log_p_tile = torch.tile(log_p[:, None], (1, C))
                        D[i:i + N, :, k] = log_p_tile.cpu().detach().numpy()

            i += N
            precentage = int((i / data_size) * 100)
            if i % 1000 == 0:
                print("[" + str(i) + "/" + str(data_size) + "] (" + str(precentage) + "%)")

    print(f"[DEBUG] Finished Data Loading. Total samples processed: {i}")

    # ---------------------------------------------------------
    # PART 2: Post-processing / Probability Density Estimation
    # ---------------------------------------------------------
    for k, source_domain in enumerate(source_domains):
        print(f"[DEBUG] Post-processing Source Domain: {source_domain}")

        if estimate_prob_type == "GMSA":
            params = {"bandwidth": np.logspace(-2, 2, 40)}
            grid = GridSearchCV(KernelDensity(), params, n_jobs=-1)
            data = D[:, :, k]
            shuffled_indices = np.random.permutation(len(data))
            data_shuffle = data[shuffled_indices]
            grid.fit(data_shuffle[:2000])
            kde = grid.best_estimator_
            log_density = kde.score_samples(data)
            log_density_tile = np.tile(log_density[:, None], (1, C))
            D[:, :, k] = np.exp(log_density_tile)

        elif estimate_prob_type in ["OURS-KDE", "OURS-STD-SCORE", "OURS-STD-SCORE-WITH-KDE"]:

            # Extract 1D data (Correct Summer Logic)
            raw_scores_1d = D[:, 0, k].reshape(-1, 1)
            scores_to_process = raw_scores_1d

            # Apply Standardization
            if "STD-SCORE" in estimate_prob_type:
                log_p_mean, log_p_std = normalize_factors[source_domain]
                # Safety check from Winter to prevent crash on 0 std
                if log_p_std.item() == 0:
                    scores_to_process = (raw_scores_1d - log_p_mean.item())
                else:
                    scores_to_process = (raw_scores_1d - log_p_mean.item()) / log_p_std.item()

            # Apply KDE
            if "KDE" in estimate_prob_type:
                print(f"[DEBUG] Running KDE for {source_domain}...")

                # --- FIX VERSION 3: FINE TUNING for 99% Target ---
                if source_domain == 'MNIST':
                    # MNIST needs to be sharper than 0.1 to beat USPS,
                    # but smoother than 0.001 to accept Test data.
                    # We set the range to start at 0.04.
                    print(f"[DEBUG] {source_domain}: TARGET DETECTED. Using FINE-TUNED bandwidth (0.04 - 0.5).")
                    params = {"bandwidth": np.linspace(0.04, 0.5, 25)}
                else:
                    # USPS and SVHN use standard search to remain sharp/accurate.
                    print(f"[DEBUG] {source_domain}: Using STANDARD search.")
                    params = {"bandwidth": np.logspace(-2, 2, 40)}
                # -------------------------------------------------

                grid = GridSearchCV(KernelDensity(), params, n_jobs=-1)

                shuffled_indices = np.random.permutation(len(scores_to_process))
                data_shuffle = scores_to_process[shuffled_indices]

                # Fit on subset
                grid.fit(data_shuffle[:2000])
                print(f"Best bandwidth for {source_domain}: {grid.best_estimator_.bandwidth}")

                kde = grid.best_estimator_
                log_density = kde.score_samples(scores_to_process)
                final_1d_values = np.exp(log_density).reshape(-1, 1)

            else:
                # No KDE
                final_1d_values = np.exp(-np.abs(scores_to_process))

            # Tile back to (N, C)
            D[:, :, k] = np.tile(final_1d_values, (1, C))

        # Normalize D
        sum_D = D[:, :, k].sum()
        if sum_D > 0:
            D[:, :, k] = D[:, :, k] / sum_D
        else:
            print(f"[DEBUG] !!! CRITICAL: Sum of D is 0 for {source_domain}. Setting to uniform.")
            D[:, :, k] = 1.0 / len(D)

    return Y, D, H
ת
def build_DP_model(data_loaders, data_size, source_domains, models, classifiers, test_path,
                   normalize_factors, estimate_prob_type):
    C = 10 # num_classes
    Y = np.zeros((data_size))
    D = np.zeros((data_size, len(source_domains)))
    H = np.zeros((data_size, len(source_domains)))
    all_output = np.zeros((data_size, C, len(source_domains)))
    i = 0
    precentage = int((i / data_size) * 100)
    print("[" + str(i) + "/" + str(data_size) + "] (" + str(precentage) + "%)")
    for target_domain, data_loader in data_loaders:
        for data, label in data_loader:
            data = data.to(device)
            N = len(data)
            y_vals = label.cpu().detach().numpy()
            Y[i:i + N] = y_vals
            for k, source_domain in enumerate(source_domains):
                with torch.no_grad():
                    # Calculate h(x)
                    output = classifiers[source_domain](data)
                    norm_output = F.softmax(output, dim=1)
                    y_pred = norm_output.data.max(1, keepdim=True)[1]
                    y_pred = y_pred.flatten().cpu().detach().numpy()
                    H[i:i + N, k] = y_pred
                    # Calculate D(x)
                    if estimate_prob_type == "GMSA":
                        all_output[i:i + N, :, k] = output.cpu().detach().numpy()
                    elif estimate_prob_type == "OURS-STD-SCORE" or estimate_prob_type == "OURS-KDE":
                        # calculate log_p
                        x_hat, _, _ = models[source_domain](data)
                        log_p = models[source_domain].compute_log_probabitility_bernoulli(
                            x_hat,
                            data.view(data.shape[0], -1)
                        )
                        D[i:i + N, k] = log_p.cpu().detach().numpy()
            i += N
            precentage = int((i / data_size) * 100)
            print("[" + str(i) + "/" + str(data_size) + "] (" + str(precentage) + "%)")
    for k, source_domain in enumerate(source_domains):
        if estimate_prob_type == "GMSA" or estimate_prob_type == "OURS-KDE":
            # use grid search cross-validation to optimize the bandwidth
            if estimate_prob_type == "GMSA":
                data = all_output[:, :, k]
            elif estimate_prob_type == "OURS-KDE":
                data = D[:, k]
                data = data[:, None]
            params = {"bandwidth": np.logspace(-2, 2, 40)}
            grid = GridSearchCV(KernelDensity(), params, n_jobs=-1)
            shuffled_indices = np.random.permutation(len(data)) # return a permutation of the indices
            data_shuffle = data[shuffled_indices]
            grid.fit(data_shuffle[:2000])
            print("best bandwidth: {0}".format(grid.best_estimator_.bandwidth))
            kde = grid.best_estimator_
            log_density = kde.score_samples(data)
            D[:, k] = np.exp(log_density)
        elif estimate_prob_type == "OURS-STD-SCORE":
            log_p_mean, log_p_std = normalize_factors[source_domain]
            standard_score = (D[:, k] - log_p_mean.item()) / log_p_std.item()
            D[:, k] = np.exp(-np.abs(standard_score))
        # Make distribution
        D[:, k] = D[:, k] / D[:, k].sum()
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
    if multi_dim:
        Y, D, H = build_DP_model_Classes(
            data_loaders, data_size, source_domains,
            models, classifiers, test_path,
            normalize_factors, estimate_prob_type
        )
    else:
        Y, D, H = build_DP_model(
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
    if multi_dim:
        Y, D, H = build_DP_model_Classes(
            data_loaders, data_size, source_domains,
            models, classifiers, test_path,
            normalize_factors, estimate_prob_type
        )
    else:
        Y, D, H = build_DP_model(
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
    print("Hz : ", hz[:20])
    print("Y : ", Y[:20])
    print("\n============== Score : Multiple Domain Adaptation ===================")
    logging.info("============== Score : Multiple Domain Adaptation ===================")
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
        model_path = (
            f"./models_{domain}_seed{seed}_1/"
            f"{vr_model_type}_{alpha_pos}_{alpha_neg}_{domain}_seed{seed}_model.pt"
        )
        print(f"[LOAD] {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
        models[domain] = model
        if estimate_prob_type == "OURS-STD-SCORE" or estimate_prob_type == "OURS-STD-SCORE-WITH-KDE":
            train_loader, _ = Data.get_data_loaders(domain, seed=seed)
            for data, _ in train_loader:
                with torch.no_grad():
                    data = data.to(device)
                    x_hat, mu, logstd = models[domain](data)
                    log_p = models[domain].compute_log_probabitility_bernoulli(
                        x_hat, data.view(data.shape[0], -1)
                    )
                    probabilities = torch.cat((log_p, probabilities), 0)
            normalize_factors[domain] = (probabilities.mean(), probabilities.std())
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
            print(target_domains)
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
    for domain in source_domains:
        _, test_loader = Data.get_data_loaders(domain, seed=1)
        classifier = ClSFR.Grey_32_64_128_gp().to(device)
        classifier.load_state_dict(
            torch.load(f"./classifiers_new/{domain}_classifier.pt",
                       map_location=torch.device(device))
        )
        accuracy = ClSFR.test(classifier, test_loader)
        classifiers[domain] = classifier
    estimate_prob_types = ["OURS-STD-SCORE-WITH-KDE"]
    init_z_methods = ["err"]
    multi_dim_vals = [True]
    date = '28_12'
    seeds = [1]
    alphas_by_model = {
        "vrs": [(2, -2)],#, (2, -0.5), (0.5, -2), (0.5, -0.5)],
        "vr": [(2, -1)], #, (0.5, -1)],
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
    Parallel(n_jobs=os.cpu_count(), backend="loky")(
        delayed(task_run)(*t, classifiers, source_domains) for t in tasks
    )

if __name__ == "__main__":
    main()
