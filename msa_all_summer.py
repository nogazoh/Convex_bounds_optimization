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
import itertools
from torchvision import models, transforms
import torch.nn as nn

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

# ==========================================
# --- CONFIGURATION SECTION ---
# ==========================================

DATASET_MODE = "OFFICE"

CONFIGS = {
    "DIGITS": {
        "DOMAINS": ['MNIST', 'USPS', 'SVHN'],
        "CLASSES": 10,
        "INPUT_DIM": 2048,
        "SOURCE_ERRORS": {'MNIST': 1.0 - 0.9948, 'USPS': 1.0 - 0.972596, 'SVHN': 1.0 - 0.949716},
        "TEST_SET_SIZES": {'MNIST': 10000, 'USPS': 2007, 'SVHN': 26032}
    },
    "OFFICE": {
        "DOMAINS": ['Art', 'Clipart', 'Product', 'Real World'],
        "CLASSES": 65,
        "INPUT_DIM": 2048,
        "SOURCE_ERRORS": {'Art': 0.1111, 'Clipart': 0.0802, 'Product': 0.0315, 'Real World': 0.0734},
        "TEST_SET_SIZES": {'Art': 490, 'Clipart': 870, 'Product': 880, 'Real World': 870}
    }
}

CURRENT_CFG = CONFIGS[DATASET_MODE]
SOURCE_ERRORS = CURRENT_CFG["SOURCE_ERRORS"]
TEST_SET_SIZES = CURRENT_CFG["TEST_SET_SIZES"]
ALL_DOMAINS_LIST = CURRENT_CFG["DOMAINS"]
NUM_CLASSES = CURRENT_CFG["CLASSES"]
INPUT_DIM = CURRENT_CFG["INPUT_DIM"]

print(f"--- RUNNING IN MODE: {DATASET_MODE} ---")
print(f"--- Domains: {ALL_DOMAINS_LIST} ---")


class FeatureExtractor(nn.Module):
    def __init__(self, original_model):
        super(FeatureExtractor, self).__init__()
        self.backbone = nn.Sequential(*list(original_model.children())[:-1])
        self.head = original_model.fc

    def forward(self, x):
        feats = self.backbone(x)
        feats = torch.flatten(feats, 1)
        logits = self.head(feats)
        return feats, logits


# ==========================================
# --- HELPER: ON-THE-FLY FIX ---
# ==========================================
def fix_batch_office_home(data):
    """
    1. Resizes to 224x224 (ResNet Requirement).
    2. Applies ImageNet Normalization manually on tensor.
    """
    if DATASET_MODE == "OFFICE":
        # 1. Resize if smaller than 224
        if data.shape[-1] < 224:
            data = F.interpolate(data, size=(224, 224), mode='bilinear', align_corners=False)

        # 2. Normalize (ImageNet stats)
        # mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(data.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(data.device)

        # Check if data is already normalized (centered around 0) or raw (0-1)
        # If mean is small (<0.1), it might be normalized already. If mean is ~0.5, it's 0-1.
        if data.mean() > 0.1:
            data = (data - mean) / std

    return data


# ==========================================
# --- LOGIC ---
# ==========================================

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
    if total_samples == 0: total_samples = 1

    for source in source_domains:
        if source in target_domains:
            weights.append(TEST_SET_SIZES.get(source, 0) / total_samples)
        else:
            weights.append(0.0)
    return np.array(weights)


def create_loaders(domains, seed, strategy):
    loaders = []
    total_size = 0
    for domain in domains:
        _, full_loader, _ = Data.get_data_loaders(domain, seed=seed)
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
                           normalize_factors, estimate_prob_type, vae_norm_stats):
    log_print(f"--- Building Matrices (Size: {data_size}) ---")
    C = NUM_CLASSES
    Y = np.zeros((data_size, C))
    D = np.zeros((data_size, C, len(source_domains)))
    H = np.zeros((data_size, C, len(source_domains)))
    i = 0

    for target_domain, data_loader in data_loaders:
        for batch_idx, (data, label) in enumerate(data_loader):
            data = data.to(device)

            # --- FIX: Apply Resize & Normalize ---
            data = fix_batch_office_home(data)
            # -------------------------------------

            N = len(data)
            y_vals = label.cpu().detach().numpy()
            one_hot = np.zeros((y_vals.size, C))
            valid_idx = y_vals < C
            one_hot[np.arange(y_vals.size)[valid_idx], y_vals[valid_idx]] = 1
            Y[i:i + N] = one_hot

            for k, source_domain in enumerate(source_domains):
                with torch.no_grad():
                    if DATASET_MODE == "OFFICE":
                        # Ensure eval mode
                        classifiers[source_domain].eval()
                        features, logits = classifiers[source_domain](data)
                    else:
                        logits = classifiers[source_domain](data)
                        features = data.view(N, -1)

                    H[i:i + N, :, k] = F.softmax(logits, dim=1).cpu().detach().numpy()

                    if estimate_prob_type == "GMSA":
                        D[i:i + N, :, k] = logits.cpu().detach().numpy()
                    elif estimate_prob_type in ["OURS-STD-SCORE", "OURS-KDE", "OURS-STD-SCORE-WITH-KDE"]:
                        vae_input = features
                        if DATASET_MODE == "OFFICE":
                            min_v, max_v = vae_norm_stats[source_domain]
                            vae_input = (features - min_v) / (max_v - min_v + 1e-6)
                            vae_input = torch.clamp(vae_input, 0.0, 1.0)

                        x_hat, _, _ = models[source_domain](vae_input)
                        log_p = models[source_domain].compute_log_probabitility_gaussian(
                            x_hat, vae_input, torch.zeros_like(x_hat)
                        )
                        D[i:i + N, :, k] = torch.tile(log_p[:, None], (1, C)).cpu().detach().numpy()
            i += N

    for k, source_domain in enumerate(source_domains):
        if estimate_prob_type in ["OURS-KDE", "OURS-STD-SCORE", "OURS-STD-SCORE-WITH-KDE"]:
            raw_scores = D[:, 0, k].reshape(-1, 1)
            scores_proc = raw_scores
            if "STD-SCORE" in estimate_prob_type:
                mean_val, std_val = normalize_factors[source_domain]
                mean_val, std_val = mean_val.item(), std_val.item()
                scores_proc = (raw_scores - mean_val) / std_val if std_val > 1e-6 else (raw_scores - mean_val)

            if "KDE" in estimate_prob_type:
                # Dynamic bandwidth
                score_std = np.std(scores_proc)
                if score_std < 1e-9: score_std = 1.0
                params = {"bandwidth": np.linspace(score_std / 50.0, score_std / 2.0, 25)}

                subset_idx = np.random.permutation(len(scores_proc))[:3000]
                grid = GridSearchCV(KernelDensity(), params, n_jobs=1, cv=3)
                grid.fit(scores_proc[subset_idx])
                kde = grid.best_estimator_
                D[:, :, k] = np.tile(np.exp(kde.score_samples(scores_proc)).reshape(-1, 1), (1, C))
            else:
                D[:, :, k] = np.tile(np.exp(-np.abs(scores_proc)), (1, C))

    # --- SCALING ---
    mask_weak = D < (D.max(axis=2, keepdims=True) * 0.3)
    D[mask_weak] = 0.0
    for k in range(len(source_domains)):
        max_val = D[:, :, k].max()
        if max_val > 1e-9:
            D[:, :, k] = D[:, :, k] / max_val

    return Y, D, H


def evaluate_accuracy(weights, D, H, Y, multi_dim):
    if weights is None: return 0.0
    z = weights.reshape(1, 1, -1)
    preds = (D * H) * z
    final = preds.sum(axis=2)
    y_true = Y.argmax(axis=1) if multi_dim else Y
    return accuracy_score(y_true, final.argmax(axis=1)) * 100.0


def debug_matrices(Y, D, H, sources):
    print("\n" + "=" * 40)
    print("   DEBUG: MATRICES SANITY CHECK")
    print("=" * 40)

    N, C, K = D.shape
    print(f"Shape: N={N}, Classes={C}, Sources={K}")

    # 1. Source Accuracies (on this subset)
    y_true = Y.argmax(axis=1)
    print("\n--- 1. Source Accuracies (on this subset) ---")
    for k in range(K):
        preds = H[:, :, k].argmax(axis=1)
        acc = (preds == y_true).mean()
        print(f"   Source {sources[k]}: {acc:.2%}")

    # 2. Density (D) Stats
    print("\n--- 2. Density (D) Stats ---")
    for k in range(K):
        d_flat = D[:, :, k].flatten()
        # Filter zeroes
        d_active = d_flat[d_flat > 1e-6]
        mean_val = d_active.mean() if len(d_active) > 0 else 0.0
        max_val = d_active.max() if len(d_active) > 0 else 0.0
        min_val = d_active.min() if len(d_active) > 0 else 0.0
        print(f"   Source {sources[k]}: Mean={mean_val:.4f}, Max={max_val:.4f}, Min={min_val:.4f}")

    # 3. Model Overlap / Agreement
    if K == 2:
        preds_0 = H[:, :, 0].argmax(axis=1)
        preds_1 = H[:, :, 1].argmax(axis=1)
        agreement = (preds_0 == preds_1).mean()
        print(f"\n--- 3. Model Agreement ---")
        print(f"   Agreement between {sources[0]} and {sources[1]}: {agreement:.2%}")
    print("=" * 40 + "\n")


def run_baselines(Y, D, H, source_domains, target_domains, seed, init_z_method, multi_dim, all_source_domains):
    output_buffer = io.StringIO()

    # 1. Oracle & Uniform
    oracle_z = get_oracle_weights(target_domains, source_domains, mode="real_ratio")
    uniform_w = np.ones(len(source_domains)) / len(source_domains)

    # Evaluate Oracle
    sc = evaluate_accuracy(oracle_z, D, H, Y, multi_dim)
    w_full = map_weights_to_full_source_list(oracle_z, source_domains, all_source_domains)

    output_buffer.write(
        f"{'ORACLE':<18} | {'N/A':<4} | {'N/A':<4} | {'N/A':<6} | {'Oracle':<22} | {sc:<20.4f} | {str(np.round(w_full, 4)):<30}\n")
    output_buffer.write("-" * 120 + "\n")
    print(f"    >>> [Baseline] ORACLE  | Acc: {sc:.2f}%")

    # Evaluate Uniform
    sc = evaluate_accuracy(uniform_w, D, H, Y, multi_dim)
    w_full = map_weights_to_full_source_list(uniform_w, source_domains, all_source_domains)

    output_buffer.write(
        f"{'UNIFORM':<18} | {'N/A':<4} | {'N/A':<4} | {'N/A':<6} | {'Fixed':<22} | {sc:<20.4f} | {str(np.round(w_full, 4)):<30}\n")
    output_buffer.write("-" * 120 + "\n")
    print(f"    >>> [Baseline] UNIFORM | Acc: {sc:.2f}%")

    # 2. DC Solver (Run once as a baseline)
    scores, ws = [], []
    K = len(source_domains)
    C_count = NUM_CLASSES

    for i in range(4):
        try:
            dp = init_problem_from_model(Y, D, H, p=K, C=C_count)
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
        print(f"    >>> [Baseline] DC      | Acc: {evaluate_accuracy(learned_z, D, H, Y, multi_dim):.2f}%")
    else:
        status = "Failed"
        score_disp = "---"
        learned_z = None
        print(f"    >>> [Baseline] DC      | Failed")

    w_full = map_weights_to_full_source_list(learned_z, source_domains, all_source_domains)
    w_str = str(np.round(w_full, 4)) if w_full is not None else "---"

    output_buffer.write(
        f"{'DC':<18} | {'N/A':<4} | {'N/A':<4} | {'N/A':<6} | {status:<22} | {score_disp:<20} | {w_str:<30}\n")
    output_buffer.write("-" * 120 + "\n")

    return output_buffer.getvalue()


def run_solver_sweep_worker(Y, D, H, eps_mult, source_domains, target_domains,
                            seed, init_z_method, multi_dim, precomputed_weights,
                            all_source_domains):
    output_buffer = io.StringIO()

    errors_list = [(SOURCE_ERRORS.get(d, 0.1) + 0.01) * eps_mult for d in source_domains]
    eps_global = max(errors_list)
    eps_vector = np.array(errors_list)

    K = len(source_domains)
    max_entropy = np.log(K) if K > 1 else 0.0

    multipliers = [1.0, 1.2]
    solvers = ["CVXPY_GLOBAL", "CVXPY_PER_DOMAIN"]
    SUBSET = 3000

    if len(Y) > SUBSET:
        np.random.seed(seed)
        idx = np.random.choice(len(Y), SUBSET, replace=False)
        Y_opt, D_opt, H_opt = Y[idx], D[idx], H[idx]
    else:
        Y_opt, D_opt, H_opt = Y, D, H

    results = {}
    for solver in solvers:
        results[solver] = []

        for mult in multipliers:
            delta_val = mult * max_entropy if K > 1 else 0.0
            learned_z, status, score_disp = None, "Unknown", "---"

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

            if learned_z is None:
                status = "Failed"
                score_disp = "---"
                print(f"    >>> [Worker] Eps={eps_mult} | {solver} | Mult={mult} | FAILED")
            else:
                status = "Converged"
                acc_val = evaluate_accuracy(learned_z, D, H, Y, multi_dim)
                score_disp = f"{acc_val:.4f}"
                print(f"    >>> [Worker] Eps={eps_mult} | {solver} | Mult={mult} | Acc: {acc_val:.2f}%")

            results[solver].append({
                "mult": mult, "delta": delta_val, "w": learned_z, "sc": score_disp, "st": status
            })

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
        f'./{estimate_prob_type}_results_{DATASET_MODE}_{date}/'
        f'init_z_{init_z_method}/use_multi_dim_{multi_dim}/seed_{seed}/'
        f'{opt_scope}_{sample_strat}_{source_mode_str}_OPTIMIZED_PARALLEL'
    )
    os.makedirs(test_path, exist_ok=True)

    # 1. LOAD MODELS & STATS
    models = {}
    normalize_factors = {d: (0, 0) for d in all_source_domains}
    vae_norm_stats = {d: (0, 1) for d in all_source_domains}

    for domain in all_source_domains:
        model = vr_model(INPUT_DIM, pos_alpha, neg_alpha).to(device)
        path_pattern = f"./models_{domain}_seed{seed}_*/{model_type}_*"
        paths = glob.glob(path_pattern)

        if not paths:
            print(f"[WARNING] No model found for {domain} with pattern {path_pattern}")
            continue

        try:
            model.load_state_dict(torch.load(paths[0], map_location=torch.device(device)))
            models[domain] = model
            model.eval()  # Ensure eval
        except Exception as e:
            print(f"[ERROR] Loading model for {domain}: {e}")
            continue

        if "STD-SCORE" in estimate_prob_type:
            train_loader, _, _ = Data.get_data_loaders(domain, seed=seed)
            extractor = classifiers[domain]
            all_feats = []

            if DATASET_MODE == "OFFICE":
                with torch.no_grad():
                    for i, (imgs, _) in enumerate(train_loader):
                        if i > 20: break
                        imgs = imgs.to(device)

                        # --- FIX: Apply Resize & Normalize ---
                        imgs = fix_batch_office_home(imgs)
                        # -------------------------------------

                        feats, _ = extractor(imgs)
                        all_feats.append(feats)

                cat_feats = torch.cat(all_feats, dim=0)
                min_v = cat_feats.min().to(device)
                max_v = cat_feats.max().to(device)
                vae_norm_stats[domain] = (min_v, max_v)

            probs = []
            for i, (imgs, _) in enumerate(train_loader):
                if i > 50: break
                with torch.no_grad():
                    imgs = imgs.to(device)

                    if DATASET_MODE == "OFFICE":
                        # --- FIX: Apply Resize & Normalize ---
                        imgs = fix_batch_office_home(imgs)
                        # -------------------------------------

                        feats, _ = extractor(imgs)
                        min_v, max_v = vae_norm_stats[domain]
                        vae_in = (feats - min_v) / (max_v - min_v + 1e-6)
                        vae_in = torch.clamp(vae_in, 0.0, 1.0)
                    else:
                        vae_in = imgs.view(imgs.shape[0], -1)

                    x_hat, _, _ = models[domain](vae_in)
                    probs.append(models[domain].compute_log_probabitility_bernoulli(x_hat, vae_in))

            if len(probs) > 0:
                all_p = torch.cat(probs, 0)
                normalize_factors[domain] = (all_p.mean(), all_p.std())

    # 3. MAIN LOOP
    filename = f'Sweep_Results_OPTIMIZED_{seed}.txt'
    full_file_path = os.path.join(test_path, filename)

    completed_mixes = []
    if os.path.exists(full_file_path):
        print(f"[RESUME] Scanning existing file: {full_file_path}")
        with open(full_file_path, 'r') as f_read:
            for line in f_read:
                if "TARGET MIX:" in line:
                    clean_mix = line.split("TARGET MIX:")[1].strip()
                    completed_mixes.append(clean_mix)
        print(f"[RESUME] Found {len(completed_mixes)} completed tasks. Skipping them.")

    with open(full_file_path, 'a') as fp:
        log_print(f"Appending to {filename}")
        target_sets = []
        n_domains = len(all_source_domains)
        min_r = 2
        max_r = n_domains
        for r in range(min_r, max_r + 1):
            for subset in itertools.combinations(all_source_domains, r):
                target_sets.append(list(subset))

        for target in target_sets:
            target_str = str(target)
            if target_str in completed_mixes:
                print(f"[SKIP] Target {target_str} already done.")
                continue

            fp.write(f"\n{'=' * 120}\nTARGET MIX: {target_str}\n{'=' * 120}\n")
            header = f"{'Solver':<18} | {'Eps':<4} | {'Mult':<4} | {'Delta':<6} | {'Status':<22} | {'Score':<20} | {'Weights':<30}\n"
            fp.write(header + "-" * 120 + "\n")
            fp.flush()

            if restrict_sources:
                src = [d for d in all_source_domains if d in target]
            else:
                src = all_source_domains

            log_print(f"Building Matrices for {target_str} using sources: {src}")
            el, es = create_loaders(target, seed, "all_data")

            try:
                Y_ev, D_ev, H_ev = build_DP_model_Classes(el, es, src, models, classifiers, normalize_factors,
                                                          estimate_prob_type, vae_norm_stats)
            except Exception as e:
                log_print(f"Error building matrices for {target}: {e}")
                import traceback
                traceback.print_exc()
                continue

            Y_tr, D_tr, H_tr = Y_ev, D_ev, H_ev
            debug_matrices(Y_tr, D_tr, H_tr, src)

            log_print("    -> Calculating Baselines (Oracle, Uniform, DC)...")
            baseline_str = run_baselines(Y_tr, D_tr, H_tr, src, target, seed, init_z_method, multi_dim,
                                         all_source_domains)
            fp.write(baseline_str)
            fp.flush()

            log_print(f"    -> Launching parallel solvers for Epsilons [1.0, 1.1, 1.2]")
            parallel_results = Parallel(n_jobs=-1, backend="loky")(
                delayed(run_solver_sweep_worker)(
                    Y_tr, D_tr, H_tr, eps, src, target, seed, init_z_method, multi_dim, None, all_source_domains
                ) for eps in [1.0, 1.1, 1.2]
            )

            for res_str in parallel_results:
                fp.write(res_str)
                fp.write("." * 120 + "\n")
            fp.flush()


def main():
    print(f"✅ Runtime Device: {device}")
    classifiers = {}
    domains = ALL_DOMAINS_LIST

    print(f"Loading classifiers for {DATASET_MODE} (Optimized)...")

    for d in domains:
        try:
            path = f"./classifiers_new/{d}_classifier.pt"
            if not os.path.exists(path):
                print(f"❌ [ERROR] File not found: {path}")
                continue

            if DATASET_MODE == "DIGITS":
                classifier = ClSFR.Grey_32_64_128_gp().to(device)
                state = torch.load(path, map_location=device)
                classifier.load_state_dict(state)
            elif DATASET_MODE == "OFFICE":
                full_model = models.resnet50(weights=None)
                full_model.fc = nn.Linear(full_model.fc.in_features, NUM_CLASSES)

                # Optimized Load
                state_dict = torch.load(path, map_location=device)
                full_model.load_state_dict(state_dict)
                classifier = FeatureExtractor(full_model).to(device)

            # --- CRITICAL SAFETY ---
            classifier.eval()
            for param in classifier.parameters():
                param.requires_grad = False

            classifiers[d] = classifier
            print(f"✅ Loaded {d}")
        except Exception as e:
            print(f"[WARNING] Could not load classifier for {d}: {e}")

    if len(classifiers) == 0:
        print("ERROR: No classifiers loaded.")
        return

    date = '03_01_FIXED'
    for seed in [1]:
        print(f"\n--- Starting Run for Seed {seed} ---")
        task_run(date, seed, "OURS-STD-SCORE-WITH-KDE", "err", True, "vrs", 2, -2,
                 classifiers, domains, "per_target", "all_data", True)
    print("Done.")


if __name__ == "__main__":
    main()