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
from torchvision import models, transforms, datasets
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import traceback

from dc import *
from vae import *
import data as Data
import classifier as ClSFR

# --- SOLVERS ---
try:
    from old_cvxpy_solver import solve_convex_problem_mosek
    from cvxpy_solver_per_domain import solve_convex_problem_per_domain
except ImportError:
    pass
if "solve_convex_problem_mosek" not in globals():
    raise ImportError("solve_convex_problem_mosek not imported. Check old_cvxpy_solver import/path.")

from cvxpy_3_21 import solve_convex_problem_smoothed_kl_321
from cvxpy_3_22 import solve_convex_problem_domain_anchored_smoothed
from cvxpy_3_23 import solve_convex_problem_smoothed_original_p

from helpers import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "1")
torch.set_num_threads(1)

# ==========================================
# --- CONFIGURATION ---
# ==========================================
DATASET_MODE = "DIGITS"# "DIGITS" #"OFFICE224" #"OFFICE31"

CONFIGS = {
    "DIGITS": {
        "DOMAINS": ['MNIST', 'USPS', 'SVHN'],
        "CLASSES": 10,
        "INPUT_DIM": 784,
        "SOURCE_ERRORS": {'MNIST': 0.005, 'USPS': 0.027, 'SVHN': 0.05},
        "TEST_SET_SIZES": {'MNIST': 10000, 'USPS': 2007, 'SVHN': 26032},
        "D_PRECOMP_PATH":"/data/nogaz/Convex_bounds_optimization/LatentFlow_Pixel_Experiments/results/D_Matrix_FINAL_GMM_Soft_DIGITS.npy"
    },
    "OFFICE": {  # for office-home after resizing by 0.5
        "DOMAINS": ['Art', 'Clipart', 'Product', 'Real World'],
        "CLASSES": 65,
        "INPUT_DIM": 2048,
        "SOURCE_ERRORS": {'Art': 0.11, 'Clipart': 0.08, 'Product': 0.03, 'Real World': 0.07},
        "TEST_SET_SIZES": {'Art': 490, 'Clipart': 870, 'Product': 880, 'Real World': 870}
    },
    "OFFICE224": {  # for office-home
        "DOMAINS": ['Art', 'Clipart', 'Product', 'Real World'],
        "CLASSES": 65,
        "INPUT_DIM": 2048,
        "SOURCE_ERRORS": {'Art': 0.0535, 'Clipart': 0.0435, 'Product': 0.0169, 'Real World': 0.0310},
        "TEST_SET_SIZES": {'Art': 486, 'Clipart': 873, 'Product': 888, 'Real World': 870},
        "D_PRECOMP_PATH":"/data/nogaz/Convex_bounds_optimization/LatentFlow_Pixel_Experiments/results/D_Matrix_FINAL_GMM_Soft_OfficeHome.npy"
    },
    "OFFICE31": {
        "DOMAINS": ['amazon', 'dslr', 'webcam'],
        "CLASSES": 31,
        "INPUT_DIM": 2048,
        "SOURCE_ERRORS": {'amazon': 0.1352, 'dslr': 0.0178, 'webcam': 0.0225},
        "TEST_SET_SIZES": {'amazon': 2197, 'dslr': 281, 'webcam': 578},
        "D_PRECOMP_PATH":"/data/nogaz/Convex_bounds_optimization/LatentFlow_Pixel_Experiments/results/D_Matrix_FINAL_GMM_Soft.npy"
    }
}

CURRENT_CFG = CONFIGS[DATASET_MODE]
SOURCE_ERRORS = CURRENT_CFG["SOURCE_ERRORS"]
TEST_SET_SIZES = CURRENT_CFG["TEST_SET_SIZES"]
ALL_DOMAINS_LIST = CURRENT_CFG["DOMAINS"]
NUM_CLASSES = CURRENT_CFG["CLASSES"]
INPUT_DIM = CURRENT_CFG["INPUT_DIM"]

# ==========================================
# --- OPTIONAL: Use precomputed Global D ---
# ==========================================
USE_PRECOMPUTED_D = False   # If False -> always use KDE path
D_PRECOMP_PATH = CURRENT_CFG["D_PRECOMP_PATH"] #"/data/nogaz/Convex_bounds_optimization/LatentFlow_Pixel_Experiments/results/D_Matrix_FINAL_GMM_Soft_OfficeHome.npy"

# Root directories (adjust if needed)
OFFICEHOME_DIR = "/data/nogaz/Convex_bounds_optimization/OfficeHome"  # <-- change if your OfficeHome root differs


class FeatureExtractor(nn.Module):
    def __init__(self, original_model):
        super(FeatureExtractor, self).__init__()
        self.backbone = nn.Sequential(*list(original_model.children())[:-1])
        self.head = original_model.fc

    def forward(self, x):
        feats = torch.flatten(self.backbone(x), 1)
        return feats, self.head(feats)


# ============================================================
# --- NEW: Precomputed D support utilities (OfficeHome/Office31)
# ============================================================
def compute_domain_lengths(domains):
    """
    Computes the TOTAL number of samples (Train + Test) in each domain.
    Needed for correctly slicing the Global_D matrix.
    """
    lengths = {}
    for d in domains:
        if DATASET_MODE == 'DIGITS':
            try:
                ret = Data.get_data_loaders(d, seed=1)
                if isinstance(ret, tuple) or isinstance(ret, list):
                    l_train = len(ret[0].dataset)
                    l_test = len(ret[1].dataset)
                    lengths[d] = l_train + l_test
                else:
                    lengths[d] = len(ret.dataset)

            except Exception as e:
                print(f"Error computing length for {d}: {e}")
                lengths[d] = 0
        else:
            # OFFICE logic
            path = get_domain_path(d)
            if os.path.exists(path):
                ds = datasets.ImageFolder(path)
                lengths[d] = len(ds)
            else:
                lengths[d] = 0

    return lengths


def get_domain_path(domain_name: str) -> str:
    """
    Build path for a given domain based on DATASET_MODE.
    Adjust if your dataset layout differs.
    """
    if DATASET_MODE in ["OFFICE", "OFFICE224"]:
        # OfficeHome usually has: ROOT/<Domain>/images or ROOT/<Domain> with ImageFolder structure
        p1 = os.path.join(OFFICEHOME_DIR, domain_name, "images")
        p2 = os.path.join(OFFICEHOME_DIR, domain_name)
        return p1 if os.path.exists(p1) else p2
    elif DATASET_MODE == "OFFICE31":
        # If you want Office31 precomputed D similarly, set OFFICE31_DIR here
        OFFICE31_DIR = "/data/nogaz/Convex_bounds_optimization/Office-31"
        p1 = os.path.join(OFFICE31_DIR, domain_name, "images")
        p2 = os.path.join(OFFICE31_DIR, domain_name)
        return p1 if os.path.exists(p1) else p2
    else:
        raise ValueError("Precomputed D slicing is implemented for OFFICE/OFFICE224/OFFICE31 only.")


class TransformedSubset(torch.utils.data.Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


def get_train_test_loaders_and_indices(domain: str, seed: int, batch_size=64, test_batch_size=64):
    """
    Matches the other file behavior:
    - ImageFolder without transform
    - permute indices with seed
    - split 80/20
    - build transformed Subset loaders
    - return ordered loaders (shuffle=False) so D slicing aligns deterministically
    """
    if DATASET_MODE == 'DIGITS':
        train_loader, test_loader, _ = Data.get_data_loaders(domain, seed=seed, batch_size=batch_size)
        n_train = len(train_loader.dataset)
        n_test = len(test_loader.dataset)
        N_total = n_train + n_test
        train_idx = np.arange(0, n_train)
        test_idx = np.arange(n_train, n_train + n_test)
        return train_loader, test_loader, train_idx, test_idx, N_total

    path = get_domain_path(domain)
    ds_full = datasets.ImageFolder(path)

    N = len(ds_full)
    rng = np.random.RandomState(seed)
    indices = rng.permutation(N)
    split_point = int(0.8 * N)
    train_idx = indices[:split_point]
    test_idx = indices[split_point:]

    # Transforms (match your current pipeline)
    tr_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tr_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = TransformedSubset(Subset(ds_full, train_idx), tr_train)
    test_ds = TransformedSubset(Subset(ds_full, test_idx), tr_test)

    # IMPORTANT: ordered loaders (shuffle=False) for alignment with sliced D
    train_loader_ordered = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader_ordered = DataLoader(test_ds, batch_size=test_batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader_ordered, test_loader_ordered, train_idx, test_idx, N


# ==========================================
# --- Core matrix builder (KDE path) ---
# ==========================================

def build_DP_model_Classes(data_loaders, data_size, source_domains, models, classifiers, normalize_factors,
                           vae_norm_stats):
    print(f"--- Building Matrices (Size: {data_size}) ---")
    C, K = NUM_CLASSES, len(source_domains)
    Y, D, H = np.zeros((data_size, C)), np.zeros((data_size, C, K)), np.zeros((data_size, C, K))
    i = 0
    for dom_name, loader in data_loaders:
        print(f" [DATA] Processing {dom_name} batch extraction...")
        for data, label in loader:
            data = data.to(device)
            data = fix_batch_resnet(data, DATASET_MODE)
            N = len(data)
            one_hot = np.zeros((N, C))
            one_hot[np.arange(N)[label < C], label[label < C]] = 1
            Y[i:i + N] = one_hot
            with torch.no_grad():
                for k, src in enumerate(source_domains):
                    if DATASET_MODE == 'DIGITS':
                        logits = classifiers[src](data)
                        vae_in = data.view(data.size(0), -1)
                    else:
                        feats, logits = classifiers[src](data)
                        min_v, max_v = vae_norm_stats[src]
                        vae_in = torch.clamp((feats - min_v) / (max_v - min_v + 1e-6), 0, 1)

                    H[i:i + N, :, k] = F.softmax(logits, dim=1).cpu().numpy()
                    x_hat, _, _ = models[src](vae_in)
                    log_p = models[src].compute_log_probabitility_bernoulli(x_hat, vae_in)
                    D[i:i + N, :, k] = np.tile(log_p[:, None].cpu().numpy(), (1, C))

            i += N
            torch.cuda.empty_cache()

    for k, src in enumerate(source_domains):
        print(f" [KDE] Starting GridSearch for {src}...")
        raw = D[:, 0, k].reshape(-1, 1)
        mean_v, std_v = normalize_factors[src]
        proc = (raw - mean_v.item()) / max(std_v.item(), 1e-4)
        bw_range = np.linspace(0.1, 0.5, 10)
        kde = GridSearchCV(KernelDensity(kernel='gaussian'), {'bandwidth': bw_range}, cv=3)
        kde.fit(proc[np.random.permutation(len(proc))[:min(2000, len(proc))]])
        print(f" [KDE DEBUG] {src:<10} | Best BW: {kde.best_estimator_.bandwidth:.3f}")
        scores = np.exp(kde.best_estimator_.score_samples(proc))
        D[:, :, k] = np.tile(scores.reshape(-1, 1), (1, C))
        print(f" -> Score Range: {scores.min():.4e} to {scores.max():.4e} | Mean: {scores.mean():.4f}")

    print("--- Normalizing Matrices ---")
    dom_idx = np.argmax(D[:, 0, :], axis=1)
    unique, counts = np.unique(dom_idx, return_counts=True)
    print("\n[DOMAIN DOMINANCE CHECK]")
    for idx, count in zip(unique, counts):
        print(f" Source {source_domains[idx]:<10} is strongest for {count:>5} samples ({100 * count / data_size:.1f}%)")
    print("-" * 50)
    D[D < (D.max(axis=2, keepdims=True) * 0.2)] = 0.0
    for k in range(K):
        if D[:, :, k].max() > 1e-9:
            D[:, :, k] /= D[:, :, k].max()
    return Y, D, H


# ==========================================
# --- Baselines / Solvers ---
# ==========================================

# def evaluate_accuracy(w, D, H, Y):
#     preds = ((D * H) * w.reshape(1, 1, -1)).sum(axis=2)
#     return accuracy_score(Y.argmax(axis=1), preds.argmax(axis=1)) * 100.0

def evaluate_accuracy_wd(w, D, H, Y):
    preds = ((D * H) * w.reshape(1, 1, -1)).sum(axis=2)
    return accuracy_score(Y.argmax(axis=1), preds.argmax(axis=1)) * 100.0


def evaluate_accuracy_q(Q, H, Y):
    """
    Supports:
    1) Q shape (N, K)  -> sample-level weights
    2) Q shape (N*C, K)-> (sample,class)-level weights, reshape to (N, C, K)
    """
    Q = np.asarray(Q)
    N, C, K = H.shape
    if Q.shape == (N, K):
        preds = (H * Q[:, None, :]).sum(axis=2)          # (N, C)
    elif Q.shape == (N * C, K):
        Q3 = Q.reshape(N, C, K)                          # (N, C, K)
        preds = (H * Q3).sum(axis=2)                     # (N, C)
    else:
        raise ValueError(f"Unexpected Q shape {Q.shape}, expected {(N,K)} or {(N*C,K)}")
    return accuracy_score(Y.argmax(axis=1), preds.argmax(axis=1)) * 100.0


def run_baselines(Y, D, H, source_domains, target_domains, all_source_domains, seed):
    print(" [Baseline] Running Oracle and Uniform...")
    buf = io.StringIO()
    total = sum([TEST_SET_SIZES.get(d, 1) for d in target_domains])
    oracle_z = np.array([TEST_SET_SIZES.get(s, 0) / total if s in target_domains else 0.0 for s in source_domains])
    uniform_w = np.ones(len(source_domains)) / len(source_domains)
    for name, w in [("ORACLE", oracle_z), ("UNIFORM", uniform_w)]:
        acc = evaluate_accuracy_wd(w, D, H, Y)
        w_f = map_weights_to_full_source_list(w, source_domains, all_source_domains)
        buf.write(f"{name:<18} | {'N/A':<15} | {'N/A':<15} | {acc:<12.2f} | {str(np.round(w_f, 4))}\n")
        print(f" >>> [Baseline] {name:<7} | Acc: {acc:.2f}%")

    y_true = Y.argmax(axis=1)
    pred_per_source = H.argmax(axis=1)  # (N, K)
    any_correct = (pred_per_source == y_true[:, None]).any(axis=1)
    oracle_any_acc = any_correct.mean() * 100.0

    buf.write(f"{'ORACLE_ANY_CORRECT':<18} | {'N/A':<15} | {'N/A':<15} | {oracle_any_acc:<12.2f} | N/A\n")
    print(f" >>> [Baseline] ORACLE_ANY_CORRECT | Acc: {oracle_any_acc:.2f}%")

    print(" [Baseline] Running DC Solver...")
    dc_accuracies, best_z_dc = [], None
    for i in range(1):
        try:
            dp = init_problem_from_model(Y, D, H, p=len(source_domains), C=NUM_CLASSES)
            slv = ConvexConcaveSolver(ConvexConcaveProblem(dp), seed + (i * 100), "err")
            z_dc, _, _ = slv.solve()
            if z_dc is not None:
                acc = evaluate_accuracy_wd(z_dc, D, H, Y)
                dc_accuracies.append(acc)
                if best_z_dc is None or acc >= max(dc_accuracies):
                    best_z_dc = z_dc
        except:
            continue
    if dc_accuracies:
        avg_res = f"{np.mean(dc_accuracies):.2f}±{np.std(dc_accuracies):.2f}"
        w_f = map_weights_to_full_source_list(best_z_dc, source_domains, all_source_domains)
        buf.write(f"{'DC (5-Seeds)':<18} | {'N/A':<15} | {'N/A':<15} | {avg_res:<12} | {str(np.round(w_f, 4))}\n")
    return buf.getvalue()

# from cvxpy_3_21 import solve_convex_problem_smoothed_kl_321
# from cvxpy_3_22 import solve_convex_problem_domain_anchored_smoothed
# from cvxpy_3_23 import solve_convex_problem_smoothed_original_p


def run_solver_sweep_worker(Y, D, H, eps_mult, source_domains, all_source_domains):
    print(f" [Worker] Starting sweep for Epsilon Mult: {eps_mult}")
    buf = io.StringIO()
    errors = np.array([(SOURCE_ERRORS.get(d, 0.1) + 0.05) * eps_mult for d in source_domains])
    max_ent = np.log(len(source_domains)) if len(source_domains) > 1 else 0.1
    for solver in ["3.21", "3.22", "3.23", "CVXPY_GLOBAL"]:  # ,"CVXPY_GLOBAL", "CVXPY_PER_DOMAIN"
        for mult in [1.0]:#, 1.2]:
            delta = mult * max_ent
            try:
                if solver == "CVXPY_GLOBAL":
                    w, Q = solve_convex_problem_mosek(Y, D, H, delta=delta, epsilon=max(errors), solver_type='SCS')
                    if w is None:
                        print("[CVXPY] Returned None (likely infeasible)")
                        continue
                elif solver == "3.21":
                    w, Q = solve_convex_problem_smoothed_kl_321(Y, D, H, epsilon=max(errors), solver_type='SCS')
                    if w is None:
                        print("[3.21] Returned None (likely infeasible)")
                        continue
                elif solver == "3.22":
                    w, Q = solve_convex_problem_domain_anchored_smoothed(Y, D, H, epsilon=max(errors), solver_type='SCS')
                    if w is None:
                        print("[3.22] Returned None (likely infeasible)")
                        continue
                elif solver == "3.23":
                    w, Q = solve_convex_problem_smoothed_original_p(Y, D, H, epsilon=max(errors), solver_type='SCS')
                    if w is None:
                        print("[3.23] Returned None (likely infeasible)")
                        continue
                else:
                    w, Q = solve_convex_problem_per_domain(
                        Y, D, H,
                        delta=np.full(len(source_domains), delta),
                        epsilon=errors,
                        solver_type='SCS'
                    )
                acc_w = evaluate_accuracy_wd(w, D, H, Y)
                acc_Q = evaluate_accuracy_q(Q, H, Y)
                w_f = map_weights_to_full_source_list(w, source_domains, all_source_domains)
                buf.write(f"{solver:<18} | m:{eps_mult:<13} | m:{mult:<13} | acc_W: {acc_w:<12.2f} | {str(np.round(w_f, 4))}\n")
                buf.write(
                    f"{solver:<18} | m:{eps_mult:<13} | m:{mult:<13} | acc_Q: {acc_Q:<12.2f} | {str(np.round(w_f, 4))}\n")
            except Exception as e:
                tb = traceback.format_exc()
                msg = f"[{solver}] ERROR eps_mult={eps_mult} delta_mult={mult}: {repr(e)}\n{tb}\n"
                print(msg)
                buf.write(msg)
    return buf.getvalue()


# ============================================================
# --- NEW: Build Y,H using precomputed D (no KDE) ---
# ============================================================

def build_YDH_with_precomputed_D(target_domains, seed, classifiers,
                                 Global_D, domain_lengths):
    """
    Build Y, H from loaders (deterministic ordered test split),
    and fill D from precomputed Global_D (sliced per domain + split idx),
    then expand to (N, C, K) to match downstream code.
    """
    C = NUM_CLASSES
    K = len(target_domains)

    # Build ordered test loaders per domain (like the other file)
    loaders = []
    split_info = {}
    for d in target_domains:
        _, teL, tr_idx, te_idx, Ndom = get_train_test_loaders_and_indices(d, seed=seed, batch_size=64, test_batch_size=64)
        loaders.append((d, teL))
        split_info[d] = {"te_idx": te_idx, "N": Ndom}

    data_size = sum(len(l.dataset) for _, l in loaders)
    Y = np.zeros((data_size, C))
    H = np.zeros((data_size, C, K))
    D_loaded = np.zeros((data_size, K))  # (N, K)

    i = 0
    for dom_name, loader in loaders:
        # Domain block (Ndom, K_total)
        D_dom_full = slice_global_D_for_domain(Global_D, ALL_DOMAINS_LIST, domain_lengths, dom_name)

        # Select columns for active K domains, in the same order as target_domains
        col_idx = [ALL_DOMAINS_LIST.index(s) for s in target_domains]
        D_dom_full = D_dom_full[:, col_idx]

        # Slice by the same indices used by the loader (test split indices)
        te_idx = split_info[dom_name]["te_idx"]
        D_dom = D_dom_full[te_idx]  # (N_te, K)

        cursor = 0
        for data, label in loader:
            data = data.to(device)
            data = fix_batch_resnet(data, DATASET_MODE)
            N = len(data)

            one_hot = np.zeros((N, C))
            one_hot[np.arange(N)[label < C], label[label < C]] = 1
            Y[i:i + N] = one_hot

            with torch.no_grad():
                for k, src in enumerate(target_domains):
                    if DATASET_MODE == 'DIGITS':
                        logits = classifiers[src](data)
                    else:
                        feats, logits = classifiers[src](data)
                    H[i:i + N, :, k] = F.softmax(logits, dim=1).cpu().numpy()
            # Fill D rows corresponding to this batch
            D_loaded[i:i + N, :] = D_dom[cursor:cursor + N, :]
            cursor += N
            i += N

            torch.cuda.empty_cache()

    # Expand D to (N, C, K) so evaluate_accuracy stays unchanged
    D = np.tile(D_loaded[:, None, :], (1, C, 1))
    return Y, D, H


def task_run(classifiers, all_source_domains):
    seed = 1
    torch.manual_seed(seed)
    np.random.seed(seed)
    test_path = f'./results_{DATASET_MODE}_recreate/seed_{seed}/'
    os.makedirs(test_path, exist_ok=True)

    if USE_PRECOMPUTED_D is False:
        models, normalize_factors, vae_norm_stats = handle_vae_models(all_source_domains, classifiers, seed)

    # --- Precomputed D initialization (optional) ---
    Global_D = None
    domain_lengths = None
    if USE_PRECOMPUTED_D and os.path.exists(D_PRECOMP_PATH):
        Global_D = load_global_D_matrix(D_PRECOMP_PATH)
        if Global_D is not None:
            domain_lengths = compute_domain_lengths(all_source_domains)

    print("\n--- Starting Target Combinations ---")
    with open(os.path.join(test_path, f'Sweep_Results_{seed}_PRE_D_{USE_PRECOMPUTED_D}.txt'), 'a') as fp:
        for target in [list(s) for r in range(2, len(all_source_domains) + 1) for s in
                       itertools.combinations(all_source_domains, r)]:
            if set(target) in ({'MNIST', 'USPS'}): #, {'MNIST', 'SVHN'}):
                print("skip " + str(set(target)))
                continue

            print(f"\n[TARGET] Starting run for: {target}")
            total_s = sum([TEST_SET_SIZES.get(d, 0) for d in target])
            true_r = map_weights_to_full_source_list(
                np.array([TEST_SET_SIZES.get(d, 0) / total_s if total_s > 0 else 0 for d in target]),
                target,
                all_source_domains
            )

            fp.write(f"\n{'=' * 120}\nTARGET: {target} | TRUE RATIOS: {np.round(true_r, 4)}\n{'=' * 120}\n")
            fp.write(
                f"{'Solver':<18} | {'Epsilon Mult':<15} | {'Delta Mult':<15} | {'Acc (%)':<12} | {'Learned Weights'}\n" +
                "-" * 120 + "\n"
            )

            # ---------------------------------------------------------
            # CASE A: Use precomputed D (no KDE) - "like them"
            # ---------------------------------------------------------
            if Global_D is not None and domain_lengths is not None:
                print("✅ Using PRECOMPUTED Global D (sliced) instead of KDE.")
                Y, D, H = build_YDH_with_precomputed_D(
                    target_domains=target,
                    seed=seed,
                    classifiers=classifiers,
                    Global_D=Global_D,
                    domain_lengths=domain_lengths
                )

            # ---------------------------------------------------------
            # CASE B: Fallback: KDE (your existing pipeline)
            # ---------------------------------------------------------
            else:
                el = []
                for d in target:
                    _, l, _ = Data.get_data_loaders(d, seed=seed)
                    el.append((d, l))
                Y, D, H = build_DP_model_Classes(
                    el,
                    sum(len(l.dataset) for _, l in el),
                    target,
                    models,
                    classifiers,
                    normalize_factors,
                    vae_norm_stats
                )
            fp.write(run_baselines(Y, D, H, target, target, all_source_domains, seed))
            print(f" [Solver] Launching parallel workers for {target}...")
            results = Parallel(n_jobs=2, verbose=10)(
                delayed(run_solver_sweep_worker)(Y, D, H, e, target, all_source_domains)
                for e in [1.0, 1.1, 2]
            )
            for r in results:
                fp.write(r)
            fp.flush()
            print(f"✅ Finished target: {target}")


def handle_vae_models(all_source_domains, classifiers, seed):
    models, normalize_factors, vae_norm_stats = {}, {}, {}
    ap, an = (0.5, -0.5) if DATASET_MODE == "OFFICE31" else (2.0, -2.0)
    print("\n--- Initializing Models for all Domains ---")
    for d in all_source_domains:
        m = vr_model(INPUT_DIM, ap, an).to(device)
        pattern = f"./models_{d}_seed{seed}_224_features/vrs_*_model.pt"
        matches = glob.glob(pattern)
        if not matches:
            pattern = f"./models_{d}_seed{seed}_*/vrs_*_model.pt"
            matches = glob.glob(pattern)
        path = matches[0]
        m.load_state_dict(torch.load(path, map_location=device))
        models[d] = m.eval()
        loader, _, _ = Data.get_data_loaders(d, seed=seed)
        feats = []
        with torch.no_grad():
            with torch.no_grad():
                for j, (imgs, _) in enumerate(loader):
                    if j > 10: break
                    if DATASET_MODE == 'DIGITS':
                        f = imgs.view(imgs.size(0), -1).to(device)
                    else:
                        batch_fixed = fix_batch_resnet(imgs.to(device))
                        f, _ = classifiers[d](batch_fixed)
                    feats.append(f.cpu())

        cat_f = torch.cat(feats, 0).to(device)
        vae_norm_stats[d] = (cat_f.min(), cat_f.max())
        vae_in = torch.clamp((cat_f - cat_f.min()) / (cat_f.max() - cat_f.min() + 1e-6), 0, 1)
        out, _, _ = m(vae_in)
        lp = m.compute_log_probabitility_bernoulli(out, vae_in)
        normalize_factors[d] = (lp.mean(), lp.std())
        print(f"✅ Loaded VRS for {d}")
    return models, normalize_factors, vae_norm_stats


def main():
    print(f"Starting Main Process | Dataset Mode: {DATASET_MODE}")
    classifiers = {}
    for d in ALL_DOMAINS_LIST:
        if DATASET_MODE == 'DIGITS':
            m = ClSFR.Grey_32_64_128_gp()
            path = f"./classifiers_new/{d}_classifier.pt"
            if not os.path.exists(path):
                path = f"./classifiers/{d}_classifier.pt"
            print(f"Loading DIGITS model ({d}) from: {path}")
            m.load_state_dict(torch.load(path, map_location=device))
            m = m.to(device).eval()
            classifiers[d] = m
        # --- OFFICE (ResNet50) ---
        else:
            m = models.resnet50(weights=None)
            m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
            path = f"./classifiers/{d}_224.pt"
            if not os.path.exists(path):
                path = f"./classifiers_new/{d}_classifier.pt"
            print(f"Loading ResNet model ({d}) from: {path}")
            m.load_state_dict(torch.load(path, map_location=device))
            classifiers[d] = FeatureExtractor(m).to(device).eval()
        print(f"✅ Loaded Classifier: {d}")
    task_run(classifiers, ALL_DOMAINS_LIST)


if __name__ == "__main__":
    main()
