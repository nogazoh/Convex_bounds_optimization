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
from vae import *
import data as Data

# --- SOLVERS ---
try:
    from cvxpy_solver import solve_convex_problem_mosek
    from cvxpy_solver_per_domain import solve_convex_problem_per_domain
except ImportError:
    pass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "1")
torch.set_num_threads(1)

# ==========================================
# --- CONFIGURATION ---
# ==========================================
# OPTIONS: "DIGITS", "OFFICE", "OFFICE31"
DATASET_MODE = "OFFICE31"

CONFIGS = {
    "DIGITS": {
        "DOMAINS": ['MNIST', 'USPS', 'SVHN'],
        "CLASSES": 10,
        "INPUT_DIM": 2048,
        "SOURCE_ERRORS": {'MNIST': 0.005, 'USPS': 0.027, 'SVHN': 0.05},
        "TEST_SET_SIZES": {'MNIST': 10000, 'USPS': 2007, 'SVHN': 26032}
    },
    "OFFICE": {
        "DOMAINS": ['Art', 'Clipart', 'Product', 'Real World'],
        "CLASSES": 65,
        "INPUT_DIM": 2048,
        "SOURCE_ERRORS": {'Art': 0.11, 'Clipart': 0.08, 'Product': 0.03, 'Real World': 0.07},
        "TEST_SET_SIZES": {'Art': 490, 'Clipart': 870, 'Product': 880, 'Real World': 870}
    },
    "OFFICE31": {
        "DOMAINS": ['amazon', 'dslr', 'webcam'],
        "CLASSES": 31,
        "INPUT_DIM": 2048,
        "SOURCE_ERRORS": {'amazon': 0.1352, 'dslr': 0.0178, 'webcam': 0.0225},
        "TEST_SET_SIZES": {'amazon': 2197, 'dslr': 281, 'webcam': 578}
    }
}

CURRENT_CFG = CONFIGS[DATASET_MODE]
SOURCE_ERRORS = CURRENT_CFG["SOURCE_ERRORS"]
TEST_SET_SIZES = CURRENT_CFG["TEST_SET_SIZES"]
ALL_DOMAINS_LIST = CURRENT_CFG["DOMAINS"]
NUM_CLASSES = CURRENT_CFG["CLASSES"]
INPUT_DIM = CURRENT_CFG["INPUT_DIM"]


class FeatureExtractor(nn.Module):
    def __init__(self, original_model):
        super(FeatureExtractor, self).__init__()
        self.backbone = nn.Sequential(*list(original_model.children())[:-1])
        self.head = original_model.fc

    def forward(self, x):
        feats = torch.flatten(self.backbone(x), 1)
        return feats, self.head(feats)


def fix_batch_resnet(data):
    # Only resize for image-based datasets
    if DATASET_MODE in ["OFFICE", "OFFICE31"]:
        if data.shape[-1] < 224:
            data = F.interpolate(data, size=(224, 224), mode='bilinear', align_corners=False)
    return data


def map_weights_to_full_source_list(subset_weights, subset_sources, full_source_list):
    full_weights = np.zeros(len(full_source_list))
    if subset_weights is not None:
        for i, source in enumerate(full_source_list):
            if source in subset_sources:
                full_weights[i] = subset_weights[subset_sources.index(source)]
    return full_weights


def build_DP_model_Classes(data_loaders, data_size, source_domains, models, classifiers, normalize_factors,
                           vae_norm_stats):
    print(f"--- Building Matrices (Size: {data_size}) ---")
    C, K = NUM_CLASSES, len(source_domains)
    Y, D, H = np.zeros((data_size, C)), np.zeros((data_size, C, K)), np.zeros((data_size, C, K))
    i = 0
    for _, loader in data_loaders:
        for data, label in loader:
            data = data.to(device);
            data = fix_batch_resnet(data);
            N = len(data)
            one_hot = np.zeros((N, C));
            one_hot[np.arange(N)[label < C], label[label < C]] = 1
            Y[i:i + N] = one_hot
            with torch.no_grad():
                for k, src in enumerate(source_domains):
                    feats, logits = classifiers[src](data)
                    H[i:i + N, :, k] = F.softmax(logits, dim=1).cpu().numpy()
                    min_v, max_v = vae_norm_stats[src]
                    vae_in = torch.clamp((feats - min_v) / (max_v - min_v + 1e-6), 0, 1)
                    x_hat, _, _ = models[src](vae_in)
                    log_p = models[src].compute_log_probabitility_bernoulli(x_hat, vae_in)
                    D[i:i + N, :, k] = np.tile(log_p[:, None].cpu().numpy(), (1, C))
            i += N
            torch.cuda.empty_cache()

    # --- KDE Debugging Section ---
    for k, src in enumerate(source_domains):
        raw = D[:, 0, k].reshape(-1, 1)
        mean_v, std_v = normalize_factors[src]
        proc = (raw - mean_v.item()) / max(std_v.item(), 1e-4)
        bw_range = np.linspace(0.1, 0.5, 10)
        kde = GridSearchCV(KernelDensity(kernel='gaussian'), {'bandwidth': bw_range}, cv=3)
        kde.fit(proc[np.random.permutation(len(proc))[:min(2000, len(proc))]])

        print(f"   [KDE DEBUG] {src:<10} | Best BW: {kde.best_estimator_.bandwidth:.3f}")

        # Calculate scores
        scores = np.exp(kde.best_estimator_.score_samples(proc))
        D[:, :, k] = np.tile(scores.reshape(-1, 1), (1, C))
        print(f"     -> Score Range: {scores.min():.4e} to {scores.max():.4e} | Mean: {scores.mean():.4f}")

    # Dominance Check: Who is the "closest" source for each target sample?
    dom_idx = np.argmax(D[:, 0, :], axis=1)
    unique, counts = np.unique(dom_idx, return_counts=True)
    print("\n[DOMAIN DOMINANCE CHECK]")
    for idx, count in zip(unique, counts):
        print(
            f"   Source {source_domains[idx]:<10} is strongest for {count:>5} samples ({100 * count / data_size:.1f}%)")
    print("-" * 50)

    D[D < (D.max(axis=2, keepdims=True) * 0.2)] = 0.0
    for k in range(K):
        if D[:, :, k].max() > 1e-9: D[:, :, k] /= D[:, :, k].max()
    return Y, D, H


def run_baselines(Y, D, H, source_domains, target_domains, all_source_domains, seed):
    buf = io.StringIO()
    total = sum([TEST_SET_SIZES.get(d, 1) for d in target_domains])
    oracle_z = np.array([TEST_SET_SIZES.get(s, 0) / total if s in target_domains else 0.0 for s in source_domains])
    uniform_w = np.ones(len(source_domains)) / len(source_domains)

    for name, w in [("ORACLE", oracle_z), ("UNIFORM", uniform_w)]:
        acc = evaluate_accuracy(w, D, H, Y)
        w_f = map_weights_to_full_source_list(w, source_domains, all_source_domains)
        buf.write(f"{name:<18} | {'N/A':<15} | {'N/A':<15} | {acc:<12.2f} | {str(np.round(w_f, 4))}\n")
        print(f"    >>> [Baseline] {name:<7} | Acc: {acc:.2f}%")

    dc_accuracies, best_z_dc = [], None
    for i in range(5):
        try:
            dp = init_problem_from_model(Y, D, H, p=len(source_domains), C=NUM_CLASSES)
            slv = ConvexConcaveSolver(ConvexConcaveProblem(dp), seed + (i * 100), "err")
            z_dc, _, _ = slv.solve()
            if z_dc is not None:
                acc = evaluate_accuracy(z_dc, D, H, Y)
                dc_accuracies.append(acc)
                if best_z_dc is None or acc >= max(dc_accuracies): best_z_dc = z_dc
        except:
            continue
    if dc_accuracies:
        avg_res = f"{np.mean(dc_accuracies):.2f}±{np.std(dc_accuracies):.2f}"
        w_f = map_weights_to_full_source_list(best_z_dc, source_domains, all_source_domains)
        buf.write(f"{'DC (5-Seeds)':<18} | {'N/A':<15} | {'N/A':<15} | {avg_res:<12} | {str(np.round(w_f, 4))}\n")
    return buf.getvalue()


def run_solver_sweep_worker(Y, D, H, eps_mult, source_domains, all_source_domains):
    buf = io.StringIO()
    errors = np.array([(SOURCE_ERRORS.get(d, 0.1) + 0.05) * eps_mult for d in source_domains])
    max_ent = np.log(len(source_domains)) if len(source_domains) > 1 else 0.1
    for solver in ["CVXPY_GLOBAL", "CVXPY_PER_DOMAIN"]:
        for mult in [1.0, 1.2]:
            delta = mult * max_ent
            try:
                if solver == "CVXPY_GLOBAL":
                    w = solve_convex_problem_mosek(Y, D, H, delta=delta, epsilon=max(errors), solver_type='SCS')
                else:
                    w = solve_convex_problem_per_domain(Y, D, H, delta=np.full(len(source_domains), delta),
                                                        epsilon=errors, solver_type='SCS')
                acc = evaluate_accuracy(w, D, H, Y)
                w_f = map_weights_to_full_source_list(w, source_domains, all_source_domains)
                config_eps = f"m:{eps_mult}"
                config_dlt = f"m:{mult} (v:{delta:.2f})"
                buf.write(
                    f"{solver:<18} | {config_eps:<15} | {config_dlt:<15} | {acc:<12.2f} | {str(np.round(w_f, 4))}\n")
            except:
                pass
    return buf.getvalue()


def evaluate_accuracy(w, D, H, Y):
    preds = ((D * H) * w.reshape(1, 1, -1)).sum(axis=2)
    return accuracy_score(Y.argmax(axis=1), preds.argmax(axis=1)) * 100.0


def task_run(classifiers, all_source_domains):
    seed = 1;
    torch.manual_seed(seed);
    np.random.seed(seed)
    test_path = f'./results_{DATASET_MODE}_V8/seed_{seed}/';
    os.makedirs(test_path, exist_ok=True)
    models, normalize_factors, vae_norm_stats = {}, {}, {}

    # Auto-adjust VRS parameters: Office31 uses (0.5, -0.5), Others use (2.0, -2.0)
    ap, an = (0.5, -0.5) if DATASET_MODE == "OFFICE31" else (2.0, -2.0)

    for d in all_source_domains:
        m = vr_model(INPUT_DIM, ap, an).to(device)
        path = glob.glob(f"./models_{d}_seed{seed}_*/vrs_*_model.pt")[0]
        m.load_state_dict(torch.load(path, map_location=device));
        models[d] = m.eval()
        loader, _, _ = Data.get_data_loaders(d, seed=seed)
        feats = []
        with torch.no_grad():
            for j, (imgs, _) in enumerate(loader):
                if j > 10: break
                f, _ = classifiers[d](fix_batch_resnet(imgs.to(device)))
                feats.append(f.cpu())
        cat_f = torch.cat(feats, 0).to(device)
        vae_norm_stats[d] = (cat_f.min(), cat_f.max())
        vae_in = torch.clamp((cat_f - cat_f.min()) / (cat_f.max() - cat_f.min() + 1e-6), 0, 1)
        out, _, _ = m(vae_in);
        lp = m.compute_log_probabitility_bernoulli(out, vae_in)
        normalize_factors[d] = (lp.mean(), lp.std())
        torch.cuda.empty_cache()

    with open(os.path.join(test_path, f'Sweep_Results_{seed}.txt'), 'a') as fp:
        # Range adjusted to cover all possible subset sizes
        for target in [list(s) for r in range(2, len(all_source_domains) + 1) for s in
                       itertools.combinations(all_source_domains, r)]:
            total_s = sum([TEST_SET_SIZES.get(d, 0) for d in target])
            true_r = map_weights_to_full_source_list(
                np.array([TEST_SET_SIZES.get(d, 0) / total_s if total_s > 0 else 0 for d in target]), target,
                all_source_domains)
            fp.write(f"\n{'=' * 120}\nTARGET: {target} | TRUE RATIOS: {np.round(true_r, 4)}\n{'=' * 120}\n")
            fp.write(
                f"{'Solver':<18} | {'Epsilon Mult':<15} | {'Delta Mult':<15} | {'Acc (%)':<12} | {'Learned Weights'}\n" + "-" * 120 + "\n")
            el = []
            for d in target:
                _, l, _ = Data.get_data_loaders(d, seed=seed);
                el.append((d, l))
            Y, D, H = build_DP_model_Classes(el, sum(len(l.dataset) for _, l in el), target, models, classifiers,
                                             normalize_factors, vae_norm_stats)
            fp.write(run_baselines(Y, D, H, target, target, all_source_domains, seed))
            results = Parallel(n_jobs=-1)(
                delayed(run_solver_sweep_worker)(Y, D, H, e, target, all_source_domains) for e in [1.0, 1.1])
            for r in results: fp.write(r)
            fp.flush()
            torch.cuda.empty_cache()


def main():
    classifiers = {}
    for d in ALL_DOMAINS_LIST:
        m = models.resnet50(weights=None);
        m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
        m.load_state_dict(torch.load(f"./classifiers_new/{d}_classifier.pt", map_location=device))
        classifiers[d] = FeatureExtractor(m).to(device).eval()
        print(f"✅ Loaded {d}")
    task_run(classifiers, ALL_DOMAINS_LIST)


if __name__ == "__main__":
    main()