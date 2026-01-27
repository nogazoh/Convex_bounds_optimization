# from __future__ import print_function
# import time
# import torch.utils.data
# import logging
# from sklearn.metrics import accuracy_score
# import pandas as pd
# import numpy as np
# from scipy import stats
# from sklearn.neighbors import KernelDensity
# from sklearn.model_selection import GridSearchCV
# import os
# from joblib import Parallel, delayed
# import torch
# import torch.nn.functional as F
# import glob
# import sys
# import io
# import itertools
# from torchvision import models, transforms
# import torch.nn as nn
# import cvxpy as cp  # <--- ADDED THIS
#
# # --- YOUR MODULES ---
# from dc import *
# import classifier as ClSFR
# from vae import *
# import data as Data
#
# # --- SOLVERS ---
# solve_convex_problem_mosek_original = None # ברירת מחדל כדי למנוע NameError
# try:
#     from cvxpy_solver import solve_convex_problem_mosek as solve_convex_problem_mosek_original
#     from cvxpy_solver_per_domain import solve_convex_problem_per_domain
# except ImportError:
#     print("⚠️ Warning: Original 'cvxpy_solver' not found. Skipping those solvers.")
#
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# os.environ.setdefault("OMP_NUM_THREADS", "1")
# torch.set_num_threads(1)
#
# # ==========================================
# # --- CONFIGURATION ---
# # ==========================================
# DATASET_MODE = "OFFICE224"
#
# CONFIGS = {
#     "DIGITS": {
#         "DOMAINS": ['MNIST', 'USPS', 'SVHN'],
#         "CLASSES": 10,
#         "INPUT_DIM": 2048,
#         "SOURCE_ERRORS": {'MNIST': 0.005, 'USPS': 0.027, 'SVHN': 0.05},
#         "TEST_SET_SIZES": {'MNIST': 10000, 'USPS': 2007, 'SVHN': 26032}
#     },
#     "OFFICE": { # for office-home after resizing by 0.5
#         "DOMAINS": ['Art', 'Clipart', 'Product', 'Real World'],
#         "CLASSES": 65,
#         "INPUT_DIM": 2048,
#         "SOURCE_ERRORS": {'Art': 0.11, 'Clipart': 0.08, 'Product': 0.03, 'Real World': 0.07},
#         "TEST_SET_SIZES": {'Art': 490, 'Clipart': 870, 'Product': 880, 'Real World': 870}
#     },
#     "OFFICE224": {# for office-home
#         "DOMAINS": ['Art', 'Clipart', 'Product', 'Real World'],
#         "CLASSES": 65,
#         "INPUT_DIM": 2048,
#         "SOURCE_ERRORS": {'Art': 0.0535, 'Clipart': 0.0435, 'Product': 0.0169, 'Real World': 0.0310},
#         "TEST_SET_SIZES": {'Art': 486, 'Clipart': 873, 'Product': 888, 'Real World': 870}
#     },
#     "OFFICE31": {
#         "DOMAINS": ['amazon', 'dslr', 'webcam'],
#         "CLASSES": 31,
#         "INPUT_DIM": 2048,
#         "SOURCE_ERRORS": {'amazon': 0.1352, 'dslr': 0.0178, 'webcam': 0.0225},
#         "TEST_SET_SIZES": {'amazon': 2197, 'dslr': 281, 'webcam': 578}
#     }
# }
#
# CURRENT_CFG = CONFIGS[DATASET_MODE]
# SOURCE_ERRORS = CURRENT_CFG["SOURCE_ERRORS"]
# TEST_SET_SIZES = CURRENT_CFG["TEST_SET_SIZES"]
# ALL_DOMAINS_LIST = CURRENT_CFG["DOMAINS"]
# NUM_CLASSES = CURRENT_CFG["CLASSES"]
# INPUT_DIM = CURRENT_CFG["INPUT_DIM"]
#
#
# class FeatureExtractor(nn.Module):
#     def __init__(self, original_model):
#         super(FeatureExtractor, self).__init__()
#         self.backbone = nn.Sequential(*list(original_model.children())[:-1])
#         self.head = original_model.fc
#
#     def forward(self, x):
#         feats = torch.flatten(self.backbone(x), 1)
#         return feats, self.head(feats)
#
#
# def fix_batch_resnet(data):
#     if DATASET_MODE in ["OFFICE", "OFFICE31", "OFFICE224"]:
#         if data.shape[-1] < 224:
#             data = F.interpolate(data, size=(224, 224), mode='bilinear', align_corners=False)
#     return data
#
#
# def map_weights_to_full_source_list(subset_weights, subset_sources, full_source_list):
#     full_weights = np.zeros(len(full_source_list))
#     if subset_weights is not None:
#         for i, source in enumerate(full_source_list):
#             if source in subset_sources:
#                 full_weights[i] = subset_weights[subset_sources.index(source)]
#     return full_weights
#
#
# # ==========================================
# # --- NEW SOLVER FUNCTIONS (FINAL FIX) ---
# # ==========================================
#
# def compute_p_tilde(D, eps=1e-15):
#     """ Domain-anchored normalized expert density """
#     if D.ndim == 3: D = D.reshape(-1, D.shape[2])
#     row_sum = np.sum(D, axis=1, keepdims=True)
#     return D / np.maximum(row_sum, eps)
#
#
# def solve_convex_problem_mosek_custom(Y, D, H, delta=1e-2, epsilon=1e-2, L_mat=None, solver_type='SCS'):
#     if D.ndim == 3: D = D.reshape(-1, D.shape[2])
#     N, k = D.shape
#
#     if L_mat is None:
#         if isinstance(epsilon, (list, np.ndarray)):
#             L_mat = np.tile(epsilon, (N, 1))
#         else:
#             L_mat = np.full((N, k), epsilon)
#
#     col_sums = D.sum(axis=0, keepdims=True)
#     col_sums[col_sums == 0] = 1.0
#     D_norm = D / col_sums
#
#     w = cp.Variable(k, nonneg=True, name='w')
#     Q = cp.Variable((N, k), nonneg=True, name='Q')
#     R = cp.Variable((k, k), nonneg=True, name='R')
#
#     obj_terms = []
#     for j in range(k):
#         obj_terms.append(cp.sum(cp.multiply(D_norm[:, j], -cp.rel_entr(w[j], Q[:, j]))))
#     objective = cp.Maximize(cp.sum(obj_terms))
#
#     constraints = [cp.sum(w) == 1, cp.sum(Q, axis=1) == 1, cp.sum(R, axis=0) == 1]
#
#     kl_terms = []
#     for t in range(k):
#         p_t = D_norm[:, t]
#         inner = []
#         for j in range(k):
#             re = cp.rel_entr(R[j, t], Q[:, j])
#             inner.append(cp.sum(cp.multiply(p_t, re)))
#         kl_terms.append(cp.sum(inner))
#     constraints.append((1.0 / k) * cp.sum(kl_terms) <= delta)
#
#     if isinstance(epsilon, (list, np.ndarray)):
#         for t in range(k):
#             constraints.append(cp.sum(cp.multiply(Q[:, t], L_mat[:, t])) <= epsilon[t])
#     else:
#         for t in range(k):
#             constraints.append(cp.sum(cp.multiply(Q[:, t], L_mat[:, t])) <= epsilon)
#
#     prob = cp.Problem(objective, constraints)
#     try:
#         prob.solve(solver=cp.SCS, verbose=False, eps=1e-4, max_iters=10000)
#     except:
#         return None
#
#     if prob.status not in ["optimal", "optimal_inaccurate"]: return None
#     return np.asarray(w.value).reshape(-1)
#
#
# def solve_convex_problem_smoothed_kl(Y, D, H, epsilon=1e-2, eta=1e-2, solver_type="SCS", q_min=1e-12, w_min=1e-12,
#                                      scs_eps=1e-4, scs_max_iters=20000):
#     if D.ndim == 3: D = D.reshape(-1, D.shape[2])
#     N, k = D.shape
#
#     if isinstance(epsilon, (list, np.ndarray)):
#         L_mat = np.tile(epsilon, (N, 1))
#     else:
#         L_mat = np.full((N, k), epsilon)
#
#     w = cp.Variable(k, nonneg=True, name="w")
#     Q = cp.Variable((N, k), nonneg=True, name="Q")
#
#     main_terms = [cp.sum(cp.multiply(D[:, j], -cp.rel_entr(w[j], Q[:, j]))) for j in range(k)]
#     main_obj = cp.sum(main_terms)
#     smooth_obj = (eta / (k * N)) * cp.sum(cp.log(Q))
#     objective = cp.Maximize(main_obj + smooth_obj)
#
#     constraints = [cp.sum(w) == 1, w >= w_min, cp.sum(Q, axis=1) == 1, Q >= q_min]
#
#     for t in range(k):
#         eps_t = epsilon[t] if isinstance(epsilon, (list, np.ndarray)) else epsilon
#         constraints.append(cp.sum(cp.multiply(Q[:, t], L_mat[:, t])) <= eps_t)
#
#     prob = cp.Problem(objective, constraints)
#     try:
#         if solver_type == "MOSEK":
#             prob.solve(solver=cp.MOSEK, verbose=False)
#         else:
#             prob.solve(solver=cp.SCS, verbose=False, eps=scs_eps, max_iters=scs_max_iters)
#     except:
#         return None
#     if prob.status not in ["optimal", "optimal_inaccurate"]: return None
#     return np.asarray(w.value).reshape(-1)
#
#
# def solve_convex_problem_domain_anchored_smoothed(Y, D, H, epsilon=1e-2, eta=1e-2, solver_type="SCS", q_min=1e-12,
#                                                   w_min=1e-12, scs_eps=1e-4, scs_max_iters=20000,
#                                                   normalize_domains=True, ptilde_eps=1e-15):
#     if D.ndim == 3: D = D.reshape(-1, D.shape[2])
#     N, k = D.shape
#
#     p_tilde = compute_p_tilde(D, eps=ptilde_eps) if normalize_domains else D
#
#     if isinstance(epsilon, (list, np.ndarray)):
#         L_mat = np.tile(epsilon, (N, 1))
#     else:
#         L_mat = np.full((N, k), epsilon)
#
#     w = cp.Variable(k, nonneg=True, name="w")
#     Q = cp.Variable((N, k), nonneg=True, name="Q")
#
#     main_terms = [cp.sum(cp.multiply(D[:, j], -cp.rel_entr(w[j], Q[:, j]))) for j in range(k)]
#     main_obj = cp.sum(main_terms)
#     anchored_smooth_obj = (eta / k) * cp.sum(cp.multiply(p_tilde, cp.log(Q)))
#     objective = cp.Maximize(main_obj + anchored_smooth_obj)
#
#     constraints = [cp.sum(w) == 1, w >= w_min, cp.sum(Q, axis=1) == 1, Q >= q_min]
#
#     for t in range(k):
#         eps_t = epsilon[t] if isinstance(epsilon, (list, np.ndarray)) else epsilon
#         constraints.append(cp.sum(cp.multiply(Q[:, t], L_mat[:, t])) <= eps_t)
#
#     prob = cp.Problem(objective, constraints)
#     try:
#         if solver_type == "MOSEK":
#             prob.solve(solver=cp.MOSEK, verbose=False)
#         else:
#             prob.solve(solver=cp.SCS, verbose=False, eps=scs_eps, max_iters=scs_max_iters)
#     except:
#         return None
#     if prob.status not in ["optimal", "optimal_inaccurate"]: return None
#     return np.asarray(w.value).reshape(-1)
#
#
# def solve_convex_problem_smoothed_original_p(Y, D, H, epsilon=1e-2, eta=1e-2, solver_type="SCS", q_min=1e-12,
#                                              w_min=1e-12, scs_eps=1e-4, scs_max_iters=20000):
#     if D.ndim == 3: D = D.reshape(-1, D.shape[2])
#     N, k = D.shape
#
#     if isinstance(epsilon, (list, np.ndarray)):
#         L_mat = np.tile(epsilon, (N, 1))
#     else:
#         L_mat = np.full((N, k), epsilon)
#
#     w = cp.Variable(k, nonneg=True, name="w")
#     Q = cp.Variable((N, k), nonneg=True, name="Q")
#
#     main_terms = [cp.sum(cp.multiply(D[:, j], -cp.rel_entr(w[j], Q[:, j]))) for j in range(k)]
#     main_obj = cp.sum(main_terms)
#     smooth_obj = eta * cp.sum(cp.multiply(D, cp.log(Q)))
#     objective = cp.Maximize(main_obj + smooth_obj)
#
#     constraints = [cp.sum(w) == 1, w >= w_min, cp.sum(Q, axis=1) == 1, Q >= q_min]
#
#     for t in range(k):
#         eps_t = epsilon[t] if isinstance(epsilon, (list, np.ndarray)) else epsilon
#         constraints.append(cp.sum(cp.multiply(Q[:, t], L_mat[:, t])) <= eps_t)
#
#     prob = cp.Problem(objective, constraints)
#     try:
#         if solver_type == "MOSEK":
#             prob.solve(solver=cp.MOSEK, verbose=False)
#         else:
#             prob.solve(solver=cp.SCS, verbose=False, eps=scs_eps, max_iters=scs_max_iters)
#     except:
#         return None
#     if prob.status not in ["optimal", "optimal_inaccurate"]: return None
#     return np.asarray(w.value).reshape(-1)
#
# # ==========================================
# # --- NEW SOLVER FUNCTIONS END ---
# # ==========================================
#
# def build_DP_model_Classes(data_loaders, data_size, source_domains, models, classifiers, normalize_factors,
#                            vae_norm_stats):
#     print(f"--- Building Matrices (Size: {data_size}) ---")
#     C, K = NUM_CLASSES, len(source_domains)
#     Y, D, H = np.zeros((data_size, C)), np.zeros((data_size, C, K)), np.zeros((data_size, C, K))
#     i = 0
#     for dom_name, loader in data_loaders:
#         print(f"   [DATA] Processing {dom_name} batch extraction...")
#         for data, label in loader:
#             data = data.to(device)
#             data = fix_batch_resnet(data)
#             N = len(data)
#             one_hot = np.zeros((N, C))
#             one_hot[np.arange(N)[label < C], label[label < C]] = 1
#             Y[i:i + N] = one_hot
#             with torch.no_grad():
#                 for k, src in enumerate(source_domains):
#                     feats, logits = classifiers[src](data)
#                     H[i:i + N, :, k] = F.softmax(logits, dim=1).cpu().numpy()
#                     min_v, max_v = vae_norm_stats[src]
#                     vae_in = torch.clamp((feats - min_v) / (max_v - min_v + 1e-6), 0, 1)
#                     x_hat, _, _ = models[src](vae_in)
#                     log_p = models[src].compute_log_probabitility_bernoulli(x_hat, vae_in)
#                     D[i:i + N, :, k] = np.tile(log_p[:, None].cpu().numpy(), (1, C))
#             i += N
#             torch.cuda.empty_cache()
#
#     for k, src in enumerate(source_domains):
#         print(f"   [KDE] Starting GridSearch for {src}...")
#         raw = D[:, 0, k].reshape(-1, 1)
#         mean_v, std_v = normalize_factors[src]
#         proc = (raw - mean_v.item()) / max(std_v.item(), 1e-4)
#         bw_range = np.linspace(0.1, 0.5, 10)
#         kde = GridSearchCV(KernelDensity(kernel='gaussian'), {'bandwidth': bw_range}, cv=3)
#         kde.fit(proc[np.random.permutation(len(proc))[:min(2000, len(proc))]])
#         print(f"   [KDE DEBUG] {src:<10} | Best BW: {kde.best_estimator_.bandwidth:.3f}")
#         scores = np.exp(kde.best_estimator_.score_samples(proc))
#         D[:, :, k] = np.tile(scores.reshape(-1, 1), (1, C))
#         print(f"     -> Score Range: {scores.min():.4e} to {scores.max():.4e} | Mean: {scores.mean():.4f}")
#
#     print("--- Normalizing Matrices ---")
#     dom_idx = np.argmax(D[:, 0, :], axis=1)
#     unique, counts = np.unique(dom_idx, return_counts=True)
#     print("\n[DOMAIN DOMINANCE CHECK]")
#     for idx, count in zip(unique, counts):
#         print(
#             f"   Source {source_domains[idx]:<10} is strongest for {count:>5} samples ({100 * count / data_size:.1f}%)")
#     print("-" * 50)
#     D[D < (D.max(axis=2, keepdims=True) * 0.2)] = 0.0
#     for k in range(K):
#         if D[:, :, k].max() > 1e-9: D[:, :, k] /= D[:, :, k].max()
#     return Y, D, H
#
#
# def run_baselines(Y, D, H, source_domains, target_domains, all_source_domains, seed):
#     print("   [Baseline] Running Oracle and Uniform...")
#     buf = io.StringIO()
#     total = sum([TEST_SET_SIZES.get(d, 1) for d in target_domains])
#     oracle_z = np.array([TEST_SET_SIZES.get(s, 0) / total if s in target_domains else 0.0 for s in source_domains])
#     uniform_w = np.ones(len(source_domains)) / len(source_domains)
#     for name, w in [("ORACLE", oracle_z), ("UNIFORM", uniform_w)]:
#         acc = evaluate_accuracy(w, D, H, Y)
#         w_f = map_weights_to_full_source_list(w, source_domains, all_source_domains)
#         buf.write(f"{name:<18} | {'N/A':<15} | {'N/A':<15} | {acc:<12.2f} | {str(np.round(w_f, 4))}\n")
#         print(f"    >>> [Baseline] {name:<7} | Acc: {acc:.2f}%")
#
#     print("   [Baseline] Running DC Solver...")
#     dc_accuracies, best_z_dc = [], None
#     for i in range(5):
#         try:
#             dp = init_problem_from_model(Y, D, H, p=len(source_domains), C=NUM_CLASSES)
#             slv = ConvexConcaveSolver(ConvexConcaveProblem(dp), seed + (i * 100), "err")
#             z_dc, _, _ = slv.solve()
#             if z_dc is not None:
#                 acc = evaluate_accuracy(z_dc, D, H, Y)
#                 dc_accuracies.append(acc)
#                 if best_z_dc is None or acc >= max(dc_accuracies): best_z_dc = z_dc
#         except:
#             continue
#     if dc_accuracies:
#         avg_res = f"{np.mean(dc_accuracies):.2f}±{np.std(dc_accuracies):.2f}"
#         w_f = map_weights_to_full_source_list(best_z_dc, source_domains, all_source_domains)
#         buf.write(f"{'DC (5-Seeds)':<18} | {'N/A':<15} | {'N/A':<15} | {avg_res:<12} | {str(np.round(w_f, 4))}\n")
#     return buf.getvalue()
#
#
# def run_solver_sweep_worker(Y, D, H, eps_mult, source_domains, all_source_domains):
#     print(f"      [Worker] Starting sweep for Epsilon Mult: {eps_mult}")
#     buf = io.StringIO()
#     errors = np.array([(SOURCE_ERRORS.get(d, 0.1) + 0.05) * eps_mult for d in source_domains])
#     max_ent = np.log(len(source_domains)) if len(source_domains) > 1 else 0.1
#
#     delta_solvers = ["CVXPY_GLOBAL", "New_Mosek_Custom"]
#
#     for solver in delta_solvers:
#         for mult in [1.0, 1.2]:
#             delta = mult * max_ent
#             try:
#                 w = None
#                 if solver == "CVXPY_GLOBAL":
#                     if solve_convex_problem_mosek_original is None:
#                         continue
#                     w = solve_convex_problem_mosek_original(Y, D, H, delta=delta, epsilon=max(errors), solver_type='SCS')
#
#                 elif solver == "New_Mosek_Custom":
#                     w = solve_convex_problem_mosek_custom(Y, D, H, delta=delta, epsilon=errors)
#
#                 if w is not None:
#                     acc = evaluate_accuracy(w, D, H, Y)
#                     w_f = map_weights_to_full_source_list(w, source_domains, all_source_domains)
#                     param_str = f"d_mul:{mult}"
#                     buf.write(
#                         f"{solver:<18} | m:{eps_mult:<13} | {param_str:<15} | {acc:<12.2f} | {str(np.round(w_f, 4))}\n")
#             except Exception as e:
#                 print(f"Error in {solver}: {e}")
#                 pass
#
#     eta_solvers = ["Eq_3.21_SmoothKL", "Eq_3.22_AnchorSm", "Eq_3.23_OrigP_Sm"]
#     FIXED_ETA = 1e-2
#
#     for solver in eta_solvers:
#         try:
#             w = None
#             if solver == "Eq_3.21_SmoothKL":
#                 w = solve_convex_problem_smoothed_kl(Y, D, H, epsilon=errors, eta=FIXED_ETA)
#             elif solver == "Eq_3.22_AnchorSm":
#                 w = solve_convex_problem_domain_anchored_smoothed(Y, D, H, epsilon=errors, eta=FIXED_ETA)
#             elif solver == "Eq_3.23_OrigP_Sm":
#                 w = solve_convex_problem_smoothed_original_p(Y, D, H, epsilon=errors, eta=FIXED_ETA)
#
#             if w is not None:
#                 acc = evaluate_accuracy(w, D, H, Y)
#                 w_f = map_weights_to_full_source_list(w, source_domains, all_source_domains)
#                 param_str = f"eta:{FIXED_ETA}"
#                 buf.write(
#                     f"{solver:<18} | m:{eps_mult:<13} | {param_str:<15} | {acc:<12.2f} | {str(np.round(w_f, 4))}\n")
#         except Exception as e:
#             print(f"Error in {solver}: {e}")
#             pass
#
#     return buf.getvalue()
#
# def evaluate_accuracy(w, D, H, Y):
#     preds = ((D * H) * w.reshape(1, 1, -1)).sum(axis=2)
#     return accuracy_score(Y.argmax(axis=1), preds.argmax(axis=1)) * 100.0
#
#
# def task_run(classifiers, all_source_domains):
#     seed = 1
#     torch.manual_seed(seed)
#     np.random.seed(seed)
#     test_path = f'./results_{DATASET_MODE}_recreate/seed_{seed}/'
#     os.makedirs(test_path, exist_ok=True)
#     models, normalize_factors, vae_norm_stats = {}, {}, {}
#     ap, an = (0.5, -0.5) if DATASET_MODE == "OFFICE31" else (2.0, -2.0)
#
#     print("\n--- Initializing Models for all Domains ---")
#     for d in all_source_domains:
#         m = vr_model(INPUT_DIM, ap, an).to(device)
#         pattern = f"./models_{d}_seed{seed}_224_features/vrs_*_model.pt"
#         matches = glob.glob(pattern)
#         if not matches:
#             pattern = f"./models_{d}_seed{seed}_*/vrs_*_model.pt"
#             matches = glob.glob(pattern)
#
#         path = matches[0]
#         m.load_state_dict(torch.load(path, map_location=device));
#         models[d] = m.eval()
#         loader, _, _ = Data.get_data_loaders(d, seed=seed)
#         feats = []
#         with torch.no_grad():
#             for j, (imgs, _) in enumerate(loader):
#                 if j > 10: break
#                 f, _ = classifiers[d](fix_batch_resnet(imgs.to(device)))
#                 feats.append(f.cpu())
#         cat_f = torch.cat(feats, 0).to(device)
#         vae_norm_stats[d] = (cat_f.min(), cat_f.max())
#         vae_in = torch.clamp((cat_f - cat_f.min()) / (cat_f.max() - cat_f.min() + 1e-6), 0, 1)
#         out, _, _ = m(vae_in)
#         lp = m.compute_log_probabitility_bernoulli(out, vae_in)
#         normalize_factors[d] = (lp.mean(), lp.std())
#         print(f"✅ Loaded VRS for {d}")
#
#     print("\n--- Starting Target Combinations ---")
#     with open(os.path.join(test_path, f'Sweep_Results_{seed}.txt'), 'a') as fp:
#         for target in [list(s) for r in range(2, len(all_source_domains) + 1) for s in
#                        itertools.combinations(all_source_domains, r)]:
#             # if set(target) == {'Art', 'Clipart', 'Product'}:
#             #     print(f"⏩ Skipping excluded target: {target}")
#             #     continue
#
#             print(f"\n[TARGET] Starting run for: {target}")
#             total_s = sum([TEST_SET_SIZES.get(d, 0) for d in target])
#             true_r = map_weights_to_full_source_list(
#                 np.array([TEST_SET_SIZES.get(d, 0) / total_s if total_s > 0 else 0 for d in target]), target,
#                 all_source_domains)
#
#             fp.write(f"\n{'=' * 120}\nTARGET: {target} | TRUE RATIOS: {np.round(true_r, 4)}\n{'=' * 120}\n")
#             fp.write(
#                 f"{'Solver':<18} | {'Epsilon Mult':<15} | {'Param (D/E)':<15} | {'Acc (%)':<12} | {'Learned Weights'}\n" + "-" * 120 + "\n")
#
#             el = []
#             for d in target:
#                 _, l, _ = Data.get_data_loaders(d, seed=seed);
#                 el.append((d, l))
#
#             Y, D, H = build_DP_model_Classes(el, sum(len(l.dataset) for _, l in el), target, models, classifiers,
#                                              normalize_factors, vae_norm_stats)
#
#             fp.write(run_baselines(Y, D, H, target, target, all_source_domains, seed))
#
#             print(f"   [Solver] Launching parallel workers for {target}...")
#             # Added verbose=10 to see parallel job completion in console
#             results = Parallel(n_jobs=2, verbose=10)(
#                 delayed(run_solver_sweep_worker)(Y, D, H, e, target, all_source_domains) for e in [1.0, 1.1])
#
#             for r in results: fp.write(r)
#             fp.flush()
#             print(f"✅ Finished target: {target}")
#
#
# def main():
#     print(f"Starting Main Process | Dataset Mode: {DATASET_MODE}")
#     classifiers = {}
#     for d in ALL_DOMAINS_LIST:
#         m = models.resnet50(weights=None);
#         m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
#         path = f"./classifiers/{d}_224.pt"
#         if not os.path.exists(path):
#             path = f"./classifiers_new/{d}_classifier.pt"
#         m.load_state_dict(torch.load(path, map_location=device))
#         classifiers[d] = FeatureExtractor(m).to(device).eval()
#         print(f"✅ Loaded Classifier: {d}")
#
#     task_run(classifiers, ALL_DOMAINS_LIST)
#
#
# if __name__ == "__main__":
#     main()

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
    from old_cvxpy_solver import solve_convex_problem_mosek
    from cvxpy_solver_per_domain import solve_convex_problem_per_domain
except ImportError:
    pass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "1")
torch.set_num_threads(1)

# ==========================================
# --- CONFIGURATION ---
# ==========================================
DATASET_MODE = "OFFICE224"

CONFIGS = {
    "DIGITS": {
        "DOMAINS": ['MNIST', 'USPS', 'SVHN'],
        "CLASSES": 10,
        "INPUT_DIM": 2048,
        "SOURCE_ERRORS": {'MNIST': 0.005, 'USPS': 0.027, 'SVHN': 0.05},
        "TEST_SET_SIZES": {'MNIST': 10000, 'USPS': 2007, 'SVHN': 26032}
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
        "TEST_SET_SIZES": {'Art': 486, 'Clipart': 873, 'Product': 888, 'Real World': 870}
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
    if DATASET_MODE in ["OFFICE", "OFFICE31", "OFFICE224"]:
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
    for dom_name, loader in data_loaders:
        print(f" [DATA] Processing {dom_name} batch extraction...")
        for data, label in loader:
            data = data.to(device)
            data = fix_batch_resnet(data)
            N = len(data)
            one_hot = np.zeros((N, C))
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


def run_baselines(Y, D, H, source_domains, target_domains, all_source_domains, seed):
    print(" [Baseline] Running Oracle and Uniform...")
    buf = io.StringIO()
    total = sum([TEST_SET_SIZES.get(d, 1) for d in target_domains])
    oracle_z = np.array([TEST_SET_SIZES.get(s, 0) / total if s in target_domains else 0.0 for s in source_domains])
    uniform_w = np.ones(len(source_domains)) / len(source_domains)
    for name, w in [("ORACLE", oracle_z), ("UNIFORM", uniform_w)]:
        acc = evaluate_accuracy(w, D, H, Y)
        w_f = map_weights_to_full_source_list(w, source_domains, all_source_domains)
        buf.write(f"{name:<18} | {'N/A':<15} | {'N/A':<15} | {acc:<12.2f} | {str(np.round(w_f, 4))}\n")
        print(f" >>> [Baseline] {name:<7} | Acc: {acc:.2f}%")

    print(" [Baseline] Running DC Solver...")
    dc_accuracies, best_z_dc = [], None
    for i in range(1):
        try:
            dp = init_problem_from_model(Y, D, H, p=len(source_domains), C=NUM_CLASSES)
            slv = ConvexConcaveSolver(ConvexConcaveProblem(dp), seed + (i * 100), "err")
            z_dc, _, _ = slv.solve()
            if z_dc is not None:
                acc = evaluate_accuracy(z_dc, D, H, Y)
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


def run_solver_sweep_worker(Y, D, H, eps_mult, source_domains, all_source_domains):
    print(f" [Worker] Starting sweep for Epsilon Mult: {eps_mult}")
    buf = io.StringIO()
    errors = np.array([(SOURCE_ERRORS.get(d, 0.1) + 0.05) * eps_mult for d in source_domains])
    max_ent = np.log(len(source_domains)) if len(source_domains) > 1 else 0.1
    for solver in ["CVXPY_GLOBAL"]:#, "CVXPY_PER_DOMAIN"]:
        for mult in [1.0, 1.2]:
            delta = mult * max_ent
            try:
                if solver == "CVXPY_GLOBAL":
                    w = solve_convex_problem_mosek(Y, D, H, delta=delta, epsilon=max(errors), solver_type='SCS')
                    if w is None:
                        print("[CVXPY] Returned None (likely infeasible)")
                        continue

                else:
                    w = solve_convex_problem_per_domain(Y, D, H, delta=np.full(len(source_domains), delta),
                                                        epsilon=errors, solver_type='SCS')
                acc = evaluate_accuracy(w, D, H, Y)
                w_f = map_weights_to_full_source_list(w, source_domains, all_source_domains)
                buf.write(f"{solver:<18} | m:{eps_mult:<13} | m:{mult:<13} | {acc:<12.2f} | {str(np.round(w_f, 4))}\n")
            except Exception as e:
                print("[CVXPY ERROR]", e)
    return buf.getvalue()


def evaluate_accuracy(w, D, H, Y):
    preds = ((D * H) * w.reshape(1, 1, -1)).sum(axis=2)
    return accuracy_score(Y.argmax(axis=1), preds.argmax(axis=1)) * 100.0


def task_run(classifiers, all_source_domains):
    seed = 1
    torch.manual_seed(seed)
    np.random.seed(seed)
    test_path = f'./results_{DATASET_MODE}_recreate/seed_{seed}/'
    os.makedirs(test_path, exist_ok=True)

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
            for j, (imgs, _) in enumerate(loader):
                if j > 10:
                    break
                f, _ = classifiers[d](fix_batch_resnet(imgs.to(device)))
                feats.append(f.cpu())

        cat_f = torch.cat(feats, 0).to(device)
        vae_norm_stats[d] = (cat_f.min(), cat_f.max())
        vae_in = torch.clamp((cat_f - cat_f.min()) / (cat_f.max() - cat_f.min() + 1e-6), 0, 1)
        out, _, _ = m(vae_in)
        lp = m.compute_log_probabitility_bernoulli(out, vae_in)
        normalize_factors[d] = (lp.mean(), lp.std())
        print(f"✅ Loaded VRS for {d}")

    print("\n--- Starting Target Combinations ---")
    with open(os.path.join(test_path, f'Sweep_Results_{seed}.txt'), 'a') as fp:
        for target in [list(s) for r in range(3, len(all_source_domains) + 1) for s in
                       itertools.combinations(all_source_domains, r)]:

            # if set(target) == {'Art', 'Clipart', 'Product'}:
            #     print(f"⏩ Skipping excluded target: {target}")
            #     continue

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
                for e in [1.0, 1.1]
            )

            for r in results:
                fp.write(r)
            fp.flush()
            print(f"✅ Finished target: {target}")


def main():
    print(f"Starting Main Process | Dataset Mode: {DATASET_MODE}")
    classifiers = {}
    for d in ALL_DOMAINS_LIST:
        m = models.resnet50(weights=None)
        m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
        path = f"./classifiers/{d}_224.pt"
        if not os.path.exists(path):
            path = f"./classifiers_new/{d}_classifier.pt"
        m.load_state_dict(torch.load(path, map_location=device))
        classifiers[d] = FeatureExtractor(m).to(device).eval()
        print(f"✅ Loaded Classifier: {d}")

    task_run(classifiers, ALL_DOMAINS_LIST)


if __name__ == "__main__":
    main()
