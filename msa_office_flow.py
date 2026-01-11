from __future__ import print_function
import torch
import numpy as np
import os
import sys
import io
import itertools
import torch.nn.functional as F
from torch import nn
from torchvision import models, transforms, datasets
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

# --- IMPORTS FOR SOLVERS ---
try:
    from dc import *
    from cvxpy_solver import solve_convex_problem_mosek
    from cvxpy_solver_per_domain import solve_convex_problem_per_domain
except ImportError:
    print("[WARNING] Solvers (dc/cvxpy) not found in current path.")
    pass

# ==========================================
# --- CONFIGURATION ---
# ==========================================

ROOT_DIR = "/data/nogaz/Convex_bounds_optimization/LatentFlow_Pixel_Experiments"
OFFICE_DIR = "/data/nogaz/Convex_bounds_optimization/Office-31"  # Path to raw images
D_MATRIX_NAME = "D_Matrix_LatentFlow.npy"
D_MATRIX_PATH = os.path.join(ROOT_DIR, "results", D_MATRIX_NAME)
RESULTS_DIR = os.path.join(ROOT_DIR, "Results_MSA")
os.makedirs(RESULTS_DIR, exist_ok=True)

DOMAINS = ['amazon', 'dslr', 'webcam']
NUM_CLASSES = 31
SOURCE_ERRORS = {'amazon': 0.1352, 'dslr': 0.0178, 'webcam': 0.0225}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- CUSTOM RATIOS CONFIGURATION ---
# Added entry for 3 domains as an example (equal split if not specified)
TARGET_RATIOS = {
    ('amazon', 'dslr'): {'amazon': 0.25, 'dslr': 0.75},
    ('amazon', 'webcam'): {'amazon': 0.70, 'webcam': 0.30},
    ('dslr', 'webcam'): {'dslr': 0.50, 'webcam': 0.50},
    ('amazon', 'dslr', 'webcam'): {'amazon': 0.33, 'dslr': 0.33, 'webcam': 0.34}
}

print(f"==================================================")
print(f"Running MSA Solver Optimization (Delta Loop Mode)")
print(f"==================================================")
print(f"   -> Mode: Latent Flow Matching (Pixel)")
print(f"   -> D Matrix: {D_MATRIX_PATH}")
print(f"   -> Results:  {RESULTS_DIR}")
print(f"==================================================\n")


# ==========================================
# 1. HELPERS & LOADING
# ==========================================

class ClassifierHeadOnly(nn.Module):
    def __init__(self, original_model):
        super(ClassifierHeadOnly, self).__init__()
        self.head = original_model.fc

    def forward(self, x):
        return self.head(x)


def get_test_loader(domain, batch_size=64):
    path = os.path.join(OFFICE_DIR, domain, 'images')
    if not os.path.exists(path):
        path = os.path.join(OFFICE_DIR, domain)

    tr = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    try:
        ds = datasets.ImageFolder(path, transform=tr)
    except FileNotFoundError:
        print(f"   [ERROR] Data not found at {path}")
        return None

    N = len(ds)
    rng = np.random.RandomState(42)  # FIXED SEED
    indices = rng.permutation(N)
    test_idx = indices[int(0.8 * N):]
    test_ds = torch.utils.data.Subset(ds, test_idx)

    return DataLoader(test_ds, batch_size=batch_size, shuffle=False)


def load_data_and_compute_matrices():
    print(f"[INIT] Loading Global D Matrix...")
    if not os.path.exists(D_MATRIX_PATH):
        raise FileNotFoundError(f"D Matrix not found: {D_MATRIX_PATH}")

    Global_D = np.load(D_MATRIX_PATH)
    print(f"   -> Global D Shape: {Global_D.shape}")

    # Load Pre-trained Classifiers
    print(f"[INIT] Loading Classifiers...")
    classifiers = {}
    for d in DOMAINS:
        m = models.resnet50(weights=None)
        m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
        path = f"/data/nogaz/Convex_bounds_optimization/classifiers_new/{d}_classifier.pt"
        if not os.path.exists(path):
            path = f"/data/nogaz/Convex_bounds_optimization/classifiers/{d}_224.pt"

        if os.path.exists(path):
            state = torch.load(path, map_location=device, weights_only=True)
            m.load_state_dict(state)
            classifiers[d] = m.to(device).eval()
        else:
            print(f"   [WARN] Classifier for {d} not found!")

    Y_list = []
    H_list_per_domain = {d: [] for d in DOMAINS}
    domain_ranges = {}
    start_idx = 0

    print("   -> Computing Y and H matrices from Raw Images...")

    for dom in DOMAINS:
        loader = get_test_loader(dom)
        if loader is None: continue

        print(f"      Processing domain: {dom}...")
        dom_y_list = []
        dom_h_lists = {k: [] for k in DOMAINS}

        with torch.no_grad():
            for imgs, labels in loader:
                imgs = imgs.to(device)
                N_batch = labels.size(0)
                y_one_hot = np.zeros((N_batch, NUM_CLASSES))
                y_one_hot[np.arange(N_batch), labels.numpy()] = 1
                dom_y_list.append(y_one_hot)

                for k in DOMAINS:
                    if k in classifiers:
                        logits = classifiers[k](imgs)
                        probs = F.softmax(logits, dim=1).cpu().numpy()
                        dom_h_lists[k].append(probs)
                    else:
                        dom_h_lists[k].append(np.ones((N_batch, NUM_CLASSES)) / NUM_CLASSES)

        dom_Y = np.concatenate(dom_y_list, axis=0)
        Y_list.append(dom_Y)

        for k in DOMAINS:
            H_list_per_domain[k].append(np.concatenate(dom_h_lists[k], axis=0))

        N_dom = dom_Y.shape[0]
        domain_ranges[dom] = (start_idx, start_idx + N_dom)
        start_idx += N_dom

    Global_Y = np.concatenate(Y_list, axis=0)
    N_total = Global_Y.shape[0]
    K = len(DOMAINS)
    Global_H = np.zeros((N_total, NUM_CLASSES, K))

    for k_idx, k_name in enumerate(DOMAINS):
        Global_H[:, :, k_idx] = np.concatenate(H_list_per_domain[k_name], axis=0)

    if Global_D.shape[0] != Global_Y.shape[0]:
        print(f"\n[CRITICAL WARNING] Dimension Mismatch! D:{Global_D.shape[0]} Y:{Global_Y.shape[0]}")
        min_len = min(Global_D.shape[0], Global_Y.shape[0])
        Global_D = Global_D[:min_len]
        Global_Y = Global_Y[:min_len]
        Global_H = Global_H[:min_len]

    return Global_D, Global_Y, Global_H, domain_ranges


def slice_matrices_with_ratios(target_domains, Global_D, Global_Y, Global_H, domain_ranges):
    row_indices = []
    avail_sizes = {d: domain_ranges[d][1] - domain_ranges[d][0] for d in target_domains}
    counts = {}
    key_tuple = tuple(target_domains)

    if key_tuple in TARGET_RATIOS:
        ratios = TARGET_RATIOS[key_tuple]
        possible_totals = []
        for d, ratio in ratios.items():
            if ratio > 0:
                possible_totals.append(avail_sizes[d] / ratio)
        max_total = int(min(possible_totals))
        for d in target_domains:
            counts[d] = int(max_total * ratios[d])
    else:
        counts = avail_sizes

    rng = np.random.RandomState(42)
    for dom in target_domains:
        start, end = domain_ranges[dom]
        full_range = np.arange(start, end)
        n_needed = counts[dom]

        if n_needed < len(full_range):
            selected = rng.choice(full_range, size=n_needed, replace=False)
            selected.sort()
        else:
            selected = full_range
        row_indices.extend(selected)

    col_indices = [DOMAINS.index(d) for d in target_domains]

    Y_sub = Global_Y[row_indices]
    D_sub = Global_D[row_indices][:, col_indices]
    H_sub = Global_H[row_indices][:, :, col_indices]

    total_samples = sum(counts.values())
    oracle_w = np.array([counts[d] / total_samples for d in target_domains])

    sum_D = D_sub.sum(axis=0, keepdims=True)
    sum_D[sum_D == 0] = 1.0
    D_sub = D_sub / sum_D

    return Y_sub, D_sub, H_sub, oracle_w


def compute_Q(D, w, eps=1e-12):
    WD = D * w.reshape(1, -1)
    Z = WD.sum(axis=1, keepdims=True)
    Z = np.maximum(Z, eps)
    return WD / Z


def evaluate_accuracy(w, D, H, Y):
    Q = compute_Q(D, w)
    final_preds = (H * Q[:, None, :]).sum(axis=2)
    return accuracy_score(Y.argmax(axis=1), final_preds.argmax(axis=1)) * 100.0


# ==========================================
# 2. DEBUG & WORKER (SERIAL)
# ==========================================

def comprehensive_debug_analysis(buf, Y, D, H, target_domains, source_errors):
    print("\n   >>> COMPREHENSIVE DEBUG ANALYSIS <<<")

    # Simple check for NaN
    if np.any(np.isnan(D)):
        print("   [CRITICAL FAIL] D matrix contains NaN!")
        buf.write("   [CRITICAL FAIL] D matrix contains NaN!\n")

    K = len(target_domains)
    w_unif = np.ones(K) / K
    Q_unif = compute_Q(D, w_unif)
    pred_target = (H * Q_unif[:, None, :]).sum(axis=2)
    correct_probs = (pred_target * Y).sum(axis=1)
    empirical_target_loss = 1.0 - np.mean(correct_probs)

    current_source_errors = [source_errors[d] for d in target_domains]
    weighted_source_err = np.dot(w_unif, current_source_errors)
    gap = empirical_target_loss - weighted_source_err

    print(f"   [FEASIBILITY CHECK] Gap: {gap:.4f}")
    return gap


def run_solver_loop(Y, D, H, target_list):
    buf = io.StringIO()
    gap_approx = comprehensive_debug_analysis(buf, Y, D, H, target_list, SOURCE_ERRORS)

    solvers_to_run = []
    if 'solve_convex_problem_mosek' in globals(): solvers_to_run.append("CVXPY_GLOBAL")
    if 'solve_convex_problem_per_domain' in globals(): solvers_to_run.append("CVXPY_PER_DOMAIN")

    if not solvers_to_run:
        return "No CVXPY solvers found.\n"

    # --- PARAMETER SWEEP ---
    # 1. Epsilon Multipliers
    eps_multipliers = [1.1, 1.2, 1.5, 2.0]

    # 2. Delta Multipliers (As requested)
    delta_multipliers = [1.2, 1.5, 2.0, 4.0]

    D_expanded = np.tile(D[:, None, :], (1, NUM_CLASSES, 1))

    for eps_mult in eps_multipliers:
        errors = np.array([(SOURCE_ERRORS.get(d, 0.1) + 0.05) * eps_mult for d in target_list])
        max_err_input = np.max(errors)

        K = len(target_list)
        max_ent = np.log(K) if K > 1 else 0.1

        for d_mult in delta_multipliers:
            delta = d_mult * max_ent

            for solver in solvers_to_run:
                print(f"   [SOLVER] {solver} | EpsMult: {eps_mult} | Delta: {delta:.2f} ({d_mult}x)...")

                try:
                    if solver == "CVXPY_GLOBAL":
                        w = solve_convex_problem_mosek(Y, D_expanded, H, delta=delta, epsilon=max_err_input,
                                                       solver_type='SCS')
                    else:
                        w = solve_convex_problem_per_domain(Y, D_expanded, H, delta=np.full(K, delta), epsilon=errors,
                                                            solver_type='SCS')

                    if w is not None:
                        acc = evaluate_accuracy(w, D, H, Y)
                        w_str = ", ".join([f"{val:.3f}" for val in w])
                        # Short format log
                        msg = f"{solver:<18} | m:{eps_mult:<4} | d_x:{d_mult:<3} | Acc: {acc:.2f}% | w: [{w_str}]"
                        print(f"   >>> SUCCESS: {msg}")
                        buf.write(msg + "\n")
                    else:
                        # Only print infeasible to log file to reduce clutter
                        buf.write(f"{solver:<18} | m:{eps_mult:<4} | d_x:{d_mult:<3} | INFEASIBLE\n")

                except Exception as e:
                    err_msg = str(e).replace('\n', ' ')[:50]
                    buf.write(f"{solver:<18} | m:{eps_mult} | ERR: {err_msg}\n")
                    pass

    return buf.getvalue()


def run_baselines_and_dc(Y, D, H, target_domains, oracle_w):
    print("   [Baseline] Running Oracle, Uniform, and DC...")
    buf = io.StringIO()
    K = len(target_domains)

    # --- 1. Oracle & Uniform ---
    w_unif = np.ones(K) / K

    for name, w in [("ORACLE", oracle_w), ("UNIFORM", w_unif)]:
        acc = evaluate_accuracy(w, D, H, Y)
        w_str = str(np.round(w, 4))
        buf.write(f"{name:<18} | {'N/A':<6} | {'N/A':<5} | {acc:<12.2f} | {w_str}\n")
        print(f"    >>> [Baseline] {name:<7} | Acc: {acc:.2f}%")

    # --- 2. DC Solver (Robust 5-Seeds Run) ---
    dc_accuracies = []
    best_w_dc = None
    best_acc = -1

    # Expand D dimensions for DC: (N, K) -> (N, C, K)
    # DC expects dimensions to match for element-wise mult with H
    N, C, K_dim = H.shape
    D_expanded = np.tile(D[:, None, :], (1, C, 1))

    print("    >>> [Baseline] Running DC Solver (5 random seeds)...")
    for i in range(5):
        try:
            # Initialize Problem
            # We use the class from dc.py directly
            dp = init_problem_from_model(Y, D_expanded, H, p=K, C=NUM_CLASSES)
            problem = ConvexConcaveProblem(dp)

            # Solver with different seeds
            current_seed = 42 + (i * 100)
            slv = ConvexConcaveSolver(problem, current_seed, "err")

            # Solve (using heuristic epsilon based on source errors)
            # Note: Assuming solver.solve returns (z, obj, err)
            w_dc, _, _ = slv.solve(delta=1e-4)

            if w_dc is not None:
                acc = evaluate_accuracy(w_dc, D, H, Y)
                dc_accuracies.append(acc)

                if acc > best_acc:
                    best_acc = acc
                    best_w_dc = w_dc
        except Exception as e:
            # print(f"       [DC Seed {i} Failed] {e}") # Uncomment for deep debug
            continue

    if dc_accuracies:
        avg_res = f"{np.mean(dc_accuracies):.2f} +/- {np.std(dc_accuracies):.2f}"
        w_str = str(np.round(best_w_dc, 4))
        # Format: Name | Mult | Delta | Acc | Weights
        buf.write(f"{'DC (5-Seeds)':<18} | {'N/A':<6} | {'N/A':<5} | {avg_res:<12} | {w_str}\n")
        print(f"    >>> [Baseline] DC Best  | Acc: {best_acc:.2f}% (Avg: {np.mean(dc_accuracies):.2f})")
    else:
        buf.write(f"{'DC (5-Seeds)':<18} | {'N/A':<6} | {'N/A':<5} | {'FAILED':<12} | []\n")
        print(f"    >>> [Baseline] DC Solver Failed on all seeds.")


    return buf.getvalue()
# ==========================================
# 3. MAIN
# ==========================================
def main():
    Global_D, Global_Y, Global_H, domain_ranges = load_data_and_compute_matrices()

    results_path = os.path.join(
        RESULTS_DIR, "Results_MSA_LatentFlow_v3.txt"
    )

    combinations = []
    # 2 Domains
    # combinations.extend(list(itertools.combinations(DOMAINS, 2)))
    combinations = [c for c in itertools.combinations(DOMAINS, 2)] # if set(c) != {'amazon', 'dslr'}]
    # 3 Domains (Now Active)
    combinations.extend(list(itertools.combinations(DOMAINS, 3)))

    print(f"\n[MSA] Starting optimization loop on {len(combinations)} combinations...")

    with open(results_path, "w") as fp:
        fp.write("MSA EXPERIMENT REPORT | Latent Flow Matching | Delta Loop\n")
        fp.write(f"D Matrix: {D_MATRIX_NAME}\n")
        fp.write("=" * 100 + "\n\n")

        for target_tuple in combinations:
            target_list = list(target_tuple)
            print(f"\n>>> Processing Target Mix: {target_list}")

            Y_sub, D_sub, H_sub, oracle_w = slice_matrices_with_ratios(
                target_list, Global_D, Global_Y, Global_H, domain_ranges
            )

            header = f"TARGET MIXTURE: {target_list} (Samples: {Y_sub.shape[0]})"
            print(header)
            fp.write(header + "\n")
            fp.write("-" * len(header) + "\n")

            # --- Diagnostics ---
            preds = [H_sub[:, :, k].argmax(axis=1) for k in range(H_sub.shape[2])]
            agree_all = np.ones(len(preds[0]), dtype=bool)
            for k in range(1, len(preds)):
                agree_all &= (preds[k] == preds[0])
            agree_rate = np.mean(agree_all)

            w_skewed = np.zeros(len(target_list));
            w_skewed[0] = 0.9;
            w_skewed[1:] = 0.1 / (len(target_list) - 1)
            Q1 = compute_Q(D_sub, np.ones(len(target_list)) / len(target_list))
            Q2 = compute_Q(D_sub, w_skewed)
            q_diff = np.mean(np.abs(Q1 - Q2))

            fp.write(f"[DIAG] Agreement Rate: {agree_rate:.4f}\n")
            fp.write(f"[DIAG] Q Sensitivity:  {q_diff:.4f}\n\n")

            # --- Run ---
            base_res = run_baselines_and_dc(Y_sub, D_sub, H_sub, target_list, oracle_w)
            fp.write(base_res)

            if 'solve_convex_problem_mosek' in globals():
                res_str = run_solver_loop(Y_sub, D_sub, H_sub, target_list)
                fp.write(res_str)
            else:
                fp.write("CVXPY Solvers not found.\n")

            fp.write("\n" + "=" * 100 + "\n\n")
            fp.flush()

    print(f"\n[DONE] Results saved to {results_path}")


if __name__ == "__main__":
    main()