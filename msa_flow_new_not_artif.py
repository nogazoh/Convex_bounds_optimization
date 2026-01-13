from __future__ import print_function
import torch
import numpy as np
import os
import io
import itertools
from torchvision import models, transforms, datasets
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F

# ============================================================
# 1. IMPORTS
# ============================================================
try:
    from dc import *
except ImportError:
    print("[WARNING] DC solver (dc.py) not found.")

try:
    # 3.21, 3.22, 3.23 (Smoothed)
    from cvxpy_3_21 import solve_convex_problem_smoothed_kl
    from cvxpy_3_22 import solve_convex_problem_domain_anchored_smoothed
    from cvxpy_3_23 import solve_convex_problem_smoothed_original_p

    # 3.10 (Explicit R variable)
    from cvxpy_solver import solve_convex_problem_mosek
except ImportError:
    print("[ERROR] One of the CVXPY solver files is missing or failed to import.")
    exit(1)

# ==========================================
# --- CONFIGURATION ---
# ==========================================
ROOT_DIR = "/data/nogaz/Convex_bounds_optimization/LatentFlow_Pixel_Experiments"
OFFICE_DIR = "/data/nogaz/Convex_bounds_optimization/Office-31"
D_MATRIX_NAME = "D_Matrix_LatentFlow_T8.npy"
D_MATRIX_PATH = os.path.join(ROOT_DIR, "results", D_MATRIX_NAME)
RESULTS_DIR = os.path.join(ROOT_DIR, "Results_MSA")
os.makedirs(RESULTS_DIR, exist_ok=True)

DOMAINS = ['amazon', 'dslr', 'webcam']
NUM_CLASSES = 31
SOURCE_ERRORS = {'amazon': 0.1352, 'dslr': 0.0178, 'webcam': 0.0225}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"==================================================")
print(f"Running MSA Solver Optimization (LODO Mode)")
print(f"   -> Including P3.10, P3.21, P3.22, P3.23")
print(f"==================================================\n")


# ==========================================
# 2. HELPERS & LOADING
# ==========================================

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
    except:
        return None
    N = len(ds)
    rng = np.random.RandomState(42)
    indices = rng.permutation(N)
    test_ds = torch.utils.data.Subset(ds, indices[int(0.8 * N):])
    return DataLoader(test_ds, batch_size=batch_size, shuffle=False)


def load_data_and_compute_matrices():
    print(f"[INIT] Loading Global D Matrix...")
    Global_D = np.load(D_MATRIX_PATH)
    print(f"   -> Global D Shape: {Global_D.shape}")

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

    Y_list = []
    H_list_per_domain = {d: [] for d in DOMAINS}
    domain_ranges = {}
    start_idx = 0

    print("   -> Computing Y and H matrices...")
    for dom in DOMAINS:
        loader = get_test_loader(dom)
        if loader is None: continue
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
    Global_H = np.zeros((N_total, NUM_CLASSES, len(DOMAINS)))
    for k_idx, k_name in enumerate(DOMAINS):
        Global_H[:, :, k_idx] = np.concatenate(H_list_per_domain[k_name], axis=0)

    if Global_D.shape[0] != Global_Y.shape[0]:
        min_len = min(Global_D.shape[0], Global_Y.shape[0])
        Global_D = Global_D[:min_len]
        Global_Y = Global_Y[:min_len]
        Global_H = Global_H[:min_len]

    return Global_D, Global_Y, Global_H, domain_ranges


def compute_Q(D, w, eps=1e-12):
    WD = D * w.reshape(1, -1)
    Z = WD.sum(axis=1, keepdims=True)
    return WD / np.maximum(Z, eps)


def evaluate_accuracy(w, D, H, Y):
    Q = compute_Q(D, w)
    final_preds = (H * Q[:, None, :]).sum(axis=2)
    return accuracy_score(Y.argmax(axis=1), final_preds.argmax(axis=1)) * 100.0


# ==========================================
# 3. EXECUTION LOOPS
# ==========================================

def run_cvxpy_mixed_loop(Y, D, H, target_list):
    buf = io.StringIO()

    # Smoothed solvers (return w, Q)
    smoothed_solvers = [
        ("P3.21", solve_convex_problem_smoothed_kl),
        ("P3.22", solve_convex_problem_domain_anchored_smoothed),
        ("P3.23", solve_convex_problem_smoothed_original_p),
    ]

    eps_multipliers = [1.5, 2.0, 4, 7, 10, 20, 30, 50]
    # 'val' will be 'eta' for 3.21-3.23, and 'delta' for 3.10
    param_values = [5e-4, 5e-3, 1e-2, 0.1, 0.2, 0.5, 0.8, 1]

    for eps_mult in eps_multipliers:
        # Calculate epsilon vector per domain
        epsilon_vec = np.array([
            (SOURCE_ERRORS[d] + 0.05) * eps_mult
            for d in target_list
        ])

        # Format string for logs
        eps_str = ", ".join([f"{epsilon_vec[i]:.2f}" for i in range(len(target_list))])

        for val in param_values:

            # --- RUN 3.10 (Explicit R, uses delta) ---
            solver_name = "P3.10"
            try:
                # 3.10 returns only w. 'val' is passed as 'delta'
                w = solve_convex_problem_mosek(
                    Y, D, H,
                    delta=val,  # delta = parameter value
                    epsilon=epsilon_vec,
                    solver_type='SCS'
                )

                if w is not None:
                    acc = evaluate_accuracy(w, D, H, Y)
                    w_str = ", ".join([f"{x:.3f}" for x in w])
                    msg = f"{solver_name:<6} | eps:[{eps_str}] | delta:{val:<6g} | Acc:{acc:6.2f}% | w:[{w_str}]"
                    print(f"   >>> SUCCESS: {msg}")
                    buf.write(msg + "\n")
                else:
                    buf.write(f"{solver_name:<6} | eps:[{eps_str}] | delta:{val:<6g} | INFEASIBLE\n")
            except Exception as e:
                buf.write(f"{solver_name:<6} | eps:[{eps_str}] | ERR: {str(e)[:50]}\n")

            # --- RUN 3.21 / 3.22 / 3.23 (Smoothed, uses eta) ---
            for solver_name, solver_fn in smoothed_solvers:
                try:
                    # These return (w, Q). 'val' is passed as 'eta'
                    w, _Q = solver_fn(
                        Y, D, H,
                        epsilon=epsilon_vec,
                        eta=val,  # eta = parameter value
                        solver_type='SCS'
                    )

                    if w is not None:
                        acc = evaluate_accuracy(w, D, H, Y)
                        w_str = ", ".join([f"{x:.3f}" for x in w])
                        msg = f"{solver_name:<6} | eps:[{eps_str}] | eta:{val:<8g} | Acc:{acc:6.2f}% | w:[{w_str}]"
                        print(f"   >>> SUCCESS: {msg}")
                        buf.write(msg + "\n")
                    else:
                        buf.write(f"{solver_name:<6} | eps:[{eps_str}] | eta:{val:<8g} | INFEASIBLE\n")
                except Exception as e:
                    buf.write(f"{solver_name:<6} | eps:[{eps_str}] | ERR: {str(e)[:50]}\n")

    return buf.getvalue()


def run_baselines_and_dc(Y, D, H, target_domains, oracle_w):
    buf = io.StringIO()
    K = len(target_domains)
    w_unif = np.ones(K) / K

    # Oracle & Uniform
    for name, w in [("ORACLE", oracle_w), ("UNIFORM", w_unif)]:
        acc = evaluate_accuracy(w, D, H, Y)
        w_str = str(np.round(w, 4))
        buf.write(f"{name:<18} | {'N/A':<6} | {'N/A':<5} | {acc:<12.2f} | {w_str}\n")

    # DC Solver
    dc_accuracies, best_w, best_acc = [], None, -1
    N, C, K_dim = H.shape
    D_expanded = np.tile(D[:, None, :], (1, C, 1))

    print("    >>> [Baseline] Running DC Solver (5 random seeds)...")
    for i in range(5):
        try:
            dp = init_problem_from_model(Y, D_expanded, H, p=K, C=NUM_CLASSES)
            problem = ConvexConcaveProblem(dp)
            slv = ConvexConcaveSolver(problem, 42 + (i * 100), "err")
            w_dc, _, _ = slv.solve(delta=1e-4)
            if w_dc is not None:
                acc = evaluate_accuracy(w_dc, D, H, Y)
                dc_accuracies.append(acc)
                if acc > best_acc: best_acc, best_w = acc, w_dc
        except:
            continue

    if dc_accuracies:
        avg = f"{np.mean(dc_accuracies):.2f} +/- {np.std(dc_accuracies):.2f}"
        buf.write(f"{'DC (5-Seeds)':<18} | {'N/A':<6} | {'N/A':<5} | {avg:<12} | {str(np.round(best_w, 4))}\n")
    else:
        buf.write(f"{'DC (5-Seeds)':<18} | FAILED\n")

    return buf.getvalue()


def main():
    Global_D, Global_Y, Global_H, domain_ranges = load_data_and_compute_matrices()

    results_path = os.path.join(
        RESULTS_DIR,
        "Results_MSA_LODO_AllSolvers_T8.txt"
    )

    with open(results_path, "w") as fp:
        fp.write(
            "MSA REPORT | Real LODO Targets | Solvers 3.10 + 3.21/22/23\n"
        )
        fp.write("=" * 80 + "\n\n")

        for target_domain in DOMAINS:
            source_domains = [d for d in DOMAINS if d != target_domain]

            print(f"\n>>> TARGET DOMAIN: {target_domain}")
            print(f"    SOURCES: {source_domains}")

            header = f"TARGET: {target_domain} | SOURCES: {source_domains}"
            fp.write(header + "\n")
            fp.write("-" * len(header) + "\n")

            # 1. Slice TARGET samples
            start_t, end_t = domain_ranges[target_domain]
            row_indices = np.arange(start_t, end_t)
            source_col_indices = [DOMAINS.index(d) for d in source_domains]

            Y_sub = Global_Y[row_indices]
            D_sub = Global_D[row_indices][:, source_col_indices]
            H_sub = Global_H[row_indices][:, :, source_col_indices]


            # 3. Weighted Oracle
            source_sizes = []
            for d in source_domains:
                s, e = domain_ranges[d]
                source_sizes.append(e - s)
            source_sizes = np.array(source_sizes, dtype=float)
            oracle_w = source_sizes / source_sizes.sum()

            print(f"    Source sizes: {dict(zip(source_domains, source_sizes))}")
            print(f"    Oracle w: {oracle_w}")
            fp.write(f"Source sizes: {source_sizes.tolist()} -> Oracle W: {oracle_w.tolist()}\n\n")

            # 4. Baselines + DC
            fp.write(run_baselines_and_dc(Y_sub, D_sub, H_sub, source_domains, oracle_w))

            # 5. Mixed CVXPY Loop (3.10 + 3.2x)
            fp.write(run_cvxpy_mixed_loop(Y_sub, D_sub, H_sub, source_domains))

            fp.write("\n" + "=" * 80 + "\n\n")
            fp.flush()

    print(f"\n[DONE] Results saved to:\n{results_path}")


if __name__ == "__main__":
    main()