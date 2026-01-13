from __future__ import print_function
import torch
import numpy as np
import os
import io
import itertools
import torch.nn.functional as F
from torch import nn
from torchvision import models, transforms, datasets
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
import cvxpy as cp

# ============================================================
# IMPORTS FOR DC SOLVER (Keep external if needed)
# ============================================================
try:
    from dc import *
except ImportError:
    print("[WARNING] DC solver (dc.py) not found in current path.")
    pass

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

TARGET_RATIOS = {
    ('amazon', 'dslr'): {'amazon': 0.25, 'dslr': 0.75},
    ('amazon', 'webcam'): {'amazon': 0.70, 'webcam': 0.30},
    ('dslr', 'webcam'): {'dslr': 0.50, 'webcam': 0.50},
    ('amazon', 'dslr', 'webcam'): {'amazon': 0.33, 'dslr': 0.33, 'webcam': 0.34}
}

print(f"==================================================")
print(f"Running MSA Solver Optimization (Specific Solvers 3.21/3.22/3.23)")
print(f"==================================================")


# ============================================================
# SHARED HELPERS FOR SOLVERS
# ============================================================

def _flatten_if_needed(Y, D, H):
    """
    Supports either:
      - D: (N, k), H: (N, k), Y: (N,)
    or
      - D: (N0, C, k), H: (N0, C, k), Y: (N0, C)
    Returns flattened (Y, D, H) and (N, k).
    """
    if D.ndim == 3:
        _, _, k = D.shape
        D = D.reshape(-1, k)
        H = H.reshape(-1, k)
        Y = Y.reshape(-1)
    N, k = D.shape
    return Y, D, H, N, k


def calculate_expected_loss(Y, H, D, num_sources, eps=1e-12):
    """
    L_mat[j,t] = E_{x ~ (weights proportional to D[:,t])} [ loss_j(x) ]
    where D[:,t] is p(t|x) but we normalize across samples to get a proper weighting.
    """
    L_mat = np.zeros((num_sources, num_sources))

    for j in range(num_sources):
        loss_vec = np.abs(H[:, :, j] - Y) if H.ndim == 3 else np.abs(H[:, j] - Y)

        if loss_vec.ndim == 2:
            loss_vec = np.sum(loss_vec, axis=1) / 2.0  # (1 - p_true) in [0,1]

        for t in range(num_sources):
            p_t = D[:, t].astype(np.float64)
            Z = p_t.sum()
            if Z < eps:
                # אין מסה לדומיין הזה -> לא משפיע, תני 0 או NaN לפי מה שנוח לך
                L_mat[j, t] = 0.0
            else:
                p_t = p_t / Z     # <-- זה השינוי הקריטי
                L_mat[j, t] = np.sum(p_t * loss_vec)

    w_unif = np.ones(k) / k
    print("[SANITY] L_mat range:", L_mat.min(), L_mat.max())
    print("[SANITY] max_t (w_unif @ L_mat[:,t]) =", np.max(w_unif @ L_mat))

    return L_mat


def compute_p_tilde(D, eps=1e-15):
    """ Domain-anchored normalized expert density """
    row_sum = np.sum(D, axis=1, keepdims=True)
    return D / np.maximum(row_sum, eps)


# ============================================================
# SOLVER 3.21: Smoothed KL MSDA
# ============================================================
def solve_convex_problem_smoothed_kl(Y, D, H, epsilon=1e-2, eta=1e-2, solver_type="SCS",
                                     q_min=1e-12, w_min=1e-12, scs_eps=1e-4, scs_max_iters=20000):
    # Pre-processing to avoid logic errors with shapes
    # If D is (N, K) and H is (N, C, K), we handle loss calculation carefully above.
    N, k = D.shape
    L_mat = calculate_expected_loss(Y, H, D, k)

    w = cp.Variable(k, nonneg=True, name="w")
    Q = cp.Variable((N, k), nonneg=True, name="Q")

    main_terms = []
    for j in range(k):
        main_terms.append(cp.sum(cp.multiply(D[:, j], -cp.rel_entr(w[j], Q[:, j]))))

    main_obj = cp.sum(main_terms)
    smooth_obj = (eta / (k * N)) * cp.sum(cp.log(Q))

    objective = cp.Maximize(main_obj + smooth_obj)

    constraints = [
        cp.sum(w) == 1,
        w >= w_min,
        cp.sum(Q, axis=1) == 1,
        Q >= q_min,
    ]

    for t in range(k):
        constraints.append(w @ L_mat[:, t] <= epsilon)

    prob = cp.Problem(objective, constraints)

    # Solvers
    try:
        if solver_type == "SCS":
            prob.solve(solver=cp.SCS, verbose=False, eps=scs_eps, max_iters=scs_max_iters)
        elif solver_type == "MOSEK":
            prob.solve(solver=cp.MOSEK, verbose=False)
        else:
            prob.solve(solver=cp.ECOS, verbose=False)
    except:
        return None, None

    if prob.status not in ["optimal", "optimal_inaccurate"]:
        return None, None

    return np.asarray(w.value).reshape(-1), np.asarray(Q.value)


# ============================================================
# SOLVER 3.22: Domain-Anchored Smoothed
# ============================================================
def solve_convex_problem_domain_anchored_smoothed(Y, D, H, epsilon=1e-2, eta=1e-2, solver_type="SCS",
                                                  q_min=1e-12, w_min=1e-12, scs_eps=1e-4, scs_max_iters=20000,
                                                  normalize_domains=True, ptilde_eps=1e-15):
    N, k = D.shape
    if normalize_domains:
        D_norm = compute_p_tilde(D, eps=ptilde_eps)
    else:
        D_norm = D

    L_mat = calculate_expected_loss(Y, H, D, k)  # Use original D for loss weighting usually? Or normalized?
    # Using D as per your snippet logic where D was passed in.

    p_tilde = compute_p_tilde(D, eps=ptilde_eps)

    w = cp.Variable(k, nonneg=True, name="w")
    Q = cp.Variable((N, k), nonneg=True, name="Q")

    main_terms = []
    for j in range(k):
        main_terms.append(cp.sum(cp.multiply(D[:, j], -cp.rel_entr(w[j], Q[:, j]))))
    main_obj = cp.sum(main_terms)

    anchored_smooth_obj = (eta / k) * cp.sum(cp.multiply(p_tilde, cp.log(Q)))

    objective = cp.Maximize(main_obj + anchored_smooth_obj)

    constraints = [
        cp.sum(w) == 1,
        w >= w_min,
        cp.sum(Q, axis=1) == 1,
        Q >= q_min,
    ]
    for t in range(k):
        constraints.append(w @ L_mat[:, t] <= epsilon)

    prob = cp.Problem(objective, constraints)

    try:
        if solver_type == "SCS":
            prob.solve(solver=cp.SCS, verbose=False, eps=scs_eps, max_iters=scs_max_iters)
        elif solver_type == "MOSEK":
            prob.solve(solver=cp.MOSEK, verbose=False)
        else:
            prob.solve(solver=cp.ECOS, verbose=False)
    except:
        return None, None

    if prob.status not in ["optimal", "optimal_inaccurate"]:
        return None, None

    return np.asarray(w.value).reshape(-1), np.asarray(Q.value)


# ============================================================
# SOLVER 3.23: Smoothed Original P
# ============================================================
def solve_convex_problem_smoothed_original_p(Y, D, H, epsilon=1e-2, eta=1e-2, solver_type="SCS",
                                             q_min=1e-12, w_min=1e-12, scs_eps=1e-4, scs_max_iters=20000):
    N, k = D.shape
    L_mat = calculate_expected_loss(Y, H, D, k)

    w = cp.Variable(k, nonneg=True, name="w")
    Q = cp.Variable((N, k), nonneg=True, name="Q")

    main_terms = []
    for j in range(k):
        main_terms.append(cp.sum(cp.multiply(D[:, j], -cp.rel_entr(w[j], Q[:, j]))))
    main_obj = cp.sum(main_terms)

    smooth_obj = eta * cp.sum(cp.multiply(D, cp.log(Q)))

    objective = cp.Maximize(main_obj + smooth_obj)

    constraints = [
        cp.sum(w) == 1,
        w >= w_min,
        cp.sum(Q, axis=1) == 1,
        Q >= q_min,
    ]

    for t in range(k):
        constraints.append(w @ L_mat[:, t] <= epsilon)

    prob = cp.Problem(objective, constraints)

    try:
        if solver_type == "SCS":
            prob.solve(solver=cp.SCS, verbose=False, eps=scs_eps, max_iters=scs_max_iters)
        elif solver_type == "MOSEK":
            prob.solve(solver=cp.MOSEK, verbose=False)
        else:
            prob.solve(solver=cp.ECOS, verbose=False)
    except:
        return None, None

    if prob.status not in ["optimal", "optimal_inaccurate"]:
        return None, None

    return np.asarray(w.value).reshape(-1), np.asarray(Q.value)


# ==========================================
# HELPERS & LOADING (STANDARD)
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
    except FileNotFoundError:
        return None
    N = len(ds)
    rng = np.random.RandomState(42)
    indices = rng.permutation(N)
    test_idx = indices[int(0.8 * N):]
    test_ds = torch.utils.data.Subset(ds, test_idx)
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


def slice_matrices_with_ratios(target_domains, Global_D, Global_Y, Global_H, domain_ranges):
    # (Same as before)
    avail_sizes = {d: domain_ranges[d][1] - domain_ranges[d][0] for d in target_domains}
    counts = {}
    key_tuple = tuple(target_domains)
    if key_tuple in TARGET_RATIOS:
        ratios = TARGET_RATIOS[key_tuple]
        possible_totals = [avail_sizes[d] / ratios[d] for d in ratios if ratios[d] > 0]
        max_total = int(min(possible_totals))
        for d in target_domains: counts[d] = int(max_total * ratios[d])
    else:
        counts = avail_sizes

    row_indices = []
    rng = np.random.RandomState(42)
    for dom in target_domains:
        start, end = domain_ranges[dom]
        full_range = np.arange(start, end)
        n_needed = counts[dom]
        selected = rng.choice(full_range, size=n_needed, replace=False) if n_needed < len(full_range) else full_range
        selected.sort()
        row_indices.extend(selected)

    col_indices = [DOMAINS.index(d) for d in target_domains]
    Y_sub = Global_Y[row_indices]
    D_sub = Global_D[row_indices][:, col_indices]
    H_sub = Global_H[row_indices][:, :, col_indices]

    total = sum(counts.values())
    oracle_w = np.array([counts[d] / total for d in target_domains])

    sum_D = D_sub.sum(axis=0, keepdims=True)
    sum_D[sum_D == 0] = 1.0
    D_sub = D_sub / sum_D

    return Y_sub, D_sub, H_sub, oracle_w


def compute_Q(D, w, eps=1e-12):
    WD = D * w.reshape(1, -1)
    Z = WD.sum(axis=1, keepdims=True)
    return WD / np.maximum(Z, eps)


def evaluate_accuracy(w, D, H, Y):
    Q = compute_Q(D, w)
    final_preds = (H * Q[:, None, :]).sum(axis=2)
    return accuracy_score(Y.argmax(axis=1), final_preds.argmax(axis=1)) * 100.0


# ==========================================
# MAIN EXECUTION LOOPS
# ==========================================

def run_cvxpy_smoothed_loop(Y, D, H, target_list):
    buf = io.StringIO()

    solvers = [
        ("P3.21", solve_convex_problem_smoothed_kl),
        ("P3.22", solve_convex_problem_domain_anchored_smoothed),
        ("P3.23", solve_convex_problem_smoothed_original_p),
    ]

    eps_multipliers = [1.1, 1.2, 1.5, 2.0, 4, 7, 10, 20]
    eta_values = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 0.1, 0.2]

    for eps_mult in eps_multipliers:
        errors = np.array([(SOURCE_ERRORS.get(d, 0.1) + 0.05) * eps_mult for d in target_list])
        epsilon = float(np.max(errors))

        for eta in eta_values:
            for solver_name, solver_fn in solvers:
                print(f"   [CVXPY] {solver_name} | EpsMult: {eps_mult} | Epsilon: {epsilon:.4f} | Eta: {eta:g}")
                try:
                    w, _Q = solver_fn(Y, D, H, epsilon=epsilon, eta=eta, solver_type='SCS')
                    if w is not None:
                        acc = evaluate_accuracy(w, D, H, Y)
                        w_str = ", ".join([f"{val:.3f}" for val in w])
                        msg = f"{solver_name:<6} | m:{eps_mult:<4} | eta:{eta:<8g} | Acc: {acc:.2f}% | w: [{w_str}]"
                        print(f"   >>> SUCCESS: {msg}")
                        buf.write(msg + "\n")
                    else:
                        buf.write(f"{solver_name:<6} | m:{eps_mult:<4} | eta:{eta:<8g} | INFEASIBLE\n")
                except Exception as e:
                    buf.write(f"{solver_name:<6} | m:{eps_mult} | ERR: {str(e)[:50]}\n")

    return buf.getvalue()


def run_baselines_and_dc(Y, D, H, target_domains, oracle_w):
    buf = io.StringIO()
    K = len(target_domains)

    # Oracle & Uniform
    w_unif = np.ones(K) / K
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
    results_path = os.path.join(RESULTS_DIR, "Results_MSA_LatentFlow_Solvers_3.21_3.22_3.23_T8.txt")

    combinations = []
    combinations.extend(list(itertools.combinations(DOMAINS, 2)))
    combinations.extend(list(itertools.combinations(DOMAINS, 3)))

    with open(results_path, "w") as fp:
        fp.write("MSA REPORT | Solvers 3.21, 3.22, 3.23\n=================================\n")

        for target_tuple in combinations:
            target_list = list(target_tuple)
            print(f"\n>>> Processing: {target_list}")
            Y_sub, D_sub, H_sub, oracle_w = slice_matrices_with_ratios(target_list, Global_D, Global_Y, Global_H,
                                                                       domain_ranges)

            header = f"TARGET: {target_list} (N={Y_sub.shape[0]})"
            fp.write(header + "\n" + "-" * len(header) + "\n")

            fp.write(run_baselines_and_dc(Y_sub, D_sub, H_sub, target_list, oracle_w))
            fp.write(run_cvxpy_smoothed_loop(Y_sub, D_sub, H_sub, target_list))
            fp.write("\n\n")
            fp.flush()

    print(f"\n[DONE] Results: {results_path}")


if __name__ == "__main__":
    main()