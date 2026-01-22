from __future__ import print_function
import torch
import numpy as np
import os
import sys
import io
import itertools
import torch.nn.functional as F
from torch import nn
from torchvision import models
from sklearn.metrics import accuracy_score

# REMOVED JOBLIB TO FIX PRINTING ISSUES
# from joblib import Parallel, delayed

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

ROOT_DIR = "/data/nogaz/Convex_bounds_optimization/VAE_Renyi_Experiments"
EXP_FOLDER = "GAP_UMAP"
BASE_EXP_DIR = os.path.join(ROOT_DIR, EXP_FOLDER)
FEATURES_DIR = os.path.join(BASE_EXP_DIR, "features")
D_MATRIX_NAME = "D_Matrix_GAP.npy"
D_MATRIX_PATH = os.path.join(BASE_EXP_DIR, D_MATRIX_NAME)
RESULTS_DIR = os.path.join(BASE_EXP_DIR, "Results_MSA")
os.makedirs(RESULTS_DIR, exist_ok=True)

DOMAINS = ['amazon', 'dslr', 'webcam']
NUM_CLASSES = 31
SOURCE_ERRORS = {'amazon': 0.1352, 'dslr': 0.0178, 'webcam': 0.0225}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- CUSTOM RATIOS CONFIGURATION ---
TARGET_RATIOS = {
    ('amazon', 'dslr'): {'amazon': 0.25, 'dslr': 0.75},
    ('amazon', 'webcam'): {'amazon': 0.70, 'webcam': 0.30},
    ('dslr', 'webcam'): {'dslr': 0.50, 'webcam': 0.50}
}

print(f"==================================================")
print(f"Running MSA Solver Optimization (SERIAL DEBUG MODE)")
print(f"==================================================")
print(f"   -> Mode: GAP + UMAP")
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


def find_raw_features_dir():
    """Smart lookup for the Raw Features directory"""
    path1 = os.path.join(ROOT_DIR, "Raw_Features_GAP")
    path2 = os.path.join(os.path.dirname(ROOT_DIR), "Raw_Features_GAP")
    path3 = os.path.join(os.path.dirname(ROOT_DIR), "Flow_Experiments", "Raw_Features_GAP")

    if os.path.exists(path1) and os.path.exists(os.path.join(path1, "amazon_test.pt")):
        return path1
    if os.path.exists(path2) and os.path.exists(os.path.join(path2, "amazon_test.pt")):
        return path2
    if os.path.exists(path3) and os.path.exists(os.path.join(path3, "amazon_test.pt")):
        return path3
    return path1


def load_global_matrices():
    print(f"[INIT] Loading Global D Matrix...")
    if not os.path.exists(D_MATRIX_PATH):
        raise FileNotFoundError(f"D Matrix not found: {D_MATRIX_PATH}")

    Global_D = np.load(D_MATRIX_PATH)
    print(f"   -> Global D Shape: {Global_D.shape}")

    # Load Classifiers
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
            classifiers[d] = ClassifierHeadOnly(m).to(device).eval()

    Y_list, all_features = [], []
    domain_ranges = {}
    start_idx = 0

    print("   -> Building Global Y and H matrices...")
    for dom in DOMAINS:
        feat_path = os.path.join(FEATURES_DIR, f"{dom}_test.pt")
        feats, labels = torch.load(feat_path, weights_only=True)
        N = len(labels)
        domain_ranges[dom] = (start_idx, start_idx + N)
        start_idx += N

        y_one_hot = np.zeros((N, NUM_CLASSES))
        y_one_hot[np.arange(N), labels.numpy()] = 1
        Y_list.append(y_one_hot)
        all_features.append(feats)

    Global_Y = np.concatenate(Y_list, axis=0)
    Global_Feats = torch.cat(all_features, dim=0).to(device)

    # Handle UMAP vs Raw features for H calculation
    if Global_Feats.shape[1] != 2048:
        raw_feat_dir = find_raw_features_dir()
        print(f"   [INFO] Loading RAW features for H calculation from: {raw_feat_dir}")
        all_raw_feats = []
        for dom in DOMAINS:
            r_path = os.path.join(raw_feat_dir, f"{dom}_test.pt")
            if not os.path.exists(r_path):
                raise FileNotFoundError(f"Raw feature file missing: {r_path}")
            r_f, _ = torch.load(r_path, weights_only=True)
            all_raw_feats.append(r_f)
        Global_Feats_For_H = torch.cat(all_raw_feats, dim=0).to(device)
    else:
        Global_Feats_For_H = Global_Feats

    N_total = Global_Y.shape[0]
    K = len(DOMAINS)
    Global_H = np.zeros((N_total, NUM_CLASSES, K))

    print("   -> Computing H matrix...")
    with torch.no_grad():
        for k, src_domain in enumerate(DOMAINS):
            if src_domain not in classifiers: continue
            batch_size = 512
            for b in range(0, N_total, batch_size):
                batch_feats = Global_Feats_For_H[b:b + batch_size]
                logits = classifiers[src_domain](batch_feats)
                probs = F.softmax(logits, dim=1).cpu().numpy()
                Global_H[b:b + batch_size, :, k] = probs

    return Global_D, Global_Y, Global_H, domain_ranges


def slice_matrices_with_ratios(target_domains, Global_D, Global_Y, Global_H, domain_ranges):
    row_indices = []
    avail_sizes = {d: domain_ranges[d][1] - domain_ranges[d][0] for d in target_domains}
    counts = {}
    key_tuple = tuple(target_domains)

    if len(target_domains) == 2 and key_tuple in TARGET_RATIOS:
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
    WD = D * w.reshape(1, -1)             # (N,K)
    Z = WD.sum(axis=1, keepdims=True)     # (N,1)
    Z = np.maximum(Z, eps)
    return WD / Z                         # rows sum to 1

def evaluate_accuracy(w, D, H, Y):
    Q = compute_Q(D, w)                              # (N,K)
    final_preds = (H * Q[:, None, :]).sum(axis=2)    # (N,C)
    return accuracy_score(Y.argmax(axis=1), final_preds.argmax(axis=1)) * 100.0


# ==========================================
# 2. DEBUG & WORKER (SERIAL)
# ==========================================

def comprehensive_debug_analysis(buf, Y, D, H, target_domains, source_errors):
    """
    Checks Matrix health and Feasibility Gaps before running solvers.
    """
    print("\n   >>> COMPREHENSIVE DEBUG ANALYSIS <<<")

    # 1. Check for NaNs/Infs
    if np.any(np.isnan(D)) or np.any(np.isinf(D)):
        print("   [CRITICAL FAIL] D matrix contains NaN or Inf!")
        buf.write("   [CRITICAL FAIL] D matrix contains NaN or Inf!\n")

    # 2. Check D Stats
    d_max = D.max()
    print(f"   [D STATS] Max: {d_max:.4f} | Mean: {D.mean():.4f} | Min: {D.min():.4f}")
    if d_max > 1000:
        print("   [WARNING] D values are very large (>1000). Numerical instability likely.")
        buf.write("   [WARNING] D values > 1000.\n")

    # 3. Feasibility Gap Calculation
    K = len(target_domains)
    w_unif = np.ones(K) / K

    # Prediction P(y|x) = sum_k ( w_k * D_k(x) * H_k(x) )
    Q_unif = compute_Q(D, w_unif)
    pred_target = (H * Q_unif[:, None, :]).sum(axis=2)

    # Calculate Soft Error
    correct_probs = (pred_target * Y).sum(axis=1)
    empirical_target_loss = 1.0 - np.mean(correct_probs)

    # Calculate Source Risk Constraint (RHS)
    current_source_errors = [source_errors[d] for d in target_domains]
    weighted_source_err = np.dot(w_unif, current_source_errors)

    gap = empirical_target_loss - weighted_source_err

    print(f"   [FEASIBILITY CHECK - UNIFORM WEIGHTS]")
    print(f"   > Empirical Target Loss: {empirical_target_loss:.4f}")
    print(f"   > Weighted Source Error: {weighted_source_err:.4f}")
    print(f"   > GAP (Loss - SrcErr):   {gap:.4f}")
    print("   --------------------------------------\n")
    return gap


def run_solver_loop(Y, D, H, target_list):
    """
    Runs solvers serially (no parallel) so we can see print outputs.
    """
    buf = io.StringIO()

    # --- Perform Debug Analysis First ---
    gap_approx = comprehensive_debug_analysis(buf, Y, D, H, target_list, SOURCE_ERRORS)

    # --- Define Solvers ---
    solvers_to_run = []
    if 'solve_convex_problem_mosek' in globals(): solvers_to_run.append("CVXPY_GLOBAL")
    if 'solve_convex_problem_per_domain' in globals(): solvers_to_run.append("CVXPY_PER_DOMAIN")

    if not solvers_to_run:
        return "No CVXPY solvers found.\n"

    # --- Sweep Parameters ---
    # Relaxed multipliers even further
    eps_multipliers = [2, 5, 10]

    D_expanded = np.tile(D[:, None, :], (1, NUM_CLASSES, 1))

    for eps_mult in eps_multipliers:
        errors = np.array([(SOURCE_ERRORS.get(d, 0.1) + 0.05) * eps_mult for d in target_list])
        max_err_input = np.max(errors)

        K = len(target_list)
        max_ent = np.log(K) if K > 1 else 0.1

        for solver in solvers_to_run:
            # Only test one delta per epsilon to reduce clutter
            mult = 2.0
            delta = mult * max_ent

            print(f"   [SOLVER] {solver} | EpsMult: {eps_mult} | Delta: {delta:.2f}...")

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
                    msg = f"{solver:<18} | m:{eps_mult:<5} | d:{mult:<5} | Acc: {acc:.2f}% | w: [{w_str}]"
                    print(f"   >>> SUCCESS: {msg}")
                    buf.write(msg + "\n")
                else:
                    print(f"   !!! INFEASIBLE")
                    buf.write(
                        f"{solver:<18} | m:{eps_mult:<5} | d:{mult:<5} | INFEASIBLE (Gap was ~{gap_approx:.3f})\n")

            except Exception as e:
                err_msg = str(e).replace('\n', ' ')[:100]
                print(f"   !!! ERROR: {err_msg}")
                buf.write(f"{solver:<18} | m:{eps_mult:<5} | d:{mult:<5} | ERR: {err_msg}\n")
                pass

    return buf.getvalue()


def run_baselines_and_dc(Y, D, H, target_domains, oracle_w):
    buf = io.StringIO()
    K = len(target_domains)

    acc_oracle = evaluate_accuracy(oracle_w, D, H, Y)
    msg = f"{'ORACLE (True)':<18} | {'-':<7} | {'-':<7} | Acc: {acc_oracle:.2f}% | w: {np.round(oracle_w, 3)}"
    print(f"   {msg}")
    buf.write(msg + "\n")

    w_unif = np.ones(K) / K
    acc_unif = evaluate_accuracy(w_unif, D, H, Y)
    msg = f"{'UNIFORM':<18} | {'-':<7} | {'-':<7} | Acc: {acc_unif:.2f}% | w: {np.round(w_unif, 3)}"
    print(f"   {msg}")
    buf.write(msg + "\n")

    return buf.getvalue()


# ==========================================
# 3. MAIN
# ==========================================
def main():
    Global_D, Global_Y, Global_H, domain_ranges = load_global_matrices()
    results_path = os.path.join(
        RESULTS_DIR, "Results_MSA_GAP_UMAP_CustomRatios_SERIAL.txt"
    )

    combinations = []
    combinations.extend(list(itertools.combinations(DOMAINS, 2)))
    # combinations.extend(list(itertools.combinations(DOMAINS, 3)))  # optional

    print(f"\n[MSA] Starting optimization loop on {len(combinations)} combinations...")

    with open(results_path, "w") as fp:
        # =============================
        # Global header
        # =============================
        fp.write("MSA EXPERIMENT REPORT | Custom Ratios | SERIAL DEBUG\n")
        fp.write(f"D Matrix: {D_MATRIX_NAME}\n")
        fp.write("=" * 100 + "\n\n")

        for target_tuple in combinations:
            target_list = list(target_tuple)
            print(f"\n>>> Processing Target Mix: {target_list}")

            # ---------------------------------
            # Slice matrices
            # ---------------------------------
            Y_sub, D_sub, H_sub, oracle_w = slice_matrices_with_ratios(
                target_list, Global_D, Global_Y, Global_H, domain_ranges
            )

            header = f"TARGET MIXTURE: {target_list} (Samples: {Y_sub.shape[0]})"
            print(header)
            fp.write(header + "\n")
            fp.write("-" * len(header) + "\n")

            # ==========================================================
            # DIAGNOSTIC 1: Classifier argmax agreement
            # ==========================================================
            preds = [H_sub[:, :, k].argmax(axis=1) for k in range(H_sub.shape[2])]
            agree_all = np.ones(len(preds[0]), dtype=bool)
            for k in range(1, len(preds)):
                agree_all &= (preds[k] == preds[0])
            agree_rate = np.mean(agree_all)

            diag1_msg = f"[DIAG-1] Classifier argmax agreement rate: {agree_rate:.4f}"
            print(diag1_msg)
            fp.write(diag1_msg + "\n")

            if agree_rate > 0.95:
                msg = (
                    "         → Classifiers almost always predict the SAME label.\n"
                    "         → Mixture weights w CANNOT change argmax accuracy.\n"
                )
            elif agree_rate > 0.8:
                msg = (
                    "         → High agreement between classifiers.\n"
                    "         → Accuracy is only weakly sensitive to w.\n"
                )
            else:
                msg = (
                    "         → Significant disagreement between classifiers.\n"
                    "         → Accuracy SHOULD depend on w.\n"
                )

            print(msg.rstrip())
            fp.write(msg)

            # ==========================================================
            # DIAGNOSTIC 2: Sensitivity of q(w) to w
            # ==========================================================
            w_uniform = np.ones(len(target_list)) / len(target_list)
            w_skewed = np.zeros(len(target_list))
            w_skewed[0] = 0.9
            w_skewed[1] = 0.1

            Q1 = compute_Q(D_sub, w_uniform)
            Q2 = compute_Q(D_sub, w_skewed)

            q_diff = np.mean(np.abs(Q1 - Q2))

            diag2_msg = f"[DIAG-2] Mean |Q(uniform) - Q(skewed)| = {q_diff:.6e}"
            print(diag2_msg)
            fp.write(diag2_msg + "\n")

            if q_diff < 1e-4:
                msg = (
                    "         → q(w) is almost invariant to w.\n"
                    "         → D is very sharp; each sample is dominated by one source.\n"
                )
            elif q_diff < 1e-2:
                msg = (
                    "         → q(w) changes mildly with w.\n"
                    "         → Only soft metrics may change.\n"
                )
            else:
                msg = (
                    "         → q(w) is sensitive to w.\n"
                    "         → Accuracy SHOULD change if classifiers disagree.\n"
                )

            print(msg.rstrip())
            fp.write(msg)

            # ==========================================================
            # DIAGNOSTIC 3: Soft target score (pre-argmax)
            # ==========================================================
            def soft_target_score(w, D, H, Y):
                Q = compute_Q(D, w)
                P = (H * Q[:, None, :]).sum(axis=2)
                return np.mean((P * Y).sum(axis=1))

            soft_oracle = soft_target_score(oracle_w, D_sub, H_sub, Y_sub)
            soft_uniform = soft_target_score(w_uniform, D_sub, H_sub, Y_sub)

            diag3_msg1 = f"[DIAG-3] Soft score (oracle w):  {soft_oracle:.6f}"
            diag3_msg2 = f"[DIAG-3] Soft score (uniform w): {soft_uniform:.6f}"

            print(diag3_msg1)
            print(diag3_msg2)
            fp.write(diag3_msg1 + "\n")
            fp.write(diag3_msg2 + "\n")

            if abs(soft_oracle - soft_uniform) < 1e-4:
                msg = (
                    "         → Even soft predictions are invariant to w.\n"
                    "         → The problem is fully degenerate on this dataset.\n"
                )
            else:
                msg = (
                    "         → Soft predictions differ, but argmax hides it.\n"
                    "         → Accuracy is not a sensitive metric here.\n"
                )

            print(msg.rstrip())
            fp.write(msg)

            fp.write("\n")

            # ==========================================================
            # BASELINES
            # ==========================================================
            base_res = run_baselines_and_dc(
                Y_sub, D_sub, H_sub, target_list, oracle_w
            )
            fp.write(base_res)

            # ==========================================================
            # CVXPY SOLVERS
            # ==========================================================
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