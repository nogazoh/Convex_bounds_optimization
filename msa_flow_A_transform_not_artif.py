from __future__ import print_function
import torch
import numpy as np
import os
import io
import itertools
import copy
import torch.nn.functional as F
from torch import nn, optim
from torchvision import models, transforms, datasets
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Subset, TensorDataset
import cvxpy as cp
# from helpers import *

# ============================================================
# IMPORTS FOR FLOW MATCHING & ODE
# ============================================================
try:
    from torchcfm.conditional_flow_matching import TargetConditionalFlowMatcher
    from torchdiffeq import odeint
except ImportError:
    print("❌ Missing libraries. Please run: pip install torchcfm torchdiffeq")
    exit()
# ============================================================
# IMPORTS FOR DC SOLVER
# ============================================================
try:
    from dc import *
except ImportError:
    print("[WARNING] 'dc.py' not found. DC Solver baseline will be skipped.")
    pass

# ==========================================
# --- CONFIGURATION ---
# ==========================================
ROOT_DIR = "/data/nogaz/Convex_bounds_optimization/LatentFlow_Pixel_Experiments"
OFFICE_DIR = "/data/nogaz/Convex_bounds_optimization/Office-31"
# Ensure this points to the FULL DATA matrix created by the data generation script
D_MATRIX_NAME = "D_Matrix_FINAL_GMM_Soft.npy"# "D_Matrix_LatentFlow_T8_FULL_DATA.npy"
D_MATRIX_PATH = os.path.join(ROOT_DIR, "results", D_MATRIX_NAME)
RESULTS_DIR = os.path.join(ROOT_DIR, "Results_MSA_EM_FlowFeedback")
os.makedirs(RESULTS_DIR, exist_ok=True)

DOMAINS = ['amazon', 'dslr', 'webcam']
NUM_CLASSES = 31
SOURCE_ERRORS = {'amazon': 0.1352, 'dslr': 0.0178, 'webcam': 0.0225}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Hardcoded Loss Matrix ---
FULL_L_MAT = np.array([
    [0.1379, 0.2349, 0.3183],  # Source: Amazon
    [0.3546, 0.0178, 0.0484],  # Source: DSLR
    [0.3832, 0.0427, 0.0242]  # Source: Webcam
])

# Global variable to track which domains are currently active
CURRENT_ACTIVE_DOMAINS = []

print("=" * 80)
print("Running MSA: Train on Train-Set (Unsupervised) -> Eval on Test-Set")
print("Including Baselines: Oracle, Uniform, DC Solver")
print("=" * 80)


# ============================================================
# SHARED HELPERS
# ============================================================

def calculate_expected_loss(Y, H, D, num_sources, eps=1e-12):
    global CURRENT_ACTIVE_DOMAINS, FULL_L_MAT, DOMAINS
    if not CURRENT_ACTIVE_DOMAINS:
        indices = range(len(DOMAINS))
    else:
        indices = [DOMAINS.index(d) for d in CURRENT_ACTIVE_DOMAINS]
    L_mat = FULL_L_MAT[np.ix_(indices, indices)]
    if L_mat.shape != (num_sources, num_sources):
        L_mat = np.zeros((num_sources, num_sources))
    return L_mat


def compute_p_tilde(D, eps=1e-15):
    """ Domain-anchored normalized expert density """
    row_sum = np.sum(D, axis=1, keepdims=True)
    return D / np.maximum(row_sum, eps)


def compute_Q_analytical(D, w, eps=1e-12):
    """ Analytical Q calculation (Bayes Rule): Q_ik = (w_k * D_ik) / sum_j(w_j * D_ij) """
    numerator = D * w.reshape(1, -1)
    denominator = numerator.sum(axis=1, keepdims=True)
    Q = numerator / np.maximum(denominator, eps)
    return Q


def evaluate_accuracy(w, D, H, Y):
    Q = compute_Q_analytical(D, w)
    final_preds = (H * Q[:, None, :]).sum(axis=2)
    return accuracy_score(Y.argmax(axis=1), final_preds.argmax(axis=1)) * 100.0


def check_correlation(D, H, Y, domains):
    print("\n" + "=" * 60)
    print(">>> DIAGNOSTIC CHECK: Does High Density (D) Predict Accuracy? (On Test)")
    if Y.ndim > 1:
        y_true = np.argmax(Y, axis=1)
    else:
        y_true = Y

    for k, name in enumerate(domains):
        print(f"\n--- Domain: {name} ---")
        preds = np.argmax(H[:, :, k], axis=1)
        is_correct = (preds == y_true)
        density_scores = D[:, k]
        if len(density_scores) == 0: continue
        avg_d_correct = np.mean(density_scores[is_correct]) if np.any(is_correct) else 0
        avg_d_wrong = np.mean(density_scores[~is_correct]) if np.any(~is_correct) else 0
        print(f"  Avg D when CORRECT: {avg_d_correct:.4e}")
        print(f"  Avg D when WRONG:   {avg_d_wrong:.4e}")
    print("=" * 60 + "\n")

def load_density_models():
    """ Load Global Scaler, PCA, and GMMs """
    global DENSITY_MODELS
    try:
        print("[INIT] Loading Density Models for Dynamic Update...")
        DENSITY_MODELS['scaler'] = joblib.load(os.path.join(DENSITY_MODELS_DIR, "global_scaler.pkl"))
        DENSITY_MODELS['pca'] = joblib.load(os.path.join(DENSITY_MODELS_DIR, "global_pca.pkl"))
        DENSITY_MODELS['gmms'] = {}
        for d in DOMAINS:
            DENSITY_MODELS['gmms'][d] = joblib.load(os.path.join(DENSITY_MODELS_DIR, f"gmm_{d}.pkl"))
        print("✅ Density models loaded successfully.")
        return True
    except Exception as e:
        print(f"❌ Failed to load density models: {e}")
        return False


def update_H_and_D(loader, feature_extractor, classifiers, adapters, source_domains, temperature=5.0):
    scaler = DENSITY_MODELS['scaler']
    pca = DENSITY_MODELS['pca']
    gmms = DENSITY_MODELS['gmms']

    feature_extractor.eval()
    for net in adapters.values(): net.eval()
    for net in classifiers.values(): net.eval()

    all_H_list = []
    raw_log_probs_list = []  # רשימה לשמירת הציונים של D

    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device)
            base_feats = feature_extractor(imgs)  # 2048

            batch_H = []
            batch_log_probs = []  # ציונים לבאץ' הנוכחי

            for k_idx, dom in enumerate(source_domains):
                # 1. הפעלת האדפטר: x' = Ax
                adapter = adapters[dom]
                feats_adapted = adapter(base_feats)

                # 2. חישוב H: Classifier(x')
                head = classifiers[dom].fc
                logits = head(feats_adapted)
                probs = F.softmax(logits, dim=1)
                batch_H.append(probs.cpu().numpy())

                # 3. חישוב D: GMM(PCA(Scaler(x'))) - החלק שהיה חסר
                feats_np = feats_adapted.cpu().numpy()
                # סקיילינג ו-PCA כמו באימון המקורי
                feats_scaled = scaler.transform(feats_np)
                z_pca = pca.transform(feats_scaled)

                gmm = gmms[dom]
                log_prob = gmm.score_samples(z_pca)
                batch_log_probs.append(log_prob)

            # איחוד תוצאות הבאץ'
            batch_H_stack = np.stack(batch_H, axis=2)
            all_H_list.append(batch_H_stack)

            batch_log_p_stack = np.stack(batch_log_probs, axis=1)
            raw_log_probs_list.append(batch_log_p_stack)


    H_new = np.concatenate(all_H_list, axis=0)

    Raw_LogP = np.concatenate(raw_log_probs_list, axis=0)

    scaled_log_p = Raw_LogP / temperature
    cols_max = np.max(scaled_log_p, axis=0, keepdims=True)
    log_p_shifted = scaled_log_p - cols_max
    p_unnormalized = np.exp(log_p_shifted)

    col_sums = np.sum(p_unnormalized, axis=0, keepdims=True)
    D_new = p_unnormalized / (col_sums + 1e-12)
    D_new = np.clip(D_new, 1e-15, 1.0)

    return H_new, D_new



# ============================================================
# SOLVERS (Standardized to return only w)
# ============================================================

def solve_convex_problem_mosek(Y, D, H, delta=1e-2, epsilon=1e-2, L_mat=None, solver_type='SCS'):
    if D.ndim == 3: D = D.reshape(-1, D.shape[2])
    N, k = D.shape
    # Normalize D locally for this solver
    col_sums = D.sum(axis=0, keepdims=True)
    col_sums[col_sums == 0] = 1.0
    D_norm = D / col_sums

    w = cp.Variable(k, nonneg=True, name='w')
    Q = cp.Variable((N, k), nonneg=True, name='Q')
    R = cp.Variable((k, k), nonneg=True, name='R')

    obj_terms = []
    for j in range(k):
        obj_terms.append(cp.sum(cp.multiply(D_norm[:, j], -cp.rel_entr(w[j], Q[:, j]))))
    objective = cp.Maximize(cp.sum(obj_terms))

    constraints = [cp.sum(w) == 1, cp.sum(Q, axis=1) == 1, cp.sum(R, axis=0) == 1]

    kl_terms = []
    for t in range(k):
        p_t = D_norm[:, t]
        inner = []
        for j in range(k):
            re = cp.rel_entr(R[j, t], Q[:, j])
            inner.append(cp.sum(cp.multiply(p_t, re)))
        kl_terms.append(cp.sum(inner))
    constraints.append((1.0 / k) * cp.sum(kl_terms) <= delta)

    if L_mat is None: L_mat = calculate_expected_loss(Y, H, D, k)
    if isinstance(epsilon, (list, np.ndarray)):
        for t in range(k): constraints.append(cp.sum(cp.multiply(R[:, t], L_mat[:, t])) <= epsilon[t])
    else:
        for t in range(k): constraints.append(cp.sum(cp.multiply(R[:, t], L_mat[:, t])) <= epsilon)

    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.SCS, verbose=False, eps=1e-4, max_iters=10000)
    except:
        return None

    if prob.status not in ["optimal", "optimal_inaccurate"]: return None
    return np.asarray(w.value).reshape(-1)


def solve_convex_problem_smoothed_kl(Y, D, H, epsilon=1e-2, eta=1e-2, solver_type="SCS", q_min=1e-12, w_min=1e-12,
                                     scs_eps=1e-4, scs_max_iters=20000):
    N, k = D.shape
    L_mat = calculate_expected_loss(Y, H, D, k)
    w = cp.Variable(k, nonneg=True, name="w")
    Q = cp.Variable((N, k), nonneg=True, name="Q")

    main_terms = [cp.sum(cp.multiply(D[:, j], -cp.rel_entr(w[j], Q[:, j]))) for j in range(k)]
    main_obj = cp.sum(main_terms)
    smooth_obj = (eta / (k * N)) * cp.sum(cp.log(Q))
    objective = cp.Maximize(main_obj + smooth_obj)

    constraints = [cp.sum(w) == 1, w >= w_min, cp.sum(Q, axis=1) == 1, Q >= q_min]
    for t in range(k):
        eps_t = epsilon[t] if isinstance(epsilon, (list, np.ndarray)) else epsilon
        constraints.append(w @ L_mat[:, t] <= eps_t)

    prob = cp.Problem(objective, constraints)
    try:
        if solver_type == "MOSEK":
            prob.solve(solver=cp.MOSEK, verbose=False)
        else:
            prob.solve(solver=cp.SCS, verbose=False, eps=scs_eps, max_iters=scs_max_iters)
    except:
        return None
    if prob.status not in ["optimal", "optimal_inaccurate"]: return None
    return np.asarray(w.value).reshape(-1)


def solve_convex_problem_domain_anchored_smoothed(Y, D, H, epsilon=1e-2, eta=1e-2, solver_type="SCS", q_min=1e-12,
                                                  w_min=1e-12, scs_eps=1e-4, scs_max_iters=20000,
                                                  normalize_domains=True, ptilde_eps=1e-15):
    N, k = D.shape
    p_tilde = compute_p_tilde(D, eps=ptilde_eps) if normalize_domains else D
    L_mat = calculate_expected_loss(Y, H, D, k)
    w = cp.Variable(k, nonneg=True, name="w")
    Q = cp.Variable((N, k), nonneg=True, name="Q")

    main_terms = [cp.sum(cp.multiply(D[:, j], -cp.rel_entr(w[j], Q[:, j]))) for j in range(k)]
    main_obj = cp.sum(main_terms)
    anchored_smooth_obj = (eta / k) * cp.sum(cp.multiply(p_tilde, cp.log(Q)))
    objective = cp.Maximize(main_obj + anchored_smooth_obj)

    constraints = [cp.sum(w) == 1, w >= w_min, cp.sum(Q, axis=1) == 1, Q >= q_min]
    for t in range(k):
        eps_t = epsilon[t] if isinstance(epsilon, (list, np.ndarray)) else epsilon
        constraints.append(w @ L_mat[:, t] <= eps_t)

    prob = cp.Problem(objective, constraints)
    try:
        if solver_type == "MOSEK":
            prob.solve(solver=cp.MOSEK, verbose=False)
        else:
            prob.solve(solver=cp.SCS, verbose=False, eps=scs_eps, max_iters=scs_max_iters)
    except:
        return None
    if prob.status not in ["optimal", "optimal_inaccurate"]: return None
    return np.asarray(w.value).reshape(-1)


def solve_convex_problem_smoothed_original_p(Y, D, H, epsilon=1e-2, eta=1e-2, solver_type="SCS", q_min=1e-12,
                                             w_min=1e-12, scs_eps=1e-4, scs_max_iters=20000):
    N, k = D.shape
    L_mat = calculate_expected_loss(Y, H, D, k)
    w = cp.Variable(k, nonneg=True, name="w")
    Q = cp.Variable((N, k), nonneg=True, name="Q")

    main_terms = [cp.sum(cp.multiply(D[:, j], -cp.rel_entr(w[j], Q[:, j]))) for j in range(k)]
    main_obj = cp.sum(main_terms)
    smooth_obj = eta * cp.sum(cp.multiply(D, cp.log(Q)))
    objective = cp.Maximize(main_obj + smooth_obj)

    constraints = [cp.sum(w) == 1, w >= w_min, cp.sum(Q, axis=1) == 1, Q >= q_min]
    for t in range(k):
        eps_t = epsilon[t] if isinstance(epsilon, (list, np.ndarray)) else epsilon
        constraints.append(w @ L_mat[:, t] <= eps_t)

    prob = cp.Problem(objective, constraints)
    try:
        if solver_type == "MOSEK":
            prob.solve(solver=cp.MOSEK, verbose=False)
        else:
            prob.solve(solver=cp.SCS, verbose=False, eps=scs_eps, max_iters=scs_max_iters)
    except:
        return None
    if prob.status not in ["optimal", "optimal_inaccurate"]: return None
    return np.asarray(w.value).reshape(-1)


# ==========================================
# HELPERS & LOADING
# ==========================================

def get_train_test_loaders_and_indices(domain, batch_size=32):
    path = os.path.join(OFFICE_DIR, domain, 'images')
    if not os.path.exists(path): path = os.path.join(OFFICE_DIR, domain)

    tr_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tr_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    try:
        ds_full = datasets.ImageFolder(path)
    except FileNotFoundError:
        return None, None, None, None

    N = len(ds_full)
    rng = np.random.RandomState(42)
    indices = rng.permutation(N)
    split_point = int(0.8 * N)
    train_idx = indices[:split_point]
    test_idx = indices[split_point:]

    class TransformedSubset(torch.utils.data.Dataset):
        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform

        def __getitem__(self, index):
            x, y = self.subset[index]
            if self.transform: x = self.transform(x)
            return x, y

        def __len__(self): return len(self.subset)

    train_ds = TransformedSubset(Subset(ds_full, train_idx), tr_train)
    test_ds = TransformedSubset(Subset(ds_full, test_idx), tr_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, train_idx, test_idx


def load_global_D_matrix(path: str):
    try:
        Dg = np.load(path)
        print(f"✅ Loaded Global D from: {path}")
        print(f"   Global D shape: {Dg.shape}")
        return Dg
    except Exception as e:
        print(f"❌ Failed loading Global D: {e}")
        return None


def infer_H_matrix(loader, classifiers, source_domains):
    """ Helper to run inference and return stacked H and Y """
    H_list = {d: [] for d in source_domains}
    Y_list = []

    with torch.no_grad():
        for d in source_domains: classifiers[d].eval()
        for imgs, labels in loader:
            imgs = imgs.to(device)
            Y_list.append(labels.numpy())
            for d in source_domains:
                logits = classifiers[d](imgs)
                probs = F.softmax(logits, dim=1).cpu().numpy()
                H_list[d].append(probs)

    Y_flat = np.concatenate(Y_list)
    N = Y_flat.shape[0]

    # One-hot Y
    Y_onehot = np.zeros((N, NUM_CLASSES))
    Y_onehot[np.arange(N), Y_flat] = 1

    H_stacked = np.zeros((N, NUM_CLASSES, len(source_domains)))
    for idx, d in enumerate(source_domains):
        H_stacked[:, :, idx] = np.concatenate(H_list[d], axis=0)

    return H_stacked, Y_onehot


# ==========================================
# --- BASELINES ---
# ==========================================

def run_baselines_train_test(target_domain, source_domains,
                             train_loader_ordered, test_loader,
                             D_train, D_test):
    """
    Run baselines respecting the split:
    - Uniform/Oracle: Calculated/Evaluated on Test.
    - DC Solver: Learned on Train (unsupervised), Evaluated on Test.
    """
    buf = io.StringIO()

    # 1. Load Static Classifiers
    classifiers = {}
    for d in DOMAINS:
        m = models.resnet50(weights=None)
        m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
        path = f"/data/nogaz/Convex_bounds_optimization/classifiers_new/{d}_classifier.pt"
        if not os.path.exists(path): path = f"/data/nogaz/Convex_bounds_optimization/classifiers/{d}_224.pt"
        if os.path.exists(path):
            state = torch.load(path, map_location=device, weights_only=True)
            m.load_state_dict(state)
            classifiers[d] = m.to(device).eval()

    # 2. Compute H for Train and Test
    H_train, Y_train_dummy = infer_H_matrix(train_loader_ordered, classifiers, source_domains)
    H_test, Y_test_real = infer_H_matrix(test_loader, classifiers, source_domains)

    # --- Check Correlation on Test ---
    check_correlation(D_test, H_test, Y_test_real, source_domains)

    # --- Uniform & Oracle (on Test) ---
    K = len(source_domains)
    w_unif = np.ones(K) / K
    acc_unif = evaluate_accuracy(w_unif, D_test, H_test, Y_test_real)

    # Oracle w calculation (based on Test counts)
    # We don't have domain info inside H/D, so we approximate 'Oracle'
    # as the optimal weights if we knew the target composition relative to sources.
    # Here we assume Oracle = Uniform for simplicity OR true source ratios if known.
    # Let's stick to Uniform as baseline + Max Possible.

    # Max Possible Accuracy (Oracle Upper Bound)
    y_true = np.argmax(Y_test_real, axis=1)
    source_preds = np.argmax(H_test, axis=1)
    is_correct_matrix = (source_preds == y_true[:, None])
    solvable = np.any(is_correct_matrix, axis=1)
    max_acc = np.mean(solvable) * 100.0

    buf.write(f"BASELINE UNIFORM: {acc_unif:.2f}%\n")
    buf.write(f"BASELINE MAX_ORACLE: {max_acc:.2f}%\n")
    print(f"BASELINE UNIFORM: {acc_unif:.2f}%")
    print(f"BASELINE MAX_ORACLE: {max_acc:.2f}%")

    # --- DC Solver (Train on Train -> Eval on Test) ---
    dc_accuracies = []
    best_w = None
    best_acc = -1
    D_train_expanded = np.tile(D_train[:, None, :], (1, NUM_CLASSES, 1))

    print(" >>> Running DC Solver (5 seeds) [Train->Test]...")
    for i in range(1):
        try:
            # Init problem with TRAIN data
            dp = init_problem_from_model(Y_train_dummy, D_train_expanded, H_train, p=K, C=NUM_CLASSES)
            problem = ConvexConcaveProblem(dp)
            slv = ConvexConcaveSolver(problem, 42 + (i * 100), "err")
            w_dc, _, _ = slv.solve(delta=1e-4)

            if w_dc is not None:
                # Evaluate on TEST data
                acc = evaluate_accuracy(w_dc, D_test, H_test, Y_test_real)
                dc_accuracies.append(acc)
                if acc > best_acc: best_acc, best_w = acc, w_dc
        except Exception as e:
            # print(e)
            continue

    if dc_accuracies:
        avg = f"{np.mean(dc_accuracies):.2f} +/- {np.std(dc_accuracies):.2f}"
        buf.write(f"BASELINE DC (5-Seeds): {avg} | Best: {best_acc:.2f} | w: {np.round(best_w, 3)}\n")
        print(f"BASELINE DC: {avg} | Best: {best_acc:.2f}")
    else:
        buf.write("BASELINE DC: FAILED\n")
        print("BASELINE DC: FAILED")

    return buf.getvalue()


# ==========================================
# --- EM LOGIC ---
# ==========================================

def update_classifiers_pytorch(classifiers, train_loader, w_opt, Q_opt, source_domains,
                               lr=1e-3,
                               # --- FLAGS ---
                               hard_assignment=False,
                               use_entropy=True,  # האם להפעיל מינימיזציית אנטרופיה
                               use_diversity=True,  # האם להפעיל מקסימיזציית גיוון
                               train_classifier=False,  # האם לאמן את הקלאסיפייר המקורי
                               train_adapter=True,  # האם לאמן את שכבת A
                               adapters=None):  # המילון שמחזיק את ה-A לכל מקור

    optimizers = {}
    original_heads = {}

    for d in source_domains:
        classifiers[d].eval()
        original_heads[d] = classifiers[d].fc
        classifiers[d].fc = nn.Identity()
        params_to_update = []
        if train_classifier:
            original_heads[d].train()
            params_to_update += list(original_heads[d].parameters())
        else:
            original_heads[d].eval()
        if train_adapter and adapters is not None:
            adapters[d].train()
            params_to_update += list(adapters[d].parameters())
        if params_to_update:
            optimizers[d] = optim.SGD(params_to_update, lr=lr, momentum=0.9)
        else:
            optimizers[d] = None

    Q_tensor = torch.tensor(Q_opt, dtype=torch.float32, device=device)
    w_tensor = torch.tensor(w_opt, dtype=torch.float32, device=device)
    batch_start = 0

    for imgs, _ in train_loader:
        imgs = imgs.to(device)
        if batch_start >= Q_tensor.shape[0]: break
        Q_batch = Q_tensor[batch_start:batch_start + imgs.size(0)]
        if Q_batch.shape[0] != imgs.shape[0]: Q_batch = Q_batch[:imgs.shape[0]]

        if hard_assignment: winning_indices = Q_batch.argmax(dim=1)

        for opt in optimizers.values():
            if opt is not None: opt.zero_grad()

        loss_total = torch.tensor(0.0).to(device)

        for k_idx, domain in enumerate(source_domains):
            if optimizers[domain] is None: continue

            # === שיפור זיכרון קריטי ===
            # אם אנחנו לא מאמנים את הקלאסיפייר (אלא רק את האדפטר),
            # אין סיבה לשמור את הגרדיאנטים של ה-ResNet. זה חוסך המון זיכרון.
            if not train_classifier:
                with torch.no_grad():
                    features = classifiers[domain](imgs)
            else:
                features = classifiers[domain](imgs)
            # ==========================

            if train_adapter and adapters is not None:
                features = adapters[domain](features)

            logits = original_heads[domain](features)

            log_probs = F.log_softmax(logits, dim=1)
            probs = F.softmax(logits, dim=1)
            entropy = -(probs * log_probs).sum(dim=1) if use_entropy else torch.tensor(0.0).to(device)

            q_values = Q_batch[:, k_idx]
            threshold = 0.95

            if hard_assignment:
                is_winner = (winning_indices == k_idx)
                mask = ((q_values > threshold) & is_winner).float()
                final_weight = mask
            else:
                mask = (q_values > threshold).float()
                final_weight = q_values * w_tensor[k_idx]

            if mask.sum() > 0:
                if use_entropy: loss_total += (final_weight * entropy * mask).sum() / (mask.sum() + 1e-8)
                if use_diversity:
                    masked_probs = probs * mask.unsqueeze(1)
                    avg_probs = masked_probs.sum(dim=0) / (mask.sum() + 1e-8)
                    loss_total -= -(avg_probs * torch.log(avg_probs + 1e-8)).sum()

        if loss_total.item() != 0:
            loss_total.backward()
            for opt in optimizers.values():
                if opt is not None: opt.step()

        batch_start += imgs.size(0)

        # ניקוי זיכרון אקטיבי בסוף כל באץ'
        del imgs, features, logits, log_probs, probs, entropy, loss_total
        torch.cuda.empty_cache()

    for d in source_domains: classifiers[d].fc = original_heads[d]
    return


def run_single_em_experiment(target_domain, source_domains,
                             train_loader_ordered, test_loader,
                             D_train, D_test,
                             solver_func, solver_kwargs, epsilon_vec,
                             max_alt_iters=1,
                             # --- CONFIG FLAGS ---
                             flag_update_classifier=True,  # האם לעדכן את המודל המקורי?
                             flag_use_adapter=False,  # האם להשתמש ב-A לכל מקור?
                             flag_use_entropy=True,  # האם להשתמש בלוס אנטרופיה?
                             flag_use_diversity=True):  # האם להשתמש בלוס גיוון?

    classifiers = {}
    for d in DOMAINS:
        m = models.resnet50(weights=None)
        m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
        path = f"/data/nogaz/Convex_bounds_optimization/classifiers_new/{d}_classifier.pt"
        if not os.path.exists(path): path = f"/data/nogaz/Convex_bounds_optimization/classifiers/{d}_224.pt"
        if os.path.exists(path):
            state = torch.load(path, map_location=device, weights_only=True)
            m.load_state_dict(state)
            classifiers[d] = m.to(device)

    adapters = None
    if flag_use_adapter:
        adapters = {}
        for d in source_domains:
            adapter = nn.Linear(2048, 2048, bias=True).to(device)
            nn.init.eye_(adapter.weight)
            nn.init.zeros_(adapter.bias)
            adapters[d] = adapter

    N_train = D_train.shape[0]
    Y_train_dummy = np.zeros((N_train, NUM_CLASSES))

    y_test_list = []
    for _, lbls in test_loader: y_test_list.append(lbls.numpy())
    Y_test_labels = np.concatenate(y_test_list)
    Y_test_onehot = np.zeros((Y_test_labels.shape[0], NUM_CLASSES))
    Y_test_onehot[np.arange(Y_test_labels.shape[0]), Y_test_labels] = 1

    # --- STEP 1: SOLVER ---
    H_train, _ = infer_H_matrix(train_loader_ordered, classifiers, source_domains)

    current_args = {
        'Y': Y_train_dummy, 'D': D_train, 'H': H_train,
        'epsilon': epsilon_vec, 'solver_type': 'SCS'
    }
    current_args.update(solver_kwargs)

    try:
        w_opt = solver_func(**current_args)
    except Exception:
        w_opt = None

    if w_opt is None: return "Infeasible", "None"

    Q_opt = compute_Q_analytical(D_train, w_opt)

    # --- STEP 1.5: INITIAL EVAL ---
    H_test_initial, _ = infer_H_matrix(test_loader, classifiers, source_domains)
    initial_acc = evaluate_accuracy(w_opt, D_test, H_test_initial, Y_test_onehot)

    # --- STEP 2: UPDATE (Training based on flags) ---
    update_classifiers_pytorch(
        classifiers,
        train_loader_ordered,
        w_opt,
        Q_opt,
        source_domains,
        hard_assignment=True,
        use_entropy=flag_use_entropy,
        use_diversity=flag_use_diversity,
        train_classifier=flag_update_classifier,
        train_adapter=flag_use_adapter,
        adapters=adapters
    )

    # --- STEP 3: FINAL EVAL ---
    if flag_use_adapter and adapters is not None:
        H_list_final = {d: [] for d in source_domains}
        with torch.no_grad():
            for d in source_domains:
                adapters[d].eval()
                classifiers[d].eval()
                original_head = classifiers[d].fc
                classifiers[d].fc = nn.Identity()

                for imgs, _ in test_loader:
                    imgs = imgs.to(device)
                    feats = classifiers[d](imgs)
                    feats_trans = adapters[d](feats)
                    logits = original_head(feats_trans)
                    probs = F.softmax(logits, dim=1).cpu().numpy()
                    H_list_final[d].append(probs)

                classifiers[d].fc = original_head

        H_test_final = np.zeros((len(Y_test_labels), NUM_CLASSES, len(source_domains)))
        for idx, d in enumerate(source_domains):
            H_test_final[:, :, idx] = np.concatenate(H_list_final[d], axis=0)

    else:
        H_test_final, _ = infer_H_matrix(test_loader, classifiers, source_domains)

    final_acc = evaluate_accuracy(w_opt, D_test, H_test_final, Y_test_onehot)
    improvement = final_acc - initial_acc

    return f"{initial_acc:.2f} -> {final_acc:.2f} ({improvement:+.2f})", str(np.round(w_opt, 3))

# ==========================================
# MAIN EXECUTION
# ==========================================

def main():
    global CURRENT_ACTIVE_DOMAINS
    Global_D_Matrix = load_global_D_matrix(D_MATRIX_PATH)
    if Global_D_Matrix is None: return

    results_path = os.path.join(RESULTS_DIR, "MSA_Full_Grid_EM_TrainTest_Results.txt")

    with open(results_path, "w") as fp:
        fp.write("MSA REPORT | Train on Train (Unsup) -> Eval on Test | With Baselines\n")
        fp.write("=" * 80 + "\n\n")

        smoothed_solvers = [
            ("P3.21", solve_convex_problem_smoothed_kl),
            ("P3.22", solve_convex_problem_domain_anchored_smoothed),
            ("P3.23", solve_convex_problem_smoothed_original_p),
        ]

        eps_multipliers = [1.5, 2.0, 4, 7, 10, 20, 30, 50]
        eta_values = [5e-3, 0.1, 0.5]
        deltas = [0.1, 1, 1000]

        # Calculate offsets for D slicing
        domain_lengths = {}
        for d in DOMAINS:
            path = os.path.join(OFFICE_DIR, d, 'images')
            if not os.path.exists(path): path = os.path.join(OFFICE_DIR, d)
            ds = datasets.ImageFolder(path)
            domain_lengths[d] = len(ds)

        for target_domain in DOMAINS:
            source_domains = [d for d in DOMAINS if d != target_domain]
            CURRENT_ACTIVE_DOMAINS = source_domains

            header = f"\n>>> TARGET: {target_domain} | SOURCES: {source_domains}"
            print(header)
            fp.write(header + "\n")
            fp.write("-" * len(header) + "\n")

            # 1. Prepare Data
            tr_load, te_load, tr_idx, te_idx = get_train_test_loaders_and_indices(target_domain)

            # Ordered loader for consistent H/D mapping in solver/update
            train_ds_obj = tr_load.dataset
            train_loader_ordered = DataLoader(train_ds_obj, batch_size=64, shuffle=False)

            # Slice D
            start_pos = 0
            for d in DOMAINS:
                if d == target_domain: break
                start_pos += domain_lengths[d]

            end_pos = start_pos + domain_lengths[target_domain]
            D_target_full = Global_D_Matrix[start_pos:end_pos]
            D_target_full = D_target_full[:, [DOMAINS.index(d) for d in source_domains]]

            D_train = D_target_full[tr_idx]
            D_test = D_target_full[te_idx]
            print(f"   [Data] D_train: {D_train.shape}, D_test: {D_test.shape}")

            # 2. Run Baselines
            print("\n   --- BASELINES ---")
            fp.write("\n   --- BASELINES ---\n")
            fp.write(run_baselines_train_test(
                target_domain, source_domains,
                train_loader_ordered, te_load,
                D_train, D_test
            ))
            fp.write("-" * 40 + "\n")

            # 3. Grid Search
            for eps_mult in eps_multipliers:
                epsilon_vec = np.array([(SOURCE_ERRORS[d] + 0.05) * eps_mult for d in source_domains])
                eps_str = f"EpsMult:{eps_mult}"

                # P3.10
                for delta in deltas:
                    L_mat = calculate_expected_loss(None, None, None, len(source_domains))
                    solver_kwargs = {'delta': delta, 'L_mat': L_mat}
                    print(f" [EM-GRID] P3.10 | {eps_str} | Delta:{delta} ...")
                    prog_str, w_str = run_single_em_experiment(
                        target_domain, source_domains, train_loader_ordered, te_load,
                        D_train, D_test, solve_convex_problem_mosek, solver_kwargs, epsilon_vec
                    )
                    line = f"P3.10 | {eps_str:<12} | Delta:{delta:<6} | TestAcc: {prog_str} | w:{w_str}"
                    print(f"   >>> {line}")
                    fp.write(line + "\n")
                    fp.flush()

                # Smoothed
                for eta in eta_values:
                    for name, func in smoothed_solvers:
                        solver_kwargs = {'eta': eta}
                        print(f" [EM-GRID] {name} | {eps_str} | Eta:{eta} ...")
                        prog_str, w_str = run_single_em_experiment(
                            target_domain, source_domains, train_loader_ordered, te_load,
                            D_train, D_test, func, solver_kwargs, epsilon_vec
                        )
                        line = f"{name:<5} | {eps_str:<12} | Eta:{eta:<8} | TestAcc: {prog_str} | w:{w_str}"
                        print(f"   >>> {line}")
                        fp.write(line + "\n")
                        fp.flush()

            fp.write("\n" + "=" * 80 + "\n\n")

    print(f"\n[DONE] Results saved to: {results_path}")


if __name__ == "__main__":
    main()
######################################################################
# from __future__ import print_function
# import torch
# import numpy as np
# import os
# import io
# import itertools
# import copy
# import torch.nn.functional as F
# from torch import nn, optim
# from torchvision import models, transforms, datasets
# from sklearn.metrics import accuracy_score
# from torch.utils.data import DataLoader, Subset, TensorDataset
# import cvxpy as cp
# import joblib  # נדרש לטעינת המודלים של ה-GMM/PCA
#
# # ============================================================
# # IMPORTS FOR FLOW MATCHING & ODE
# # ============================================================
# try:
#     from torchcfm.conditional_flow_matching import TargetConditionalFlowMatcher
#     from torchdiffeq import odeint
# except ImportError:
#     print("❌ Missing libraries. Please run: pip install torchcfm torchdiffeq")
#     exit()
# # ============================================================
# # IMPORTS FOR DC SOLVER
# # ============================================================
# try:
#     from dc import *
# except ImportError:
#     print("[WARNING] 'dc.py' not found. DC Solver baseline will be skipped.")
#     pass
#
# # ==========================================
# # --- CONFIGURATION ---
# # ==========================================
# ROOT_DIR = "/data/nogaz/Convex_bounds_optimization/LatentFlow_Pixel_Experiments"
# OFFICE_DIR = "/data/nogaz/Convex_bounds_optimization/Office-31"
#
# D_MATRIX_NAME = "D_Matrix_FINAL_GMM_Soft.npy"
# D_MATRIX_PATH = os.path.join(ROOT_DIR, "results", D_MATRIX_NAME)
# RESULTS_DIR = os.path.join(ROOT_DIR, "Results_MSA_EM_FlowFeedback")
# os.makedirs(RESULTS_DIR, exist_ok=True)
#
# # נתיב למודלים שנשמרו (GMM, PCA, Scaler)
# DENSITY_MODELS_DIR = os.path.join(ROOT_DIR, "models", "gmm_models_soft")
#
# DOMAINS = ['amazon', 'dslr', 'webcam']
# NUM_CLASSES = 31
# SOURCE_ERRORS = {'amazon': 0.1352, 'dslr': 0.0178, 'webcam': 0.0225}
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# # --- Hardcoded Loss Matrix ---
# FULL_L_MAT = np.array([
#     [0.1379, 0.2349, 0.3183],  # Source: Amazon
#     [0.3546, 0.0178, 0.0484],  # Source: DSLR
#     [0.3832, 0.0427, 0.0242]  # Source: Webcam
# ])
#
# # Global variable to track which domains are currently active
# CURRENT_ACTIVE_DOMAINS = []
#
# # גלובלי - נטען בתחילת הריצה
# DENSITY_MODELS = {}
#
# print("=" * 80)
# print("Running MSA: Train on Train-Set (Unsupervised) -> Eval on Test-Set")
# print("Including Baselines: Oracle, Uniform, DC Solver")
# print("=" * 80)
#
#
# # ============================================================
# # SHARED HELPERS
# # ============================================================
#
# def calculate_expected_loss(Y, H, D, num_sources, eps=1e-12):
#     global CURRENT_ACTIVE_DOMAINS, FULL_L_MAT, DOMAINS
#     if not CURRENT_ACTIVE_DOMAINS:
#         indices = range(len(DOMAINS))
#     else:
#         indices = [DOMAINS.index(d) for d in CURRENT_ACTIVE_DOMAINS]
#     L_mat = FULL_L_MAT[np.ix_(indices, indices)]
#     if L_mat.shape != (num_sources, num_sources):
#         L_mat = np.zeros((num_sources, num_sources))
#     return L_mat
#
#
# def compute_p_tilde(D, eps=1e-15):
#     """ Domain-anchored normalized expert density """
#     row_sum = np.sum(D, axis=1, keepdims=True)
#     return D / np.maximum(row_sum, eps)
#
#
# def compute_Q_analytical(D, w, eps=1e-12):
#     """ Analytical Q calculation (Bayes Rule): Q_ik = (w_k * D_ik) / sum_j(w_j * D_ij) """
#     numerator = D * w.reshape(1, -1)
#     denominator = numerator.sum(axis=1, keepdims=True)
#     Q = numerator / np.maximum(denominator, eps)
#     return Q
#
#
# def evaluate_accuracy(w, D, H, Y):
#     Q = compute_Q_analytical(D, w)
#     final_preds = (H * Q[:, None, :]).sum(axis=2)
#     return accuracy_score(Y.argmax(axis=1), final_preds.argmax(axis=1)) * 100.0
#
#
# def check_correlation(D, H, Y, domains):
#     print("\n" + "=" * 60)
#     print(">>> DIAGNOSTIC CHECK: Does High Density (D) Predict Accuracy? (On Test)")
#     if Y.ndim > 1:
#         y_true = np.argmax(Y, axis=1)
#     else:
#         y_true = Y
#
#     for k, name in enumerate(domains):
#         print(f"\n--- Domain: {name} ---")
#         preds = np.argmax(H[:, :, k], axis=1)
#         is_correct = (preds == y_true)
#         density_scores = D[:, k]
#         if len(density_scores) == 0: continue
#         avg_d_correct = np.mean(density_scores[is_correct]) if np.any(is_correct) else 0
#         avg_d_wrong = np.mean(density_scores[~is_correct]) if np.any(~is_correct) else 0
#         print(f"  Avg D when CORRECT: {avg_d_correct:.4e}")
#         print(f"  Avg D when WRONG:   {avg_d_wrong:.4e}")
#     print("=" * 60 + "\n")
#
#
# # ============================================================
# # SOLVERS (Standardized to return only w)
# # ============================================================
#
# def solve_convex_problem_mosek(Y, D, H, delta=1e-2, epsilon=1e-2, L_mat=None, solver_type='SCS'):
#     # === SAFETY CLIP ===
#     D = np.clip(D, 1e-15, 1.0)
#     # ===================
#
#     if D.ndim == 3: D = D.reshape(-1, D.shape[2])
#     N, k = D.shape
#     # Normalize D locally for this solver
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
#     if L_mat is None: L_mat = calculate_expected_loss(Y, H, D, k)
#     if isinstance(epsilon, (list, np.ndarray)):
#         for t in range(k): constraints.append(cp.sum(cp.multiply(R[:, t], L_mat[:, t])) <= epsilon[t])
#     else:
#         for t in range(k): constraints.append(cp.sum(cp.multiply(R[:, t], L_mat[:, t])) <= epsilon)
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
#     # === SAFETY CLIP ===
#     D = np.clip(D, 1e-15, 1.0)
#     # ===================
#
#     N, k = D.shape
#     L_mat = calculate_expected_loss(Y, H, D, k)
#     w = cp.Variable(k, nonneg=True, name="w")
#     Q = cp.Variable((N, k), nonneg=True, name="Q")
#
#     main_terms = [cp.sum(cp.multiply(D[:, j], -cp.rel_entr(w[j], Q[:, j]))) for j in range(k)]
#     main_obj = cp.sum(main_terms)
#     smooth_obj = (eta / (k * N)) * cp.sum(cp.log(Q))
#     objective = cp.Maximize(main_obj + smooth_obj)
#
#     constraints = [cp.sum(w) == 1, w >= w_min, cp.sum(Q, axis=1) == 1, Q >= q_min]
#     for t in range(k):
#         eps_t = epsilon[t] if isinstance(epsilon, (list, np.ndarray)) else epsilon
#         constraints.append(w @ L_mat[:, t] <= eps_t)
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
#     # === SAFETY CLIP ===
#     D = np.clip(D, 1e-15, 1.0)
#     # ===================
#
#     N, k = D.shape
#     p_tilde = compute_p_tilde(D, eps=ptilde_eps) if normalize_domains else D
#     L_mat = calculate_expected_loss(Y, H, D, k)
#     w = cp.Variable(k, nonneg=True, name="w")
#     Q = cp.Variable((N, k), nonneg=True, name="Q")
#
#     main_terms = [cp.sum(cp.multiply(D[:, j], -cp.rel_entr(w[j], Q[:, j]))) for j in range(k)]
#     main_obj = cp.sum(main_terms)
#     anchored_smooth_obj = (eta / k) * cp.sum(cp.multiply(p_tilde, cp.log(Q)))
#     objective = cp.Maximize(main_obj + anchored_smooth_obj)
#
#     constraints = [cp.sum(w) == 1, w >= w_min, cp.sum(Q, axis=1) == 1, Q >= q_min]
#     for t in range(k):
#         eps_t = epsilon[t] if isinstance(epsilon, (list, np.ndarray)) else epsilon
#         constraints.append(w @ L_mat[:, t] <= eps_t)
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
#     # === SAFETY CLIP ===
#     D = np.clip(D, 1e-15, 1.0)
#     # ===================
#
#     N, k = D.shape
#     L_mat = calculate_expected_loss(Y, H, D, k)
#     w = cp.Variable(k, nonneg=True, name="w")
#     Q = cp.Variable((N, k), nonneg=True, name="Q")
#
#     main_terms = [cp.sum(cp.multiply(D[:, j], -cp.rel_entr(w[j], Q[:, j]))) for j in range(k)]
#     main_obj = cp.sum(main_terms)
#     smooth_obj = eta * cp.sum(cp.multiply(D, cp.log(Q)))
#     objective = cp.Maximize(main_obj + smooth_obj)
#
#     constraints = [cp.sum(w) == 1, w >= w_min, cp.sum(Q, axis=1) == 1, Q >= q_min]
#     for t in range(k):
#         eps_t = epsilon[t] if isinstance(epsilon, (list, np.ndarray)) else epsilon
#         constraints.append(w @ L_mat[:, t] <= eps_t)
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
# # ==========================================
# # HELPERS & LOADING
# # ==========================================
#
# def load_density_models():
#     """ Load Global Scaler, PCA, and GMMs """
#     global DENSITY_MODELS
#     try:
#         print("[INIT] Loading Density Models for Dynamic Update...")
#         DENSITY_MODELS['scaler'] = joblib.load(os.path.join(DENSITY_MODELS_DIR, "global_scaler.pkl"))
#         DENSITY_MODELS['pca'] = joblib.load(os.path.join(DENSITY_MODELS_DIR, "global_pca.pkl"))
#         DENSITY_MODELS['gmms'] = {}
#         for d in DOMAINS:
#             DENSITY_MODELS['gmms'][d] = joblib.load(os.path.join(DENSITY_MODELS_DIR, f"gmm_{d}.pkl"))
#         print("✅ Density models loaded successfully.")
#         return True
#     except Exception as e:
#         print(f"❌ Failed to load density models: {e}")
#         return False
#
#
# def update_H_and_D(loader, feature_extractor, classifiers, adapters, source_domains, temperature=5.0):
#     """
#     מחשב מחדש גם את H (תחזיות) וגם את D (צפיפות)
#     בהתבסס על הפיצ'רים החדשים שעברו דרך האדפטר (Ax).
#     """
#     scaler = DENSITY_MODELS['scaler']
#     pca = DENSITY_MODELS['pca']
#     gmms = DENSITY_MODELS['gmms']
#
#     feature_extractor.eval()
#     for net in adapters.values(): net.eval()
#     for net in classifiers.values(): net.eval()
#
#     all_H_list = []
#     raw_log_probs_list = []
#
#     with torch.no_grad():
#         for imgs, _ in loader:
#             imgs = imgs.to(device)
#             # 1. Base Features (ResNet)
#             base_feats = feature_extractor(imgs)
#
#             batch_H = []
#             batch_log_probs = []
#
#             for k_idx, dom in enumerate(source_domains):
#                 # 2. Apply Adapter: x' = A(x)
#                 adapter = adapters[dom]
#                 feats_adapted = adapter(base_feats)
#
#                 # 3. New H: Classifier(x')
#                 head = classifiers[dom].fc
#                 logits = head(feats_adapted)
#                 probs = F.softmax(logits, dim=1)
#                 batch_H.append(probs.cpu().numpy())
#
#                 # 4. New D: GMM(PCA(Scaler(x')))
#                 feats_np = feats_adapted.cpu().numpy()
#                 feats_scaled = scaler.transform(feats_np)
#                 z_pca = pca.transform(feats_scaled)
#
#                 gmm = gmms[dom]
#                 log_prob = gmm.score_samples(z_pca)
#                 batch_log_probs.append(log_prob)
#
#             # Stack results for batch
#             batch_H_stack = np.stack(batch_H, axis=2)  # (Batch, Classes, K)
#             all_H_list.append(batch_H_stack)
#
#             batch_log_p_stack = np.stack(batch_log_probs, axis=1)  # (Batch, K)
#             raw_log_probs_list.append(batch_log_p_stack)
#
#     # Combine all batches
#     H_new = np.concatenate(all_H_list, axis=0)
#     Raw_LogP = np.concatenate(raw_log_probs_list, axis=0)
#
#     # Normalize D (Temperature + Column Norm)
#     scaled_log_p = Raw_LogP / temperature
#     cols_max = np.max(scaled_log_p, axis=0, keepdims=True)
#     log_p_shifted = scaled_log_p - cols_max
#     p_unnormalized = np.exp(log_p_shifted)
#     col_sums = np.sum(p_unnormalized, axis=0, keepdims=True)
#     D_new = p_unnormalized / (col_sums + 1e-12)
#
#     # Clip for Solver Stability
#     D_new = np.clip(D_new, 1e-15, 1.0)
#     avg_max_prob = np.mean(np.max(D_new, axis=1))
#     zeros_count = np.sum(D_new < 1e-10)
#     print(f"   [DEBUG D_NEW] Avg Max Prob: {avg_max_prob:.4f} | Tiny Values Count: {zeros_count}/{D_new.size}")
#     return H_new, D_new
#
#
# def get_train_test_loaders_and_indices(domain, batch_size=32):
#     path = os.path.join(OFFICE_DIR, domain, 'images')
#     if not os.path.exists(path): path = os.path.join(OFFICE_DIR, domain)
#
#     tr_train = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
#     tr_test = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
#
#     try:
#         ds_full = datasets.ImageFolder(path)
#     except FileNotFoundError:
#         return None, None, None, None
#
#     N = len(ds_full)
#     rng = np.random.RandomState(42)
#     indices = rng.permutation(N)
#     split_point = int(0.8 * N)
#     train_idx = indices[:split_point]
#     test_idx = indices[split_point:]
#
#     class TransformedSubset(torch.utils.data.Dataset):
#         def __init__(self, subset, transform):
#             self.subset = subset
#             self.transform = transform
#
#         def __getitem__(self, index):
#             x, y = self.subset[index]
#             if self.transform: x = self.transform(x)
#             return x, y
#
#         def __len__(self): return len(self.subset)
#
#     train_ds = TransformedSubset(Subset(ds_full, train_idx), tr_train)
#     test_ds = TransformedSubset(Subset(ds_full, test_idx), tr_test)
#
#     train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
#     test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
#     return train_loader, test_loader, train_idx, test_idx
#
#
# def load_global_D():
#     print(f"[INIT] Loading Global D Matrix...")
#     try:
#         Global_D = np.load(D_MATRIX_PATH)
#         print(f" -> Global D Shape: {Global_D.shape}")
#         return Global_D
#     except Exception as e:
#         print(f"[ERROR] Failed to load D: {e}")
#         return None
#
#
# def infer_H_matrix(loader, classifiers, source_domains):
#     """ Helper to run inference and return stacked H and Y """
#     H_list = {d: [] for d in source_domains}
#     Y_list = []
#
#     with torch.no_grad():
#         for d in source_domains: classifiers[d].eval()
#         for imgs, labels in loader:
#             imgs = imgs.to(device)
#             Y_list.append(labels.numpy())
#             for d in source_domains:
#                 logits = classifiers[d](imgs)
#                 probs = F.softmax(logits, dim=1).cpu().numpy()
#                 H_list[d].append(probs)
#
#     Y_flat = np.concatenate(Y_list)
#     N = Y_flat.shape[0]
#
#     # One-hot Y
#     Y_onehot = np.zeros((N, NUM_CLASSES))
#     Y_onehot[np.arange(N), Y_flat] = 1
#
#     H_stacked = np.zeros((N, NUM_CLASSES, len(source_domains)))
#     for idx, d in enumerate(source_domains):
#         H_stacked[:, :, idx] = np.concatenate(H_list[d], axis=0)
#
#     return H_stacked, Y_onehot
#
#
# # ==========================================
# # --- BASELINES ---
# # ==========================================
#
# def run_baselines_train_test(target_domain, source_domains,
#                              train_loader_ordered, test_loader,
#                              D_train, D_test):
#     buf = io.StringIO()
#
#     classifiers = {}
#     for d in DOMAINS:
#         m = models.resnet50(weights=None)
#         m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
#         path = f"/data/nogaz/Convex_bounds_optimization/classifiers_new/{d}_classifier.pt"
#         if not os.path.exists(path): path = f"/data/nogaz/Convex_bounds_optimization/classifiers/{d}_224.pt"
#         if os.path.exists(path):
#             state = torch.load(path, map_location=device, weights_only=True)
#             m.load_state_dict(state)
#             classifiers[d] = m.to(device).eval()
#
#     H_train, Y_train_dummy = infer_H_matrix(train_loader_ordered, classifiers, source_domains)
#     H_test, Y_test_real = infer_H_matrix(test_loader, classifiers, source_domains)
#
#     check_correlation(D_test, H_test, Y_test_real, source_domains)
#
#     K = len(source_domains)
#     w_unif = np.ones(K) / K
#     acc_unif = evaluate_accuracy(w_unif, D_test, H_test, Y_test_real)
#
#     y_true = np.argmax(Y_test_real, axis=1)
#     source_preds = np.argmax(H_test, axis=1)
#     is_correct_matrix = (source_preds == y_true[:, None])
#     solvable = np.any(is_correct_matrix, axis=1)
#     max_acc = np.mean(solvable) * 100.0
#
#     buf.write(f"BASELINE UNIFORM: {acc_unif:.2f}%\n")
#     buf.write(f"BASELINE MAX_ORACLE: {max_acc:.2f}%\n")
#     print(f"BASELINE UNIFORM: {acc_unif:.2f}%")
#     print(f"BASELINE MAX_ORACLE: {max_acc:.2f}%")
#
#     dc_accuracies = []
#     best_w = None
#     best_acc = -1
#     D_train_expanded = np.tile(D_train[:, None, :], (1, NUM_CLASSES, 1))
#
#     print(" >>> Running DC Solver (5 seeds) [Train->Test]...")
#     for i in range(1): #range(5):
#         try:
#             dp = init_problem_from_model(Y_train_dummy, D_train_expanded, H_train, p=K, C=NUM_CLASSES)
#             problem = ConvexConcaveProblem(dp)
#             slv = ConvexConcaveSolver(problem, 42 + (i * 100), "err")
#             w_dc, _, _ = slv.solve(delta=1e-4)
#
#             if w_dc is not None:
#                 acc = evaluate_accuracy(w_dc, D_test, H_test, Y_test_real)
#                 dc_accuracies.append(acc)
#                 if acc > best_acc: best_acc, best_w = acc, w_dc
#         except Exception as e:
#             continue
#
#     if dc_accuracies:
#         avg = f"{np.mean(dc_accuracies):.2f} +/- {np.std(dc_accuracies):.2f}"
#         buf.write(f"BASELINE DC (5-Seeds): {avg} | Best: {best_acc:.2f} | w: {np.round(best_w, 3)}\n")
#         print(f"BASELINE DC: {avg} | Best: {best_acc:.2f}")
#     else:
#         buf.write("BASELINE DC: FAILED\n")
#         print("BASELINE DC: FAILED")
#
#     return buf.getvalue()
#
#
# # ==========================================
# # --- EM LOGIC ---
# # ==========================================
#
# def update_classifiers_pytorch(classifiers, train_loader, w_opt, Q_opt, source_domains,
#                                lr=1e-3,  # Increased LR
#                                hard_assignment=False,
#                                use_entropy=True,
#                                use_diversity=True,
#                                train_classifier=True,
#                                train_adapter=False,
#                                adapters=None):
#     optimizers = {}
#     original_heads = {}
#
#     for d in source_domains:
#         classifiers[d].eval()
#         original_heads[d] = classifiers[d].fc
#         classifiers[d].fc = nn.Identity()
#         params_to_update = []
#         if train_classifier:
#             original_heads[d].train()
#             params_to_update += list(original_heads[d].parameters())
#         else:
#             original_heads[d].eval()
#         if train_adapter and adapters is not None:
#             adapters[d].train()
#             params_to_update += list(adapters[d].parameters())
#         if params_to_update:
#             optimizers[d] = optim.SGD(params_to_update, lr=lr, momentum=0.9)
#         else:
#             optimizers[d] = None
#
#     Q_tensor = torch.tensor(Q_opt, dtype=torch.float32, device=device)
#     w_tensor = torch.tensor(w_opt, dtype=torch.float32, device=device)
#     batch_start = 0
#
#     for imgs, _ in train_loader:
#         imgs = imgs.to(device)
#         if batch_start >= Q_tensor.shape[0]: break
#         Q_batch = Q_tensor[batch_start:batch_start + imgs.size(0)]
#         if Q_batch.shape[0] != imgs.shape[0]: Q_batch = Q_batch[:imgs.shape[0]]
#
#         if hard_assignment: winning_indices = Q_batch.argmax(dim=1)
#
#         for opt in optimizers.values():
#             if opt is not None: opt.zero_grad()
#
#         loss_total = torch.tensor(0.0).to(device)
#
#         for k_idx, domain in enumerate(source_domains):
#             if optimizers[domain] is None: continue
#
#             if not train_classifier:
#                 with torch.no_grad():
#                     features = classifiers[domain](imgs)
#             else:
#                 features = classifiers[domain](imgs)
#
#             if train_adapter and adapters is not None:
#                 features = adapters[domain](features)
#
#             logits = original_heads[domain](features)
#
#             log_probs = F.log_softmax(logits, dim=1)
#             probs = F.softmax(logits, dim=1)
#             entropy = -(probs * log_probs).sum(dim=1) if use_entropy else torch.tensor(0.0).to(device)
#
#             q_values = Q_batch[:, k_idx]
#             threshold = 0.95
#
#             if hard_assignment:
#                 is_winner = (winning_indices == k_idx)
#                 mask = ((q_values > threshold) & is_winner).float()
#                 final_weight = mask
#             else:
#                 mask = (q_values > threshold).float()
#                 final_weight = q_values * w_tensor[k_idx]
#
#             if mask.sum() > 0:
#                 if use_entropy: loss_total += (final_weight * entropy * mask).sum() / (mask.sum() + 1e-8)
#                 if use_diversity:
#                     masked_probs = probs * mask.unsqueeze(1)
#                     avg_probs = masked_probs.sum(dim=0) / (mask.sum() + 1e-8)
#                     loss_total -= -(avg_probs * torch.log(avg_probs + 1e-8)).sum()
#
#         if loss_total.item() != 0:
#             # === DEBUG BLOCK 1: LOSS & GRADS ===
#             if batch_start == 0:  # נדפיס רק לבאץ' הראשון כדי לא להציף
#                 print(f"   [DEBUG TRAIN] Loss: {loss_total.item():.4f}")
#                 # נבדוק מה המצב של הפיצ'רים לפני ואחרי האדפטר
#                 with torch.no_grad():
#                     # בדיקה עבור הדומיין הראשון שפעיל
#                     d_debug = source_domains[0]
#                     if adapters is not None:
#                         feat_in = classifiers[d_debug](imgs)  # ResNet out
#                         feat_out = adapters[d_debug](feat_in)  # Adapter out
#                         diff = (feat_out - feat_in).abs().mean().item()
#                         norm_A = adapters[d_debug].weight.norm().item()
#                         print(f"   [DEBUG ADAPTER] {d_debug} | Feat Diff (L1): {diff:.4f} | Matrix Norm: {norm_A:.4f}")
#             # ===================================
#
#             loss_total.backward()
#             for opt in optimizers.values():
#                 if opt is not None: opt.step()
#
#         batch_start += imgs.size(0)
#         del imgs, features, logits, log_probs, probs, entropy, loss_total
#         torch.cuda.empty_cache()
#
#     for d in source_domains: classifiers[d].fc = original_heads[d]
#     return
#
#
# def run_single_em_experiment(target_domain, source_domains,
#                              train_loader_ordered, test_loader,
#                              D_train, D_test,
#                              solver_func, solver_kwargs, epsilon_vec,
#                              max_alt_iters=1,
#                              flag_update_classifier=False,
#                              flag_use_adapter=True,
#                              flag_use_entropy=True,
#                              flag_use_diversity=True):
#     # 1. Init Models & Adapters
#     classifiers = {}
#     resnet_extractor = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
#     resnet_extractor.fc = nn.Identity()
#     resnet_extractor.to(device).eval()
#
#     for d in DOMAINS:
#         m = models.resnet50(weights=None)
#         m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
#         path = f"/data/nogaz/Convex_bounds_optimization/classifiers_new/{d}_classifier.pt"
#         if not os.path.exists(path): path = f"/data/nogaz/Convex_bounds_optimization/classifiers/{d}_224.pt"
#         if os.path.exists(path):
#             state = torch.load(path, map_location=device, weights_only=True)
#             m.load_state_dict(state)
#             classifiers[d] = m.to(device)
#
#     adapters = {}
#     if flag_use_adapter:
#         for d in source_domains:
#             adapter = nn.Linear(2048, 2048, bias=True).to(device)
#             nn.init.eye_(adapter.weight)
#             nn.init.zeros_(adapter.bias)
#             adapters[d] = adapter
#
#     N_train = D_train.shape[0]
#     Y_train_dummy = np.zeros((N_train, NUM_CLASSES))
#
#     y_test_list = []
#     for _, lbls in test_loader: y_test_list.append(lbls.numpy())
#     Y_test_labels = np.concatenate(y_test_list)
#     Y_test_onehot = np.zeros((Y_test_labels.shape[0], NUM_CLASSES))
#     Y_test_onehot[np.arange(Y_test_labels.shape[0]), Y_test_labels] = 1
#
#     # Initialize State
#     current_D = D_train.copy()
#     current_H, _ = infer_H_matrix(train_loader_ordered, classifiers, source_domains)
#
#     # === PRE-CALC INITIAL ACCURACY (Before Adapter) ===
#     # נחשב את ה-H ההתחלתי של הטסט (ללא אדפטר) כדי שיהיה למה להשוות
#     H_test_initial, _ = infer_H_matrix(test_loader, classifiers, source_domains)
#     initial_acc = 0.0
#
#     final_w = None
#
#     # === MAIN EM LOOP ===
#     ITERATIONS = 1
#
#     for iter_k in range(ITERATIONS):
#
#         # A. SOLVER (E-Step)
#         current_args = {
#             'Y': Y_train_dummy, 'D': current_D, 'H': current_H,
#             'epsilon': epsilon_vec, 'solver_type': 'SCS'
#         }
#         current_args.update(solver_kwargs)
#
#         try:
#             w_opt = solver_func(**current_args)
#         except Exception:
#             w_opt = None
#
#         if w_opt is None: return "Infeasible", "None"
#         final_w = w_opt
#
#         # שמירת הדיוק ההתחלתי (באיטרציה הראשונה בלבד)
#         if iter_k == 0:
#             initial_acc = evaluate_accuracy(w_opt, D_test, H_test_initial, Y_test_onehot)
#
#         Q_opt = compute_Q_analytical(current_D, w_opt)
#
#         # B. UPDATE ADAPTER (M-Step)
#         update_classifiers_pytorch(
#             classifiers,
#             train_loader_ordered,
#             w_opt,
#             Q_opt,
#             source_domains,
#             lr=1e-3,
#             hard_assignment=True,
#             use_entropy=flag_use_entropy,
#             use_diversity=flag_use_diversity,
#             train_classifier=flag_update_classifier,
#             train_adapter=flag_use_adapter,
#             adapters=adapters
#         )
#
#         # === DEBUG BLOCK 2: PREDICTION DISTRIBUTION ===
#         with torch.no_grad():
#             imgs_dbg, _ = next(iter(train_loader_ordered))
#             imgs_dbg = imgs_dbg.to(device)
#             d_dbg = source_domains[0]
#             adapters[d_dbg].eval()
#             classifiers[d_dbg].eval()
#             # --- FIX: Handle Feature Extraction Correctly ---
#             # 1. שומרים את הראש המקורי ומנתקים אותו זמנית
#             real_head = classifiers[d_dbg].fc
#             classifiers[d_dbg].fc = nn.Identity()
#             # 2. מחלצים פיצ'רים (2048)
#             f = classifiers[d_dbg](imgs_dbg)
#             # 3. מחזירים את הראש למקום (כדי לא להרוס להמשך)
#             classifiers[d_dbg].fc = real_head
#             # 4. מעבירים באדפטר
#             f_adpt = adapters[d_dbg](f)
#             # 5. מעבירים בראש המקורי לקבלת תחזית
#             logits = real_head(f_adpt)
#             # ---------------------------------------------
#             preds = logits.argmax(dim=1).cpu().numpy()
#             unique, counts = np.unique(preds, return_counts=True)
#             print(f"   [DEBUG PREDS iter={iter_k}] Domain {d_dbg}: Unique Classes Predicted: {len(unique)}/31")
#             if len(unique) < 5:
#                 print(f"   ⚠️ WARNING: COLLAPSE DETECTED! Preds: {dict(zip(unique, counts))}")
#         # ==============================================
#         # C. UPDATE STATE (Feedback Loop)
#         if flag_use_adapter:
#             current_H, current_D = update_H_and_D(
#                 train_loader_ordered,
#                 resnet_extractor,
#                 classifiers,
#                 adapters,
#                 source_domains
#             )
#
#     # === FINAL EVALUATION ===
#     # Calculate Test Accuracy using the final learned Adapters
#     H_test_final, _ = update_H_and_D(
#         test_loader,
#         resnet_extractor,
#         classifiers,
#         adapters,
#         source_domains
#     )
#
#     final_acc = evaluate_accuracy(final_w, D_test, H_test_final, Y_test_onehot)
#     improvement = final_acc - initial_acc
#
#     return f"{initial_acc:.2f} -> {final_acc:.2f} ({improvement:+.2f})", str(np.round(final_w, 3))
#
#
# # ==========================================
# # MAIN EXECUTION
# # ==========================================
#
# def main():
#     global CURRENT_ACTIVE_DOMAINS
#     # 1. LOAD DENSITY MODELS FIRST
#     if not load_density_models(): return
#     Global_D_Matrix = load_global_D()
#     if Global_D_Matrix is None: return
#
#     results_path = os.path.join(RESULTS_DIR, "MSA_Full_Grid_EM_TrainTest_Results.txt")
#
#     with open(results_path, "w") as fp:
#         fp.write("MSA REPORT | Train on Train (Unsup) -> Eval on Test | With Baselines\n")
#         fp.write("=" * 80 + "\n\n")
#
#         smoothed_solvers = [
#             ("P3.21", solve_convex_problem_smoothed_kl),
#             ("P3.22", solve_convex_problem_domain_anchored_smoothed),
#             ("P3.23", solve_convex_problem_smoothed_original_p),
#         ]
#
#         eps_multipliers = [1.5, 2.0, 4, 7, 10, 20, 30, 50]
#         eta_values = [5e-3, 0.1, 0.5]
#         deltas = [0.1, 1, 1000]
#
#         domain_lengths = {}
#         for d in DOMAINS:
#             path = os.path.join(OFFICE_DIR, d, 'images')
#             if not os.path.exists(path): path = os.path.join(OFFICE_DIR, d)
#             ds = datasets.ImageFolder(path)
#             domain_lengths[d] = len(ds)
#
#         for target_domain in DOMAINS:
#             source_domains = [d for d in DOMAINS if d != target_domain]
#             CURRENT_ACTIVE_DOMAINS = source_domains
#
#             header = f"\n>>> TARGET: {target_domain} | SOURCES: {source_domains}"
#             print(header)
#             fp.write(header + "\n")
#             fp.write("-" * len(header) + "\n")
#
#             tr_load, te_load, tr_idx, te_idx = get_train_test_loaders_and_indices(target_domain)
#             train_ds_obj = tr_load.dataset
#             train_loader_ordered = DataLoader(train_ds_obj, batch_size=64, shuffle=False)
#
#             start_pos = 0
#             for d in DOMAINS:
#                 if d == target_domain: break
#                 start_pos += domain_lengths[d]
#
#             end_pos = start_pos + domain_lengths[target_domain]
#             D_target_full = Global_D_Matrix[start_pos:end_pos]
#             D_target_full = D_target_full[:, [DOMAINS.index(d) for d in source_domains]]
#
#             D_train = D_target_full[tr_idx]
#             D_test = D_target_full[te_idx]
#             print(f"   [Data] D_train: {D_train.shape}, D_test: {D_test.shape}")
#
#             print("\n   --- BASELINES ---")
#             fp.write("\n   --- BASELINES ---\n")
#             fp.write(run_baselines_train_test(
#                 target_domain, source_domains,
#                 train_loader_ordered, te_load,
#                 D_train, D_test
#             ))
#             fp.write("-" * 40 + "\n")
#
#             for eps_mult in eps_multipliers:
#                 epsilon_vec = np.array([(SOURCE_ERRORS[d] + 0.05) * eps_mult for d in source_domains])
#                 eps_str = f"EpsMult:{eps_mult}"
#
#                 # P3.10
#                 for delta in deltas:
#                     L_mat = calculate_expected_loss(None, None, None, len(source_domains))
#                     solver_kwargs = {'delta': delta, 'L_mat': L_mat}
#                     print(f" [EM-GRID] P3.10 | {eps_str} | Delta:{delta} ...")
#                     prog_str, w_str = run_single_em_experiment(
#                         target_domain, source_domains, train_loader_ordered, te_load,
#                         D_train, D_test, solve_convex_problem_mosek, solver_kwargs, epsilon_vec,
#                         flag_use_adapter=True,  # <--- MUST BE TRUE
#                         flag_update_classifier=False,  # <--- DON'T UPDATE HEAD
#                         flag_use_entropy=True,
#                         flag_use_diversity=True
#                     )
#                     line = f"P3.10 | {eps_str:<12} | Delta:{delta:<6} | TestAcc: {prog_str} | w:{w_str}"
#                     print(f"   >>> {line}")
#                     fp.write(line + "\n")
#                     fp.flush()
#
#                 # Smoothed
#                 for eta in eta_values:
#                     for name, func in smoothed_solvers:
#                         solver_kwargs = {'eta': eta}
#                         print(f" [EM-GRID] {name} | {eps_str} | Eta:{eta} ...")
#                         prog_str, w_str = run_single_em_experiment(
#                             target_domain, source_domains, train_loader_ordered, te_load,
#                             D_train, D_test, func, solver_kwargs, epsilon_vec,
#                             flag_use_adapter=True,  # <--- MUST BE TRUE
#                             flag_update_classifier=False,  # <--- DON'T UPDATE HEAD
#                             flag_use_entropy=True,
#                             flag_use_diversity=True
#                         )
#                         line = f"{name:<5} | {eps_str:<12} | Eta:{eta:<8} | TestAcc: {prog_str} | w:{w_str}"
#                         print(f"   >>> {line}")
#                         fp.write(line + "\n")
#                         fp.flush()
#
#             fp.write("\n" + "=" * 80 + "\n\n")
#
#     print(f"\n[DONE] Results saved to: {results_path}")
#
#
# if __name__ == "__main__":
#     main()