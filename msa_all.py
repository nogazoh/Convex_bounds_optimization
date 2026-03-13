import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Subset
import numpy as np
import os
import io
import itertools
import time
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score
from torchvision import models

# --- CUSTOM MODULES ---
from dc import *
from vae_latent_est import vr_model, load_feature_dataset
import data as Data

# ניסיון ייבוא פונקציות הסולבר מהקבצים החיצוניים
try:
    from cvxpy_solver import solve_convex_problem_mosek
    from cvxpy_solver_per_domain import solve_convex_problem_per_domain
except ImportError:
    print("[!] Warning: Solver functions not found in external files. Ensure dc.py or cvxpy_solver.py are present.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# --- CONFIGURATION ---
# ==========================================
NUM_CLASSES = 65
BASE_EXP_DIR = "./Experiments_SeparatePCA"
UNIFIED_DIM = 1941  # הממד שנקבע בשלב ה-PCA

FLAGS = {
    "INCLUDE_TARGET_IN_SOURCES": True,
    "MIN_TARGET_GROUP_SIZE": 2,
    "MAX_TARGET_GROUP_SIZE": 4,
    "N_SWEEP_STEPS": 5,
    "N_JOBS": 2,
    "TARGET_SAMPLE_SIZE": 1200
}

ERROR_MATRIX = {
    'Art': {'Art': 0.0535, 'Clipart': 0.5979, 'Product': 0.4527, 'Real World': 0.3349},
    'Clipart': {'Art': 0.5309, 'Clipart': 0.0435, 'Product': 0.4403, 'Real World': 0.3796},
    'Product': {'Art': 0.5802, 'Clipart': 0.6037, 'Product': 0.0169, 'Real World': 0.3200},
    'Real World': {'Art': 0.3930, 'Clipart': 0.5624, 'Product': 0.2646, 'Real World': 0.0310}
}


# ==========================================
# --- CORE LOGIC: ON-THE-FLY MATRIX BUILD ---
# ==========================================

def get_clear_imbalanced_ratios(k):
    if k == 1: return [1.0]
    ratios = np.geomspace(1, 4, num=k)
    return (ratios / ratios.sum()).tolist()


def build_DP_matrices(target_info, source_domains, classifiers):
    """
    מחשב את D (Confidence מה-VAE) ואת H (תחזיות הקלאסיפייר) בזמן אמת.
    כולל דיבוג עמוק לאי-יתכנות (Infeasibility).
    """
    total_samples = sum(len(idx) for _, _, idx in target_info)
    K = len(source_domains)
    all_possible_sources = ['Art', 'Clipart', 'Product', 'Real World']

    Y = np.zeros((total_samples, NUM_CLASSES))
    D = np.zeros((total_samples, NUM_CLASSES, K))
    H = np.zeros((total_samples, NUM_CLASSES, K))

    # טעינת מודלי VAE לזיכרון
    vae_models = {}
    for src in all_possible_sources:
        model_path = os.path.join(BASE_EXP_DIR, src, f"vrs_{src}_model.pt")
        if os.path.exists(model_path):
            model = vr_model(UNIFIED_DIM, alpha_pos=0.5, alpha_neg=-0.5).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
            model.eval()
            vae_models[src] = model

    curr_global_idx = 0
    print(f"\n   [BUILD] Starting on-the-fly calculation for {total_samples} samples...")

    for target_name, loader, chosen_indices in target_info:
        pca_path = os.path.join(BASE_EXP_DIR, target_name, f"{target_name}_test.pt")
        target_pca_all, _ = torch.load(pca_path, weights_only=True)
        target_pca_subset = target_pca_all[chosen_indices].to(device)

        batch_count = 0
        for imgs, labels in loader:
            N_batch = len(imgs)
            imgs = imgs.to(device)
            if imgs.shape[-1] < 224:
                imgs = F.interpolate(imgs, size=(224, 224), mode='bilinear')

            for i in range(N_batch):
                Y[curr_global_idx + i, labels[i]] = 1

            with torch.no_grad():
                batch_pca = target_pca_subset[batch_count: batch_count + N_batch]

                for k, src_name in enumerate(source_domains):
                    _, logits = classifiers[src_name](imgs)
                    H[curr_global_idx: curr_global_idx + N_batch, :, k] = F.softmax(logits, dim=1).cpu().numpy()

                    if src_name in vae_models:
                        x_hat, _, _ = vae_models[src_name](batch_pca)
                        log_p = vae_models[src_name].compute_log_probabitility_bernoulli(x_hat, batch_pca)
                        D[curr_global_idx: curr_global_idx + N_batch, :, k] = np.tile(
                            log_p.cpu().numpy().reshape(-1, 1), (1, NUM_CLASSES)
                        )
            curr_global_idx += N_batch
            batch_count += N_batch

    # === [D-DEEP-DIAGNOSTIC] ===
    print(f"\n   {'='*65}")
    print(f"   [D-DEEP-DIAGNOSTIC] Analyzing Confidence Matrix")
    print(f"   {'='*65}")

    # 1. מי בחר את מי? (Dominance)
    raw_winners = np.argmax(D[:, 0, :], axis=1)
    unique, counts = np.unique(raw_winners, return_counts=True)
    print(f"   -> Raw VAE Winners (Dominance):")
    for idx, count in zip(unique, counts):
        print(f"      {source_domains[idx]:10}: {count:4} samples ({count/total_samples:.1%})")

    # 2. האם ה-VAE והקלאסיפייר מסונכרנים? (Alignment)
    h_preds = H.argmax(1) # [Samples, Sources]
    y_true = Y.argmax(1)
    print(f"   -> Source Accuracy on VAE-Claimed Samples (Alignment):")
    for k, src in enumerate(source_domains):
        mask = (raw_winners == k)
        if mask.any():
            acc = (h_preds[mask, k] == y_true[mask]).mean() * 100
            print(f"      {src:10}: {acc:.2f}% accurate on its own samples")

    # 3. מרווח ביטחון (Margin Analysis)
    sorted_D = np.sort(D[:, 0, :], axis=1)
    avg_margin = np.mean(sorted_D[:, -1] - sorted_D[:, -2])
    print(f"   -> Average Log-P Margin (1st vs 2nd): {avg_margin:.2f}")

    # --- נורמליזציה ---
    print("\n   [BUILD] Normalizing D matrix...")
    for i in range(total_samples):
        for j in range(NUM_CLASSES):
            vec = D[i, j, :]
            v_min, v_max = vec.min(), vec.max()
            D[i, j, :] = (vec - v_min) / (v_max - v_min) if v_max - v_min > 1e-9 else 1.0 / K

    # 4. בדיקת אנטרופיה מול Delta
    norm_D_sample = D[:, 0, :] + 1e-9
    norm_D_sample /= norm_D_sample.sum(axis=1, keepdims=True)
    avg_entropy = -np.mean(np.sum(norm_D_sample * np.log(norm_D_sample), axis=1))
    delta_limit = 1.1 * np.log(len(source_domains))
    print(f"   -> Avg Normalized Entropy: {avg_entropy:.4f} (Delta Limit: {delta_limit:.4f})")
    if avg_entropy < 0.1:
        print("   [!] WARNING: Entropy is very low. This often causes INFEASIBILITY.")
    print(f"   {'='*65}\n")

    return Y, D, H


# ==========================================
# --- SOLVER WRAPPER ---
# ==========================================

def run_solver_at_epsilon(Y, D, H, epsilon_vec, source_domains, all_domains):
    buf = io.StringIO()
    max_ent = np.log(len(source_domains)) if len(source_domains) > 1 else 0.1
    delta = 1.1 * max_ent
    avg_eps = np.mean(epsilon_vec)

    print(f"\n      [SOLVER] New Step: avg_eps={avg_eps:.4f} | Delta={delta:.4f}")
    print(f"               Eps Vector: {np.round(epsilon_vec, 4)}")

    for solver_type in ["GLOBAL", "PER_DOMAIN"]:
        try:
            if solver_type == "GLOBAL":
                w = solve_convex_problem_mosek(Y, D, H, delta=delta, epsilon=np.max(epsilon_vec))
            else:
                w = solve_convex_problem_per_domain(Y, D, H, delta=np.full(len(source_domains), delta),
                                                    epsilon=epsilon_vec)

            if w is None:
                print(f"         !!! {solver_type:<12} | Status: INFEASIBLE")
                buf.write(f"{solver_type:<12} | eps: {avg_eps:.4f} | INFEASIBLE\n")
                continue

            final_pred = ((D * H) * w.reshape(1, 1, -1)).sum(2).argmax(1)
            acc = accuracy_score(Y.argmax(1), final_pred) * 100
            w_full = np.zeros(len(all_domains))
            for i, src in enumerate(all_domains):
                if src in source_domains: w_full[i] = w[source_domains.index(src)]

            print(f"         ✅ {solver_type:<12} | ACC: {acc:.2f}% | w: {np.round(w_full, 3)}")
            buf.write(f"{solver_type:<12} | eps: {avg_eps:.4f} | acc: {acc:.2f}% | w: {np.round(w_full, 3)}\n")
        except Exception as e:
            buf.write(f"{solver_type:<12} | ERROR: {e}\n")
    return buf.getvalue()


# ==========================================
# --- MAIN EXECUTION ---
# ==========================================

def load_full_classifier(domain):
    m = models.resnet50(weights=None)
    m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
    m.load_state_dict(torch.load(f"./classifiers/{domain}_224.pt", map_location=device, weights_only=True))

    class ModelWrap(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.b = nn.Sequential(*list(m.children())[:-1])
            self.f = m.fc
        def forward(self, x):
            f = torch.flatten(self.b(x), 1)
            return f, self.f(f)

    return ModelWrap(m).to(device).eval()


def main():
    timestamp = time.strftime("%Y%m%d-%H%M")
    output_dir = os.path.join("./Results", f"RESEARCH_RUN_{timestamp}");
    os.makedirs(output_dir, exist_ok=True)
    all_domains = ['Art', 'Clipart', 'Product', 'Real World']

    print(f"\n🚀 STARTING EXPERIMENT: {timestamp}")
    classifiers = {d: load_full_classifier(d) for d in all_domains}

    for r in range(FLAGS["MIN_TARGET_GROUP_SIZE"], FLAGS["MAX_TARGET_GROUP_SIZE"] + 1):
        for target_comb in itertools.combinations(all_domains, r):
            target_comb = list(target_comb)
            source_list = all_domains if FLAGS["INCLUDE_TARGET_IN_SOURCES"] else [d for d in all_domains if d not in target_comb]
            ratios = get_clear_imbalanced_ratios(len(target_comb))

            print(f"\n{'=' * 65}\n[RUN] Target Mixture: {target_comb} | Ratios: {np.round(ratios, 2)}")
            target_info = []
            for i, dom in enumerate(target_comb):
                _, test_loader, _ = Data.get_data_loaders(dom, seed=1)
                n_sample = min(int(FLAGS['TARGET_SAMPLE_SIZE'] * ratios[i]), len(test_loader.dataset))
                indices = np.random.choice(np.arange(len(test_loader.dataset)), n_sample, replace=False)
                target_info.append((dom, DataLoader(Subset(test_loader.dataset, indices), batch_size=32), indices))
                print(f"      -> {dom:10}: {n_sample} samples")

            # חישוב מטריצות בזמן אמת כולל דיבוג עמוק
            Y, D, H = build_DP_matrices(target_info, source_list, classifiers)

            # Baseline - Uniform
            uni_w = np.ones(len(source_list)) / len(source_list)
            uni_pred = ((D * H) * uni_w.reshape(1, 1, -1)).sum(2).argmax(1)
            uni_acc = accuracy_score(Y.argmax(1), uni_pred) * 100
            print(f"   [BASELINE] Uniform Accuracy: {uni_acc:.2f}%")

            # יצירת גריד האפסילונים
            eps_grid = [
                np.linspace(ERROR_MATRIX[s][s], max([ERROR_MATRIX[t][s] for t in all_domains]), FLAGS["N_SWEEP_STEPS"])
                for s in source_list]
            eps_grid = np.array(eps_grid).T

            res_filename = os.path.join(output_dir, f"Results_{'_'.join(target_comb)}.txt")
            with open(res_filename, "w") as f:
                f.write(f"Target Mixture: {target_comb}\nBASELINE UNIFORM ACC: {uni_acc:.2f}%\n" + "=" * 60 + "\n")
                results = Parallel(n_jobs=FLAGS["N_JOBS"])(
                    delayed(run_solver_at_epsilon)(Y, D, H, ev, source_list, all_domains) for ev in eps_grid
                )
                for res in results:
                    if res: f.write(res)
            print(f"   [DONE] Results: {res_filename}")


if __name__ == "__main__":
    main()