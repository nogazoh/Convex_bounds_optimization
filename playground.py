import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import ast
import itertools
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import models, transforms, datasets
from scipy.special import logsumexp


# ==========================================
# --- JSD MODE FLAG ---
# ==========================================
# Choose how to estimate JSD:
#   "REAL" -> use real Z from the test set (may yield negative values due to model mismatch)
#   "GMM"  -> sample Z directly from the GMMs (JSD should be >= 0 up to MC noise)
JSD_ESTIMATION_MODE = "GMM"   # <-- change to "REAL" if you want the old behavior
JSD_NUM_SAMPLES = 2000        # samples per distribution when mode="GMM"
JSD_SEED = 0                  # reproducibility
JSD_CLIP_AT_ZERO = True       # clip tiny negative due to Monte-Carlo noise (recommended for "GMM")

# --- JSD MIX WEIGHTS FLAG ---
# "EQUAL" -> alpha=0.5 always
# "TRUE"  -> alpha is taken from TRUE RATIOS in the results file (normalized to the pair)
JSD_MIX_WEIGHTS_MODE = "TRUE"   # "EQUAL" or "TRUE"
JSD_NORMALIZE_BY_WEIGHT_ENTROPY = False  # optional: divide by H([a,1-a]) to make comparable across mixes

# ==========================================
# --- CONFIGURATION ---
# ==========================================
BASE_ROOT_DIR = "/data/nogaz/Convex_bounds_optimization"
BASE_EXP_DIR = os.path.join(BASE_ROOT_DIR, "LatentFlow_Pixel_Experiments")

OFFICEHOME_DIR = os.path.join(BASE_ROOT_DIR, "OfficeHome")
OFFICE31_DIR = os.path.join(BASE_ROOT_DIR, "Office-31")

DATASET_CONFIGS = {
    "OFFICE31": {
        "DOMAINS": ['amazon', 'dslr', 'webcam'],
        "CLASSES": 31,
        "INPUT_DIM": 2048,
        "MODELS_DIR": os.path.join(BASE_EXP_DIR, "models", "gmm_models_soft"),
        "RESULTS_FILE": os.path.join(BASE_ROOT_DIR,
                                     "results_OFFICE31_recreate/seed_1/Sweep_Results_1_PRE_D_True_art_ratios_True.txt"),
        "DATA_DIR": OFFICE31_DIR
    },
    "OFFICE224": {
        "DOMAINS": ['Art', 'Clipart', 'Product', 'Real World'],
        "CLASSES": 65,
        "INPUT_DIM": 2048,
        "MODELS_DIR": os.path.join(BASE_EXP_DIR, "models", "gmm_models_soft_officehome"),
        "RESULTS_FILE": os.path.join(BASE_ROOT_DIR,
                                     "results_OFFICE224_recreate/seed_1/Sweep_Results_1_PRE_D_True_art_ratios_True.txt"),
        "DATA_DIR": OFFICEHOME_DIR
    }
}

OUTPUT_DIR = os.path.join(BASE_EXP_DIR, "analysis", "solver_analysis_sampled_data_w")
os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# --- ASSETS LOADING ---
# ==========================================
class FeatureExtractor(nn.Module):
    def __init__(self, original_model):
        super(FeatureExtractor, self).__init__()
        self.backbone = nn.Sequential(*list(original_model.children())[:-1])
        self.head = original_model.fc

    def forward(self, x):
        feats = torch.flatten(self.backbone(x), 1)
        return feats, self.head(feats)


class TransformedSubset(torch.utils.data.Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform: x = self.transform(x)
        return x, y

    def __len__(self): return len(self.subset)


def get_test_loader(dataset_name, domain, seed=1, batch_size=64):
    cfg = DATASET_CONFIGS[dataset_name]
    candidates = [
        os.path.join(cfg['DATA_DIR'], domain, "images"),
        os.path.join(cfg['DATA_DIR'], domain),
        os.path.join(cfg['DATA_DIR'], "images", domain)
    ]
    path = None
    for p in candidates:
        if os.path.exists(p):
            path = p
            break
    if not path:
        return None

    ds_full = datasets.ImageFolder(path)
    N = len(ds_full)
    rng = np.random.RandomState(seed)
    indices = rng.permutation(N)
    split_point = int(0.8 * N)
    test_idx = indices[split_point:]

    tr_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_ds = TransformedSubset(Subset(ds_full, test_idx), tr_test)
    return DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)


def load_assets(dataset_name):
    cfg = DATASET_CONFIGS[dataset_name]
    classifiers = {}
    for d in cfg['DOMAINS']:
        possible_paths = [
            f"./classifiers/{d}_224.pt",
            f"./classifiers/{d}_classifier.pt",
            f"./classifiers_new/{d}_classifier.pt",
            os.path.join(BASE_ROOT_DIR, "classifiers_new", f"{d}_classifier.pt"),
            os.path.join(BASE_ROOT_DIR, "classifiers", f"{d}_classifier.pt")
        ]
        for path in possible_paths:
            if os.path.exists(path):
                m = models.resnet50(weights=None)
                m.fc = nn.Linear(m.fc.in_features, cfg['CLASSES'])
                try:
                    m.load_state_dict(torch.load(path, map_location=device))
                    classifiers[d] = FeatureExtractor(m).to(device).eval()
                    break
                except:
                    pass

    try:
        scaler = joblib.load(os.path.join(cfg['MODELS_DIR'], "global_scaler.pkl"))
        pca = joblib.load(os.path.join(cfg['MODELS_DIR'], "global_pca.pkl"))
        gmms = {}
        for d in cfg['DOMAINS']:
            gmms[d] = joblib.load(os.path.join(cfg['MODELS_DIR'], f"gmm_{d}.pkl"))
    except:
        return None, None, None, None

    return classifiers, scaler, pca, gmms


@torch.no_grad()
def get_real_z(dataset_name, classifiers, scaler, pca):
    cfg = DATASET_CONFIGS[dataset_name]
    z_data = {}
    for d in cfg['DOMAINS']:
        if d not in classifiers: continue
        loader = get_test_loader(dataset_name, d)
        if not loader: continue
        feats_list = []
        for imgs, _ in loader:
            imgs = imgs.to(device)
            f, _ = classifiers[d](imgs)
            feats_list.append(f.cpu().numpy())
        if feats_list:
            raw = np.concatenate(feats_list)
            z = pca.transform(scaler.transform(raw))
            z_data[d] = z
    return z_data


import re
import numpy as np
import ast
import os

def parse_results_strict(filepath):
    if not os.path.exists(filepath):
        return []
    with open(filepath, 'r') as f:
        content = f.read()

    data = []
    blocks = content.split("TARGET:")[1:]

    BASELINE_NAMES = ["UNIFORM", "ORACLE", "ORACLE_ANY", "ORACLE_ANY_CORRECT", "Solver"]

    for block in blocks:
        lines = block.strip().split('\n')
        header = lines[0].strip()

        # --- parse header line robustly ---
        # Example:
        # "['amazon', 'dslr'] | TRUE RATIOS: [0.2 0.8 0. ]"
        try:
            left, right = header.split("| TRUE RATIOS:")
            target_list = ast.literal_eval(left.strip())

            # ratios may come without commas => parse floats via regex
            ratio_str = right.strip()
            nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", ratio_str)
            true_ratios = np.array([float(x) for x in nums], dtype=float)

            if len(target_list) < 2 or true_ratios.size == 0:
                continue
            target_set = frozenset(target_list)
        except:
            continue

        scores = {}
        dc_vals = []

        for line in lines:
            if "|" in line:
                parts = [p.strip() for p in line.split('|')]
                if len(parts) >= 4:
                    name = parts[0]
                    acc_str = parts[3]
                    try:
                        if 'acc_' in acc_str:
                            val = float(acc_str.split(':')[1])
                        elif '±' in acc_str:
                            val = float(acc_str.split('±')[0])
                        else:
                            val = float(acc_str)

                        if name not in scores or val > scores[name]:
                            scores[name] = val
                        if "DC" in name:
                            dc_vals.append(val)
                    except:
                        pass

        uniform = scores.get("UNIFORM")
        oracle  = scores.get("ORACLE")

        best_new_solver_acc = -1
        best_new_solver_name = "None"
        for name, val in scores.items():
            if (name not in BASELINE_NAMES) and ("DC" not in name):
                if val > best_new_solver_acc:
                    best_new_solver_acc = val
                    best_new_solver_name = name

        worst_dc = min(dc_vals) if dc_vals else None

        if uniform is not None and best_new_solver_acc > 0:
            d1_short = target_list[0][:2]
            d2_short = target_list[1][:2]
            label = f"{d1_short}-{d2_short}"

            data.append({
                "Target_Set": target_set,
                "Target_List": target_list,
                "True_Ratios": true_ratios,
                "Label": label,
                "Dataset": "Unknown",
                "Acc_Uniform": uniform,
                "Acc_Oracle": oracle,
                "Acc_Best_New": best_new_solver_acc,
                "Acc_Worst_DC": worst_dc,
                "Solver_Name": best_new_solver_name,
                "Gain_Uniform": best_new_solver_acc - uniform,
                "Gain_Worst_DC": (best_new_solver_acc - worst_dc) if worst_dc is not None else None,
                "Gain_Oracle": (best_new_solver_acc - oracle) if oracle is not None else None
            })

    return data


# ==========================================
# --- PARSING RESULTS (STRICT + ORACLE) ---
# ==========================================
def plot_with_regression(df, y_col, y_label, title, filename):
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")

    datasets = sorted(df['Dataset'].unique())
    palette = sns.color_palette("bright", len(datasets))
    color_map = dict(zip(datasets, palette))

    # Points
    sns.scatterplot(
        data=df, x='JSD', y=y_col, hue='Dataset',
        style='Dataset', palette=color_map,
        s=250, edgecolor='k', alpha=0.9, zorder=3
    )

    # Regression
    # for ds in datasets:
    #     subset = df[df['Dataset'] == ds]
    #     if len(subset) > 1 and subset['JSD'].nunique() > 1:
    #         sns.regplot(
    #             data=subset, x='JSD', y=y_col, scatter=False,
    #             color=color_map[ds], ax=plt.gca(), ci=None,
    #             line_kws={'linestyle': '--', 'linewidth': 2, 'alpha': 0.7},
    #             truncate=False
    #         )

    # Baseline (0) if Gain
    if "Gain" in y_col:
        plt.axhline(0, color='gray', linestyle='-', linewidth=1.5, zorder=1)

    # Labels
    for i, row in df.iterrows():
        if pd.notna(row['JSD']) and pd.notna(row[y_col]):
            txt_color = 'red' if row[y_col] < 0 else 'black'
            # Adjust offset for negative values so text doesn't overlap line
            offset = 0.4 if row[y_col] >= 0 else -0.6

            plt.text(
                row['JSD'], row[y_col] + offset,
                row['Label'], fontsize=10, weight='bold',
                ha='center', color=txt_color, zorder=4
            )

    plt.title(title, fontsize=18, fontweight='bold', pad=15)
    plt.xlabel("Symmetric JSD (Closer to 0 = More Similar)", fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    plt.legend(loc='best', fontsize=12, frameon=True, title="Dataset")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=300)
    print(f"✅ Saved plot: {path}")
    plt.close()


# ==========================================
# --- PLOTTING ---
# def calculate_empirical_jsd(gmm_p, gmm_q, Z_p, Z_q):
#     n_max = 2000
#     if len(Z_p) > n_max: Z_p = Z_p[np.random.choice(len(Z_p), n_max, replace=False)]
#     if len(Z_q) > n_max: Z_q = Z_q[np.random.choice(len(Z_q), n_max, replace=False)]
#     ll_p_xp = gmm_p.score_samples(Z_p)
#     ll_q_xp = gmm_q.score_samples(Z_p)
#     ll_p_xq = gmm_p.score_samples(Z_q)
#     ll_q_xq = gmm_q.score_samples(Z_q)
#     log_m_xp = -np.log(2) + np.logaddexp(ll_p_xp, ll_q_xp)
#     log_m_xq = -np.log(2) + np.logaddexp(ll_p_xq, ll_q_xq)
#     kl_p_m = np.mean(ll_p_xp - log_m_xp)
#     kl_q_m = np.mean(ll_q_xq - log_m_xq)
#     return 0.5 * kl_p_m + 0.5 * kl_q_m

def _entropy_2(a):
    a = float(a)
    b = 1.0 - a
    eps = 1e-15
    a = np.clip(a, eps, 1 - eps)
    b = np.clip(b, eps, 1 - eps)
    return - (a * np.log(a) + b * np.log(b))

def _jsd_from_loglikes(ll_p_xp, ll_q_xp, ll_p_xq, ll_q_xq, alpha=0.5, normalize=False):
    """
    Weighted JSD:
      M = alpha P + (1-alpha) Q
      JSD = alpha * E_{x~P}[log p - log m] + (1-alpha) * E_{x~Q}[log q - log m]
    """
    eps = 1e-15
    a = float(np.clip(alpha, eps, 1 - eps))
    b = 1.0 - a

    # log m(x) = log( a * p(x) + b * q(x) )
    log_m_xp = np.logaddexp(np.log(a) + ll_p_xp, np.log(b) + ll_q_xp)
    log_m_xq = np.logaddexp(np.log(a) + ll_p_xq, np.log(b) + ll_q_xq)

    kl_p_m = np.mean(ll_p_xp - log_m_xp)  # E_{P}[log p - log m]
    kl_q_m = np.mean(ll_q_xq - log_m_xq)  # E_{Q}[log q - log m]

    jsd = a * kl_p_m + b * kl_q_m

    if normalize:
        H = _entropy_2(a)          # upper bound of weighted JSD
        jsd = jsd / max(H, 1e-12)  # avoid div by 0

    return float(jsd)



def calculate_jsd(gmm_p, gmm_q, Z_p=None, Z_q=None,
                  mode="REAL", n_samples=2000, seed=0,
                  clip_at_zero=False, alpha=0.5, normalize=False):

    rng = np.random.RandomState(seed)

    if mode.upper() == "REAL":
        if Z_p is None or Z_q is None:
            raise ValueError("REAL mode requires Z_p and Z_q.")
        if len(Z_p) > n_samples:
            Z_p = Z_p[rng.choice(len(Z_p), n_samples, replace=False)]
        if len(Z_q) > n_samples:
            Z_q = Z_q[rng.choice(len(Z_q), n_samples, replace=False)]

        ll_p_xp = gmm_p.score_samples(Z_p)
        ll_q_xp = gmm_q.score_samples(Z_p)
        ll_p_xq = gmm_p.score_samples(Z_q)
        ll_q_xq = gmm_q.score_samples(Z_q)

        jsd = _jsd_from_loglikes(ll_p_xp, ll_q_xp, ll_p_xq, ll_q_xq, alpha=alpha, normalize=normalize)

    elif mode.upper() == "GMM":
        gmm_p.random_state = seed
        gmm_q.random_state = seed + 1

        Xp, _ = gmm_p.sample(n_samples)
        Xq, _ = gmm_q.sample(n_samples)

        ll_p_xp = gmm_p.score_samples(Xp)
        ll_q_xp = gmm_q.score_samples(Xp)
        ll_p_xq = gmm_p.score_samples(Xq)
        ll_q_xq = gmm_q.score_samples(Xq)

        jsd = _jsd_from_loglikes(ll_p_xp, ll_q_xp, ll_p_xq, ll_q_xq, alpha=alpha, normalize=normalize)

    else:
        raise ValueError(f"Unknown mode={mode}. Use 'REAL' or 'GMM'.")

    if clip_at_zero:
        jsd = max(0.0, float(jsd))

    return float(jsd)

# ==========================================

def sanity_check_jsd_xx(gmms, real_z=None, mode="GMM"):
    print("\n🔎 Sanity check: JSD(X, X) should be ~0")
    for d, gmm in gmms.items():
        try:
            if mode.upper() == "REAL":
                if real_z is None or d not in real_z:
                    print(f"   {d}: (skipped - no real Z)")
                    continue
                val = calculate_jsd(gmm, gmm, Z_p=real_z[d], Z_q=real_z[d],
                                    mode="REAL", n_samples=JSD_NUM_SAMPLES, seed=JSD_SEED,
                                    clip_at_zero=False)
            else:
                val = calculate_jsd(gmm, gmm, mode="GMM",
                                    n_samples=JSD_NUM_SAMPLES, seed=JSD_SEED,
                                    clip_at_zero=JSD_CLIP_AT_ZERO)
            print(f"   {d}: JSD({d},{d}) = {val:.6f}")
        except Exception as e:
            print(f"   {d}: ERROR {repr(e)}")

# ==========================================
# --- MAIN ---
# ==========================================
def main():
    print("🚀 Starting Analysis (Strict Separation + ORACLE)...")
    all_rows = []

    for mode in ["OFFICE31", "OFFICE224"]:
        print(f"\n--- Processing {mode} ---")
        clf, scaler, pca, gmms = load_assets(mode)
        if not clf:
            continue

        real_z = get_real_z(mode, clf, scaler, pca)
        results = parse_results_strict(DATASET_CONFIGS[mode]['RESULTS_FILE'])
        # --- Sanity check JSD(X,X) ---
        sanity_check_jsd_xx(gmms, real_z=real_z, mode=JSD_ESTIMATION_MODE)

        for res in results:
            # ✅ use ordered list from file (not set)
            t_list = res.get("Target_List", None)
            if not t_list:
                t_list = list(res["Target_Set"])

            # ✅ keep only true 2-domain mixes
            if len(t_list) != 2:
                continue

            d1 = str(t_list[0]).strip()
            d2 = str(t_list[1]).strip()

            if d1 in real_z and d2 in real_z:
                # -------------------------------
                # NEW: choose alpha (mix weight) for weighted JSD
                # -------------------------------
                alpha = 0.5  # default: equal mix
                if JSD_MIX_WEIGHTS_MODE.upper() == "TRUE":
                    ratios = res.get("True_Ratios", None)  # <-- correct key name
                    if ratios is not None:
                        all_domains = DATASET_CONFIGS[mode]["DOMAINS"]
                        try:
                            idx1 = all_domains.index(d1)
                            idx2 = all_domains.index(d2)
                            r1 = float(ratios[idx1])
                            r2 = float(ratios[idx2])
                            s = r1 + r2
                            alpha = (r1 / s) if s > 0 else 0.5
                        except Exception:
                            alpha = 0.5  # fallback safely

                # jsd_val = calculate_empirical_jsd(gmms[d1], gmms[d2], real_z[d1], real_z[d2])
                if JSD_ESTIMATION_MODE.upper() == "REAL":
                    jsd_val = calculate_jsd(
                        gmms[d1], gmms[d2],
                        Z_p=real_z[d1], Z_q=real_z[d2],
                        mode="REAL", n_samples=JSD_NUM_SAMPLES, seed=JSD_SEED,
                        clip_at_zero=False,  # keep raw (can be negative)
                        alpha=alpha,
                        normalize=JSD_NORMALIZE_BY_WEIGHT_ENTROPY
                    )
                else:
                    jsd_val = calculate_jsd(
                        gmms[d1], gmms[d2],
                        mode="GMM", n_samples=JSD_NUM_SAMPLES, seed=JSD_SEED,
                        clip_at_zero=JSD_CLIP_AT_ZERO,
                        alpha=alpha,
                        normalize=JSD_NORMALIZE_BY_WEIGHT_ENTROPY
                    )

                res['JSD'] = jsd_val
                res['Dataset'] = mode
                all_rows.append(res)

                # Debug Print
                gain_u = res['Gain_Uniform']
                gain_oracle = res['Gain_Oracle']

                status_u = "🟢" if gain_u >= 0 else "🔴"
                status_or = "🟢" if (gain_oracle is not None and gain_oracle >= 0) else "🔴"  # Rare

                print(f"   [{mode}] {res['Label']} | JSD: {jsd_val:.2f} | Best: {res['Acc_Best_New']:.2f}")
                print(f"       -> vs Uniform: {gain_u:.2f} {status_u}")
                print(f"       -> vs Oracle:  {gain_oracle:.2f} {status_or}")
            else:
                print(f"   Skipping {res['Label']} (Missing Z data)")

    if not all_rows:
        print("❌ No data.")
        return

    df = pd.DataFrame(all_rows)
    print(f"\n📊 Total Points: {len(df)}")

    # 1. Abs Acc of BEST NEW
    plot_with_regression(df, 'Acc_Best_New', 'Best New Solver Accuracy (%)', 'Absolute Accuracy vs. JSD',
                         '1_Abs_Acc.png')

    # 2. Gain vs Uniform
    plot_with_regression(df, 'Gain_Uniform', 'Accuracy Gain (%) vs Uniform',
                         'Improvement (New Solver) over Uniform vs. JSD', '2_Gain_Uniform.png')

    # 3. Gain vs Worst DC
    df_dc = df.dropna(subset=['Gain_Worst_DC'])
    if not df_dc.empty:
        plot_with_regression(df_dc, 'Gain_Worst_DC', 'Accuracy Gain (%) vs Worst DC',
                             'Improvement (New Solver) over Worst DC vs. JSD', '3_Gain_WorstDC.png')

    # 4. Gain vs Oracle
    df_or = df.dropna(subset=['Gain_Oracle'])
    if not df_or.empty:
        plot_with_regression(df_or, 'Gain_Oracle', 'Distance from Oracle (%)', 'Gap to ORACLE (Ideal) vs. JSD',
                             '4_Gap_Oracle.png')

    print(f"\n✅ All Done. Folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()