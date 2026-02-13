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

OUTPUT_DIR = os.path.join(BASE_EXP_DIR, "analysis", "final_solver_analysis_strict_v5_oracle")
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


def calculate_empirical_jsd(gmm_p, gmm_q, Z_p, Z_q):
    n_max = 2000
    if len(Z_p) > n_max: Z_p = Z_p[np.random.choice(len(Z_p), n_max, replace=False)]
    if len(Z_q) > n_max: Z_q = Z_q[np.random.choice(len(Z_q), n_max, replace=False)]
    ll_p_xp = gmm_p.score_samples(Z_p)
    ll_q_xp = gmm_q.score_samples(Z_p)
    ll_p_xq = gmm_p.score_samples(Z_q)
    ll_q_xq = gmm_q.score_samples(Z_q)
    log_m_xp = -np.log(2) + np.logaddexp(ll_p_xp, ll_q_xp)
    log_m_xq = -np.log(2) + np.logaddexp(ll_p_xq, ll_q_xq)
    kl_p_m = np.mean(ll_p_xp - log_m_xp)
    kl_q_m = np.mean(ll_q_xq - log_m_xq)
    return 0.5 * kl_p_m + 0.5 * kl_q_m


# ==========================================
# --- PARSING RESULTS (STRICT + ORACLE) ---
# ==========================================
def parse_results_strict(filepath):
    if not os.path.exists(filepath): return []
    with open(filepath, 'r') as f:
        content = f.read()

    data = []
    blocks = content.split("TARGET:")[1:]

    BASELINE_NAMES = ["UNIFORM", "ORACLE", "ORACLE_ANY", "ORACLE_ANY_CORRECT", "Solver"]

    for block in blocks:
        lines = block.strip().split('\n')
        try:
            target_str = lines[0].split('|')[0].strip()
            target_list = ast.literal_eval(target_str)
            if len(target_list) != 2: continue
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
                        val = 0.0
                        if 'acc_' in acc_str:
                            val = float(acc_str.split(':')[1])
                        elif '±' in acc_str:
                            val = float(acc_str.split('±')[0])
                        else:
                            val = float(acc_str)

                        if name not in scores or val > scores[name]:
                            scores[name] = val

                        if "DC" in name: dc_vals.append(val)
                    except:
                        pass

        uniform = scores.get("UNIFORM")
        oracle = scores.get("ORACLE")  # Exact Match Oracle (Not ANY CORRECT)

        # 1. Best New Solver (Excluding Baseline & DC)
        best_new_solver_acc = -1
        best_new_solver_name = "None"

        for name, val in scores.items():
            is_baseline = name in BASELINE_NAMES
            is_dc = "DC" in name

            if not is_baseline and not is_dc:
                if val > best_new_solver_acc:
                    best_new_solver_acc = val
                    best_new_solver_name = name

        # 2. Worst DC
        worst_dc = min(dc_vals) if dc_vals else None

        if uniform is not None and best_new_solver_acc > 0:
            d1_short = target_list[0][:2]
            d2_short = target_list[1][:2]
            label = f"{d1_short}-{d2_short}"

            data.append({
                "Target_Set": target_set,
                "Label": label,
                "Dataset": "Unknown",
                "Acc_Uniform": uniform,
                "Acc_Oracle": oracle,
                "Acc_Best_New": best_new_solver_acc,
                "Acc_Worst_DC": worst_dc,
                "Solver_Name": best_new_solver_name,

                # Metrics
                "Gain_Uniform": best_new_solver_acc - uniform,
                "Gain_Worst_DC": (best_new_solver_acc - worst_dc) if worst_dc is not None else None,
                "Gain_Oracle": (best_new_solver_acc - oracle) if oracle is not None else None
            })

    return data


# ==========================================
# --- PLOTTING ---
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
    for ds in datasets:
        subset = df[df['Dataset'] == ds]
        if len(subset) > 1 and subset['JSD'].nunique() > 1:
            sns.regplot(
                data=subset, x='JSD', y=y_col, scatter=False,
                color=color_map[ds], ax=plt.gca(), ci=None,
                line_kws={'linestyle': '--', 'linewidth': 2, 'alpha': 0.7},
                truncate=False
            )

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
# --- MAIN ---
# ==========================================
def main():
    print("🚀 Starting Analysis (Strict Separation + ORACLE)...")
    all_rows = []

    for mode in ["OFFICE31", "OFFICE224"]:
        print(f"\n--- Processing {mode} ---")
        clf, scaler, pca, gmms = load_assets(mode)
        if not clf: continue

        real_z = get_real_z(mode, clf, scaler, pca)
        results = parse_results_strict(DATASET_CONFIGS[mode]['RESULTS_FILE'])

        for res in results:
            t_list = list(res['Target_Set'])
            d1, d2 = t_list[0], t_list[1]

            if d1 in real_z and d2 in real_z:
                jsd_val = calculate_empirical_jsd(gmms[d1], gmms[d2], real_z[d1], real_z[d2])
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