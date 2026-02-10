# import os
# import sys
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.manifold import TSNE
# from scipy.special import logsumexp
# import joblib
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import DataLoader, Subset
# from torchvision import models, transforms, datasets
#
# # ==========================================
# # --- CONFIGURATION & PATHS ---
# # ==========================================
# # בחרי את המוד הרצוי כאן
# DATASET_MODE = "OFFICE31"  # Options: "DIGITS", "OFFICE31", "OFFICE224"
#
# # הגדרת נתיבים (מותאם לקוד הראשון שלך)
# BASE_ROOT_DIR = "/data/nogaz/Convex_bounds_optimization"
# OFFICEHOME_DIR = os.path.join(BASE_ROOT_DIR, "OfficeHome")
# OFFICE31_DIR = os.path.join(BASE_ROOT_DIR, "Office-31")
# BASE_EXP_DIR = os.path.join(BASE_ROOT_DIR, "LatentFlow_Pixel_Experiments")
#
# CONFIGS = {
#     "DIGITS": {
#         "DOMAINS": ['MNIST', 'USPS', 'SVHN'],
#         "CLASSES": 10,
#         "INPUT_DIM": 784,
#         "MODELS_DIR": "gmm_models_soft_digits"
#     },
#     "OFFICE224": {  # Office-Home
#         "DOMAINS": ['Art', 'Clipart', 'Product', 'Real World'],
#         "CLASSES": 65,
#         "INPUT_DIM": 2048,
#         "MODELS_DIR": "gmm_models_soft_officehome"
#     },
#     "OFFICE31": {
#         "DOMAINS": ['amazon', 'dslr', 'webcam'],
#         "CLASSES": 31,
#         "INPUT_DIM": 2048,
#         "MODELS_DIR": "gmm_models_soft"
#     }
# }
#
# CURRENT_CFG = CONFIGS[DATASET_MODE]
# DOMAINS = CURRENT_CFG["DOMAINS"]
# NUM_CLASSES = CURRENT_CFG["CLASSES"]
# MODELS_DIR = os.path.join(BASE_EXP_DIR, "models", CURRENT_CFG["MODELS_DIR"])
# ANALYSIS_OUT_DIR = os.path.join(BASE_EXP_DIR, "analysis", f"{DATASET_MODE.lower()}_final_report_exact")
#
# os.makedirs(ANALYSIS_OUT_DIR, exist_ok=True)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#
# # ==========================================
# # --- HELPER CLASSES (From Code 1) ---
# # ==========================================
#
# class FeatureExtractor(nn.Module):
#     """
#     מחלקה שעוטפת את המודל כדי להחזיר גם את הפיצ'רים וגם את ה-Logits.
#     """
#
#     def __init__(self, original_model):
#         super(FeatureExtractor, self).__init__()
#         # הנחה: ResNet, מסירים את השכבה האחרונה (fc)
#         self.backbone = nn.Sequential(*list(original_model.children())[:-1])
#         self.head = original_model.fc
#
#     def forward(self, x):
#         feats = torch.flatten(self.backbone(x), 1)
#         logits = self.head(feats)
#         return feats, logits
#
#
# class TransformedSubset(torch.utils.data.Dataset):
#     def __init__(self, subset, transform):
#         self.subset = subset
#         self.transform = transform
#
#     def __getitem__(self, index):
#         x, y = self.subset[index]
#         if self.transform:
#             x = self.transform(x)
#         return x, y
#
#     def __len__(self):
#         return len(self.subset)
#
#
# # מחלקה פקטיבית למקרה של DIGITS אם צריך, או טעינה מותאמת
# class Grey_32_64_128_gp(nn.Module):
#     # כאן אני מניח שהמחלקה הזו קיימת בקובץ classifier.py שלך.
#     # אם הקוד הזה רץ באותה תיקייה כמו classifier.py, הוא ימצא אותה בייבוא למטה.
#     pass
#
#
# # ==========================================
# # --- DATA LOADING LOGIC (From Code 1) ---
# # ==========================================
#
# def get_domain_path(domain_name: str) -> str:
#     if DATASET_MODE == "OFFICE224":
#         p1 = os.path.join(OFFICEHOME_DIR, domain_name, "images")
#         p2 = os.path.join(OFFICEHOME_DIR, domain_name)
#         return p1 if os.path.exists(p1) else p2
#     elif DATASET_MODE == "OFFICE31":
#         p1 = os.path.join(OFFICE31_DIR, domain_name, "images")
#         p2 = os.path.join(OFFICE31_DIR, domain_name)
#         return p1 if os.path.exists(p1) else p2
#     return ""
#
#
# def get_train_test_loaders_and_indices(domain: str, seed: int, batch_size=64):
#     """
#     Replicates the exact logic from the first code:
#     - Splits 80/20 based on seed.
#     - Applies transforms for Office.
#     - Returns loaders.
#     """
#     if DATASET_MODE == 'DIGITS':
#         # בגלל שהקוד הזה רץ כ-Standalone, נשתמש בלוגיקה פנימית לטעינת Digits
#         # או שנניח ש-Data.get_data_loaders זמין.
#         # לצורך התיקון, אני מניח שצריך לייבא את Data כמו בקוד הראשון.
#         try:
#             import data as Data
#             train_loader, test_loader, _ = Data.get_data_loaders(domain, seed=seed, batch_size=batch_size)
#             return train_loader, test_loader
#         except ImportError:
#             print("Error: Could not import 'data' module for DIGITS. Ensure data.py is present.")
#             sys.exit(1)
#
#     # OFFICE Logic
#     path = get_domain_path(domain)
#     if not os.path.exists(path):
#         raise FileNotFoundError(f"Domain path not found: {path}")
#
#     ds_full = datasets.ImageFolder(path)
#     N = len(ds_full)
#
#     rng = np.random.RandomState(seed)
#     indices = rng.permutation(N)
#     split_point = int(0.8 * N)
#     test_idx = indices[split_point:]  # We only need test data for analysis
#
#     tr_test = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ])
#
#     test_ds = TransformedSubset(Subset(ds_full, test_idx), tr_test)
#     test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
#
#     return None, test_loader  # We only define test_loader for this analysis
#
#
# # ==========================================
# # --- CLASSIFIER & ASSETS LOADER ---
# # ==========================================
#
# def load_classifiers():
#     classifiers = {}
#     print(f"\n📂 Loading Classifiers for {DATASET_MODE}...")
#
#     for d in DOMAINS:
#         path = ""
#         model = None
#
#         if DATASET_MODE == 'DIGITS':
#             import classifier as ClSFR
#             model = ClSFR.Grey_32_64_128_gp()
#             path = f"./classifiers_new/{d}_classifier.pt"
#             if not os.path.exists(path):
#                 path = f"./classifiers/{d}_classifier.pt"
#
#             if os.path.exists(path):
#                 model.load_state_dict(torch.load(path, map_location=device))
#                 model = model.to(device).eval()
#                 classifiers[d] = model
#             else:
#                 print(f"⚠️ Warning: Model for {d} not found at {path}")
#
#         else:  # OFFICE
#             model = models.resnet50(weights=None)
#             model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
#
#             path = f"./classifiers/{d}_224.pt" if DATASET_MODE == "OFFICE224" else f"./classifiers/{d}_classifier.pt"
#             if not os.path.exists(path):
#                 path = f"./classifiers_new/{d}_classifier.pt"
#
#             if os.path.exists(path):
#                 model.load_state_dict(torch.load(path, map_location=device))
#                 # Wrap with FeatureExtractor to get (feats, logits)
#                 classifiers[d] = FeatureExtractor(model).to(device).eval()
#                 print(f"   ✅ Loaded Classifier: {d}")
#             else:
#                 print(f"⚠️ Warning: Model for {d} not found at {path}")
#
#     return classifiers
#
#
# def load_distribution_models():
#     print(f"\n📂 Loading GMMs, PCA, and Scaler from {MODELS_DIR}...")
#     try:
#         scaler = joblib.load(os.path.join(MODELS_DIR, "global_scaler.pkl"))
#         pca = joblib.load(os.path.join(MODELS_DIR, "global_pca.pkl"))
#         gmms = {}
#         for d in DOMAINS:
#             gmm_path = os.path.join(MODELS_DIR, f"gmm_{d}.pkl")
#             gmms[d] = joblib.load(gmm_path)
#             print(f"   ✅ Loaded GMM for {d}")
#         return scaler, pca, gmms
#     except FileNotFoundError as e:
#         print(f"❌ Error loading assets: {e}")
#         print(
#             "Please ensure you have run the training script that generates global_scaler.pkl, global_pca.pkl, and gmm_*.pkl")
#         sys.exit(1)
#
#
# # ==========================================
# # --- FEATURE EXTRACTION ---
# # ==========================================
#
# @torch.no_grad()
# def extract_real_data_features(classifiers, scaler, pca):
#     """
#     Extracts features from the REAL Test Set of each domain.
#     1. Loads images.
#     2. Passes through Backbone (or flattens for Digits).
#     3. Scales and projects via PCA.
#     4. Returns dictionary of Latent Features (Z).
#     """
#     print(f"\n📥 Extracting REAL test features (Latent Z)...")
#     domain_z_data = {}
#
#     for dom in DOMAINS:
#         _, test_loader = get_train_test_loaders_and_indices(dom, seed=1)
#
#         all_feats = []
#         for imgs, _ in test_loader:
#             imgs = imgs.to(device)
#
#             # DIGITS: Features are just flattened pixels (as per Code 1 logic)
#             if DATASET_MODE == 'DIGITS':
#                 f = imgs.view(imgs.size(0), -1)
#
#             # OFFICE: Features are ResNet50 output (before FC)
#             else:
#                 if dom not in classifiers:
#                     continue  # Skip if classifier missing
#                 f, _ = classifiers[dom](imgs)
#
#             all_feats.append(f.cpu().numpy())
#
#         if len(all_feats) > 0:
#             raw_concat = np.concatenate(all_feats)
#             # Transform to Shared Latent Space (Standardize -> PCA)
#             z_feats = pca.transform(scaler.transform(raw_concat))
#             domain_z_data[dom] = z_feats
#             print(f"   ✅ {dom}: Extracted {len(z_feats)} samples -> Projected to Z-dim {z_feats.shape[1]}")
#         else:
#             print(f"   ⚠️ {dom}: No data extracted.")
#
#     return domain_z_data
#
#
# # ==========================================
# # --- EXACT METRIC CALCULATIONS ---
# # ==========================================
#
# def calculate_exact_metrics(gmm_p, gmm_q, X_p):
#     """
#     Calculates metrics between distribution P (Source) and Q (Target)
#     evaluated on Real Data from P (X_p).
#
#     Mathematical Definitions:
#     1. LogLikelihood: ll_p = log P(x), ll_q = log Q(x)
#     2. KL(P||Q) ~= mean(ll_p - ll_q)
#     3. CrossEntropy(P, Q) = -mean(ll_q)
#     4. JSD(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M) -- (Requires sampling from both, done in wrapper)
#     """
#     # log(Probability)
#     ll_p = gmm_p.score_samples(X_p)  # log P(x) for x in X_p
#     ll_q = gmm_q.score_samples(X_p)  # log Q(x) for x in X_p
#     # KL Divergence P||Q
#     # E_p [ log P(x) - log Q(x) ]
#     kl_p_q = np.mean(ll_p - ll_q)
#     # Cross Entropy H(P, Q)
#     # - E_p [ log Q(x) ]
#     ce_p_q = -np.mean(ll_q)
#     return kl_p_q, ce_p_q, ll_p, ll_q
#
#
# def compute_pairwise_matrices(gmms, domain_data):
#     K = len(DOMAINS)
#     # Matrices to store results
#     mat_jsd = np.zeros((K, K))
#     mat_kl_fwd = np.zeros((K, K))  # KL(Row || Col)
#     mat_ce = np.zeros((K, K))  # H(Row, Col)
#     mat_kl_rev = np.zeros((K, K))  # KL(Col || Row) - for symmetry check
#
#     for i, src in enumerate(DOMAINS):
#         for j, tgt in enumerate(DOMAINS):
#             X_src = domain_data[src]
#             X_tgt = domain_data[tgt]
#             gmm_src = gmms[src]
#             gmm_tgt = gmms[tgt]
#             # --- Asymmetric Metrics (on Source Data) ---
#             kl_pq, ce_pq, ll_p_xsrc, ll_q_xsrc = calculate_exact_metrics(gmm_src, gmm_tgt, X_src)
#             mat_kl_fwd[i, j] = kl_pq
#             mat_ce[i, j] = ce_pq
#             # --- Reverse Metrics (on Target Data) ---
#             kl_qp, _, _, _ = calculate_exact_metrics(gmm_tgt, gmm_src, X_tgt)
#             mat_kl_rev[i, j] = kl_qp
#             # --- JSD (Symmetric) ---
#             # JSD = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
#             # We approximate this using the real data samples from both domains.
#
#             # Term 1: KL(P||M) on X_src
#             # log M(x) = log(0.5*P(x) + 0.5*Q(x)) = logsumexp([logP, logQ]) - log(2)
#             log_m_xsrc = np.logaddexp(ll_p_xsrc, ll_q_xsrc) - np.log(2)
#             kl_p_m = np.mean(ll_p_xsrc - log_m_xsrc)
#             # Term 2: KL(Q||M) on X_tgt
#             ll_q_xtgt = gmm_tgt.score_samples(X_tgt)
#             ll_p_xtgt = gmm_src.score_samples(X_tgt)
#             log_m_xtgt = np.logaddexp(ll_q_xtgt, ll_p_xtgt) - np.log(2)
#             kl_q_m = np.mean(ll_q_xtgt - log_m_xtgt)
#
#             mat_jsd[i, j] = 0.5 * kl_p_m + 0.5 * kl_q_m
#
#     return mat_jsd, mat_kl_fwd, mat_ce, mat_kl_rev
#
#
# # ==========================================
# # --- PLOTTING ---
# # ==========================================
#
# def plot_comprehensive_report(mats):
#     mat_jsd, mat_kl, mat_ce, mat_kl_rev = mats
#
#     # חישוב המינימום והמקסימום המשותפים לשתי מטריצות ה-KL כדי לאחד סקאלה
#     kl_min = min(mat_kl.min(), mat_kl_rev.min())
#     kl_max = max(mat_kl.max(), mat_kl_rev.max())
#
#     fig, axes = plt.subplots(2, 2, figsize=(24, 20))
#
#     # 1. JSD
#     sns.heatmap(mat_jsd, annot=True, fmt=".2f", ax=axes[0, 0],
#                 xticklabels=DOMAINS, yticklabels=DOMAINS, cmap="YlGnBu")
#     axes[0, 0].set_title(r"Jensen-Shannon Divergence (Symmetric)" + "\n" +
#                          r"$JSD(P||Q) = \frac{1}{2}D_{KL}(P||M) + \frac{1}{2}D_{KL}(Q||M)$", fontsize=14)
#     axes[0, 0].set_ylabel("Domain P")
#     axes[0, 0].set_xlabel("Domain Q")
#
#     # 2. Cross Entropy
#     sns.heatmap(mat_ce, annot=True, fmt=".2f", ax=axes[0, 1],
#                 xticklabels=DOMAINS, yticklabels=DOMAINS, cmap="magma")
#     axes[0, 1].set_title(r"Cross Entropy (Asymmetric)" + "\n" +
#                          r"$H(P,Q) = -\mathbb{E}_{x \sim P} [\log Q(x)]$", fontsize=14)
#     axes[0, 1].set_ylabel("Source Data (P)")
#     axes[0, 1].set_xlabel("Target Model (Q)")
#
#     # 3. KL Divergence (Forward)
#     sns.heatmap(mat_kl, annot=True, fmt=".2f", ax=axes[1, 0],
#                 xticklabels=DOMAINS, yticklabels=DOMAINS, cmap="OrRd",
#                 vmin=kl_min, vmax=kl_max)  # שימוש בסקאלה משותפת
#     axes[1, 0].set_title(r"KL Divergence (Forward)" + "\n" +
#                          r"$D_{KL}(P||Q) = \mathbb{E}_{x \sim P} [\log P(x) - \log Q(x)]$", fontsize=14)
#     axes[1, 0].set_ylabel("Source Data (P)")
#     axes[1, 0].set_xlabel("Target Model (Q)")
#
#     # 4. KL Divergence (Reverse / Symmetry Check)
#     sns.heatmap(mat_kl_rev, annot=True, fmt=".2f", ax=axes[1, 1],
#                 xticklabels=DOMAINS, yticklabels=DOMAINS, cmap="OrRd",  # אותה פלטת צבעים בדיוק
#                 vmin=kl_min, vmax=kl_max)  # שימוש בסקאלה משותפת
#     axes[1, 1].set_title(r"KL Divergence (Reverse)" + "\n" +
#                          r"$D_{KL}(Q||P) = \mathbb{E}_{x \sim Q} [\log Q(x) - \log P(x)]$", fontsize=14)
#     axes[1, 1].set_ylabel("Target Data (Q)")
#     axes[1, 1].set_xlabel("Source Model (P)")
#
#     plt.suptitle(f"Distribution Shift Analysis: {DATASET_MODE}\nEvaluated on REAL Test Data Projections",
#                  fontsize=24, fontweight='bold', y=0.99)
#     plt.tight_layout(rect=[0, 0.03, 1, 0.96])
#
#     out_path = os.path.join(ANALYSIS_OUT_DIR, "Final_Metrics_Report.png")
#     plt.savefig(out_path, dpi=300)
#     print(f"✅ Saved Heatmaps: {out_path}")
#
#
# def plot_tsne(domain_data):
#     print("🎨 Computing t-SNE...")
#     all_z = []
#     labels = []
#
#     for d in DOMAINS:
#         z = domain_data[d]
#         # Limit samples for speed/clarity if needed, typically 1000 is good
#         n_samples = min(len(z), 2000)
#         idx = np.random.choice(len(z), n_samples, replace=False)
#         all_z.append(z[idx])
#         labels.extend([d] * n_samples)
#
#     X_all = np.concatenate(all_z, axis=0)
#
#     tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
#     X_tsne = tsne.fit_transform(X_all)
#
#     plt.figure(figsize=(14, 10))
#     sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=labels, alpha=0.6, s=50, palette="husl", edgecolor='k',
#                     linewidth=0.3)
#     plt.title(f"t-SNE of Real Test Data (Latent Space): {DATASET_MODE}", fontsize=18, fontweight='bold')
#     plt.legend(title="Domain", loc='upper right')
#
#     out_path = os.path.join(ANALYSIS_OUT_DIR, "TSNE_Overlap.png")
#     plt.savefig(out_path, dpi=300)
#     print(f"✅ Saved t-SNE: {out_path}")
#
#
# # ==========================================
# # --- MAIN RUNNER ---
# # ==========================================
#
# def main():
#     print(f"🚀 Starting Analysis for {DATASET_MODE}")
#
#     # 1. Load Assets
#     classifiers = load_classifiers()
#     scaler, pca, gmms = load_distribution_models()
#
#     # 2. Extract Data
#     domain_z_data = extract_real_data_features(classifiers, scaler, pca)
#
#     # 3. Compute Metrics
#     metrics_matrices = compute_pairwise_matrices(gmms, domain_z_data)
#
#     # 4. Visualize
#     plot_comprehensive_report(metrics_matrices)
#     plot_tsne(domain_z_data)
#
#     print("\n✅ Analysis Complete.")
#
#
# if __name__ == "__main__":
#     main()


import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import re
import ast
from scipy.special import logsumexp
import itertools

# ==========================================
# --- CONFIGURATION ---
# ==========================================
# נתיב הבסיס הראשי
BASE_ROOT_DIR = "/data/nogaz/Convex_bounds_optimization"

# נתיבים מדויקים לקבצי התוצאות (כפי שביקשת)
RESULTS_FILES = {
    "OFFICE31": os.path.join(BASE_ROOT_DIR,
                             "results_OFFICE31_recreate/seed_1/Sweep_Results_1_PRE_D_True_art_ratios_True.txt"),
    "OFFICE224": os.path.join(BASE_ROOT_DIR,
                              "results_OFFICE224_recreate/seed_1/Sweep_Results_1_PRE_D_True_art_ratios_True.txt")
}

# נתיבים למודלים (GMMs) לצורך חישוב JSD
# מבוסס על המבנה המוכר מהקודים הקודמים שלך
MODELS_BASE = os.path.join(BASE_ROOT_DIR, "LatentFlow_Pixel_Experiments", "models")
MODELS_DIRS = {
    "OFFICE31": os.path.join(MODELS_BASE, "gmm_models_soft"),
    "OFFICE224": os.path.join(MODELS_BASE, "gmm_models_soft_officehome")
}

DOMAINS_CFG = {
    "OFFICE31": ['amazon', 'dslr', 'webcam'],
    "OFFICE224": ['Art', 'Clipart', 'Product', 'Real World']
}

# תיקיית פלט לגרפים
OUTPUT_DIR = os.path.join(BASE_ROOT_DIR, "LatentFlow_Pixel_Experiments", "analysis", "solver_improvement_analysis")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================
# --- JSD CALCULATION UTILS ---
# ==========================================
def calculate_jsd(gmm_p, gmm_q, n_samples=2000):
    """
    Approximates Symmetric JSD between two GMMs using Monte Carlo sampling.
    """
    # Sample from GMMs
    X_p = gmm_p.sample(n_samples)[0]
    X_q = gmm_q.sample(n_samples)[0]

    # Score samples (log prob)
    ll_p_xp = gmm_p.score_samples(X_p)
    ll_q_xp = gmm_q.score_samples(X_p)

    ll_p_xq = gmm_p.score_samples(X_q)
    ll_q_xq = gmm_q.score_samples(X_q)

    # log M(x) = -log(2) + logsumexp(logP(x), logQ(x))
    log_m_xp = -np.log(2) + np.logaddexp(ll_p_xp, ll_q_xp)
    log_m_xq = -np.log(2) + np.logaddexp(ll_p_xq, ll_q_xq)

    # KL(P||M)
    kl_p_m = np.mean(ll_p_xp - log_m_xp)
    # KL(Q||M)
    kl_q_m = np.mean(ll_q_xq - log_m_xq)

    return 0.5 * kl_p_m + 0.5 * kl_q_m


def load_gmms_and_compute_jsd(dataset_name):
    print(f"--- Computing JSDs for {dataset_name} ---")
    model_dir = MODELS_DIRS[dataset_name]
    domains = DOMAINS_CFG[dataset_name]

    # Load GMMs
    gmms = {}
    try:
        for d in domains:
            path = os.path.join(model_dir, f"gmm_{d}.pkl")
            if os.path.exists(path):
                gmms[d] = joblib.load(path)
            else:
                print(f"⚠️ Warning: GMM for {d} not found at {path}")
    except Exception as e:
        print(f"Error loading GMMs for {dataset_name}: {e}")
        return {}

    jsd_map = {}
    # Iterate over all unique pairs
    pairs = list(itertools.combinations(domains, 2))

    for d1, d2 in pairs:
        if d1 in gmms and d2 in gmms:
            val = calculate_jsd(gmms[d1], gmms[d2])
            # Key is frozenset to be order-agnostic (e.g., {Art, Clipart} is same as {Clipart, Art})
            key = frozenset([d1, d2])
            jsd_map[key] = val
            print(f"   JSD({d1}, {d2}) = {val:.4f}")

    return jsd_map


# ==========================================
# --- RESULT PARSING UTILS ---
# ==========================================
def parse_results_file(filepath, dataset_name):
    print(f"--- Parsing Results for {dataset_name} ---")
    if not os.path.exists(filepath):
        print(f"⚠️ File not found: {filepath}")
        return []

    data = []
    current_target = None

    # Regex to find "TARGET: ['dom1', 'dom2']"
    target_pat = re.compile(r"TARGET:\s*(\[.*?\])")

    # Regex to extract solver lines.
    # Matches: Name | ... | ... | Acc | ...
    # Handles "95.4" and "95.4±0.2"
    line_pat = re.compile(r"^\s*([^\s|]+.*?)\s*\|\s*.*?\s*\|\s*.*?\s*\|\s*([\d\.]+)(?:±[\d\.]+)?\s*\|")

    with open(filepath, 'r') as f:
        lines = f.readlines()

    block_data = {}

    for line in lines:
        # 1. Check if line declares a new TARGET
        m_target = target_pat.search(line)
        if m_target:
            # If we were processing a previous block, save it first
            if current_target and block_data:
                process_block(dataset_name, current_target, block_data, data)

            # Start new block
            try:
                current_target = ast.literal_eval(m_target.group(1))
                # We only want pairs for these plots (size 2)
                if len(current_target) != 2:
                    current_target = None
            except:
                current_target = None
            block_data = {}
            continue

        # 2. Extract Solver Accuracies within the current block
        if current_target:
            m_line = line_pat.search(line)
            if m_line:
                solver_name = m_line.group(1).strip()
                acc_str = m_line.group(2).strip()
                try:
                    acc = float(acc_str)
                    block_data[solver_name] = acc
                except:
                    pass

    # Process the very last block in the file
    if current_target and block_data:
        process_block(dataset_name, current_target, block_data, data)

    return data


def process_block(dataset_name, target_list, block_data, data_list):
    """Calculates improvement for a specific target pair."""
    target_set = frozenset(target_list)

    # Baselines
    uniform = block_data.get('UNIFORM')
    dc = block_data.get('DC (5-Seeds)')

    # Find Best Solver (Max accuracy among non-baseline solvers)
    best_solver_acc = -1
    best_solver_name = ""

    # Names to exclude when looking for "The Solver"
    ignore_list = ['UNIFORM', 'ORACLE', 'ORACLE_ANY', 'DC (5-Seeds)', 'Solver']

    for name, acc in block_data.items():
        if name not in ignore_list and acc > best_solver_acc:
            best_solver_acc = acc
            best_solver_name = name

    # We need at least Uniform and a valid Solver to plot improvement
    if uniform is not None and best_solver_acc > 0:
        entry = {
            "Dataset": dataset_name,
            "Target_Set": target_set,
            # Create a short label like "Art-Cli" or "Amz-Web"
            "Label": f"{target_list[0][:4]}-{target_list[1][:4]}",
            "Uniform": uniform,
            "Best_Solver": best_solver_acc,
            "Improvement_Uniform": best_solver_acc - uniform
        }

        # Add DC Improvement if DC exists
        if dc is not None:
            entry["DC"] = dc
            entry["Improvement_DC"] = best_solver_acc - dc
        else:
            entry["Improvement_DC"] = np.nan

        data_list.append(entry)


# ==========================================
# --- PLOTTING ---
# ==========================================
def plot_improvement(df, x_col, y_col, title, filename, ylabel):
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")

    # Scatter plot with different shapes/colors for datasets
    sns.scatterplot(
        data=df,
        x=x_col,
        y=y_col,
        hue='Dataset',
        style='Dataset',
        s=200,
        palette='deep',
        alpha=0.9
    )

    # Baseline line at 0
    plt.axhline(0, color='red', linestyle='--', linewidth=1.5, label='Baseline')

    # Add labels to points
    for i, row in df.iterrows():
        if pd.notna(row[x_col]) and pd.notna(row[y_col]):
            plt.text(
                row[x_col] + 0.1,
                row[y_col] + 0.1,
                row['Label'],
                fontsize=10,
                weight='bold'
            )

    plt.title(title, fontsize=18, fontweight='bold', pad=20)
    plt.xlabel("Symmetric JSD (Visual Distance)", fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.legend(loc='upper right', fontsize=12)
    plt.grid(True, alpha=0.3)

    out_path = os.path.join(OUTPUT_DIR, filename)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"✅ Saved Plot: {out_path}")
    plt.close()


# ==========================================
# --- MAIN ---
# ==========================================
def main():
    print("🚀 Starting Solver Analysis...")

    all_data = []

    for mode in ["OFFICE31", "OFFICE224"]:
        # A. Calculate JSDs for all pairs in this dataset
        jsd_map = load_gmms_and_compute_jsd(mode)
        if not jsd_map:
            print(f"⚠️ Skipping {mode} (No GMMs or JSDs computed).")
            continue

        # B. Parse Results File
        results = parse_results_file(RESULTS_FILES[mode], mode)
        if not results:
            print(f"⚠️ No data extracted from {RESULTS_FILES[mode]}")
            continue

        # C. Merge JSD into Results
        for entry in results:
            tgt_set = entry['Target_Set']
            if tgt_set in jsd_map:
                entry['JSD'] = jsd_map[tgt_set]
                all_data.append(entry)
            else:
                print(f"Warning: Could not find JSD for pair {entry['Label']}")

    if not all_data:
        print("❌ No valid data found to plot.")
        sys.exit(1)

    df = pd.DataFrame(all_data)

    print(f"\n📊 Analyzed {len(df)} pairs total.")
    print(df[['Dataset', 'Label', 'JSD', 'Improvement_Uniform', 'Improvement_DC']])

    # 1. Plot Improvement vs. Uniform
    plot_improvement(
        df,
        x_col='JSD',
        y_col='Improvement_Uniform',
        title='Best Solver Improvement over UNIFORM vs. JSD',
        filename='Solver_vs_Uniform_JSD.png',
        ylabel='Accuracy Gain (%) vs Uniform'
    )

    # 2. Plot Improvement vs. DC
    df_dc = df.dropna(subset=['Improvement_DC'])
    if not df_dc.empty:
        plot_improvement(
            df_dc,
            x_col='JSD',
            y_col='Improvement_DC',
            title='Best Solver Improvement over DC vs. JSD',
            filename='Solver_vs_DC_JSD.png',
            ylabel='Accuracy Gain (%) vs DC'
        )
    else:
        print("⚠️ No DC baseline data found in files, skipping second plot.")


if __name__ == "__main__":
    main()