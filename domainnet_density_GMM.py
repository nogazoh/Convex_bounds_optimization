import numpy as np
import os
# Prevent oversubscription from BLAS/OpenMP when using parallel jobs
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch.utils.data import ConcatDataset
import joblib
import data as Data


# ==========================================
# --- CPU / THREAD CONFIGURATION ---
# ==========================================
TOTAL_CPUS = os.cpu_count() or 1

# How many workers to use for PyTorch DataLoader
DATA_LOADER_WORKERS = min(16, max(4, TOTAL_CPUS // 2))


# ==========================================
# --- CONFIGURATION ---
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_EXP_DIR = "/data/nogaz/Convex_bounds_optimization/LatentFlow_Pixel_Experiments"
DOMAINNET_DIR = "/data/nogaz/Bi-ATEN/dataset/domainnet"

MODELS_DIR = os.path.join(BASE_EXP_DIR, "models", "gmm_optimized_domainnet")
RESULTS_DIR = os.path.join(BASE_EXP_DIR, "results")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
FEATURES_DIR = os.path.join(MODELS_DIR, "cached_features")
PROJECTED_DIR = os.path.join(MODELS_DIR, "projected_features")

os.makedirs(FEATURES_DIR, exist_ok=True)
os.makedirs(PROJECTED_DIR, exist_ok=True)


DOMAINS = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]
LATENT_DIM = 64
TEMPERATURE = 5.0
GMM_REG_COVAR = 0.1

# Define the range of components to test
COMPONENT_OPTIONS = [2, 5, 10, 20, 50]

# Reserve some CPUs for the system / other libraries
GMM_N_JOBS =  min(len(DOMAINS), max(2, TOTAL_CPUS // 4))


# ==========================================
# 1. HELPERS
# ==========================================
def feature_cache_path(domain):
    return os.path.join(FEATURES_DIR, f"{domain}_all_features.npy")

def projected_cache_path(domain):
    return os.path.join(PROJECTED_DIR, f"{domain}_all_projected.npy")


def gmm_cache_path(domain):
    return os.path.join(MODELS_DIR, f"gmm_{domain}_all.pkl")

def scaler_cache_path():
    return os.path.join(MODELS_DIR, "scaler_all.pkl")

def pca_cache_path():
    return os.path.join(MODELS_DIR, "pca_all.pkl")

def d_matrix_cache_path():
    return os.path.join(RESULTS_DIR, "D_Matrix_Optimized_DomainNet_ALL.npy")


class ResNetFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Identity()
        self.model.eval()

    def forward(self, x):
        return self.model(x)


def get_image_loader(domain, batch_size=128):
    train_file = os.path.join(DOMAINNET_DIR, f"{domain}_train.txt")
    test_file = os.path.join(DOMAINNET_DIR, f"{domain}_test.txt")

    tr = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_ds = Data.DomainNetSplitDataset(
        root=DOMAINNET_DIR,
        split_file=train_file,
        transform=tr
    )

    test_ds = Data.DomainNetSplitDataset(
        root=DOMAINNET_DIR,
        split_file=test_file,
        transform=tr
    )

    # IMPORTANT: fixed order = train first, then test
    full_ds = ConcatDataset([train_ds, test_ds])

    return DataLoader(
        full_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=DATA_LOADER_WORKERS,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(DATA_LOADER_WORKERS > 0),
    )


def extract_or_load_features(domain, resnet):
    cache_path = feature_cache_path(domain)

    if os.path.exists(cache_path):
        print(f"[{domain}] Loading cached features from {cache_path}")
        return np.load(cache_path)

    print(f"[{domain}] Extracting features...")
    loader = get_image_loader(domain)
    feats = []

    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            x = x.to(device)
            batch_feats = resnet(x).cpu().numpy()
            feats.append(batch_feats)

            if i % 20 == 0:
                print(f"   [{domain}] Batch {i}/{len(loader)}")

    feats = np.concatenate(feats, axis=0)
    np.save(cache_path, feats)
    print(f"[{domain}] Saved features to {cache_path}")
    return feats



def fit_or_load_scaler_pca(features_by_domain):
    scaler_path = scaler_cache_path()
    pca_path = pca_cache_path()

    if os.path.exists(scaler_path) and os.path.exists(pca_path):
        print("Loading cached scaler and PCA...")
        scaler = joblib.load(scaler_path)
        pca = joblib.load(pca_path)
        return scaler, pca

    print("Fitting scaler and PCA...")
    all_raw = np.concatenate(list(features_by_domain.values()), axis=0)

    scaler = StandardScaler()
    pca = PCA(n_components=LATENT_DIM, whiten=True, random_state=42)

    X_scaled = scaler.fit_transform(all_raw)
    pca.fit(X_scaled)

    joblib.dump(scaler, scaler_path)
    joblib.dump(pca, pca_path)
    print("Saved scaler and PCA.")
    return scaler, pca

def project_or_load_domain(domain, features, scaler, pca):
    cache_path = projected_cache_path(domain)

    if os.path.exists(cache_path):
        print(f"[{domain}] Loading cached projected features from {cache_path}")
        return np.load(cache_path)

    print(f"[{domain}] Projecting features...")
    z = pca.transform(scaler.transform(features))
    np.save(cache_path, z)
    print(f"[{domain}] Saved projected features to {cache_path}")
    return z


def fit_domain_gmm(args):
    dom, z = args
    model_path = gmm_cache_path(dom)

    if os.path.exists(model_path):
        print(f"[{dom}] Loading cached GMM from {model_path}")
        gmm = joblib.load(model_path)
        return dom, gmm

    gmm = find_best_gmm(z, dom)
    joblib.dump(gmm, model_path)
    print(f"[{dom}] Saved GMM to {model_path}")
    return dom, gmm


def maybe_load_final_d_matrix():
    path = d_matrix_cache_path()
    if os.path.exists(path):
        print(f"Loading cached final D matrix from {path}")
        return np.load(path)
    return None

# ==========================================
# 2. SELECTION METRIC: BIC MINIMIZATION
# ==========================================



def find_best_gmm(data, domain_name):
    """
    Fits multiple GMMs and selects the one with the lowest BIC score.
    """
    best_bic = np.inf
    best_gmm = None
    best_n = 0

    print(f"\n   --- Optimizing GMM for {domain_name} ---")
    for n in COMPONENT_OPTIONS:
        gmm = GaussianMixture(
            n_components=n,
            covariance_type="full",
            reg_covar=GMM_REG_COVAR,
            random_state=42
        )
        gmm.fit(data)
        bic = gmm.bic(data)
        print(f"      Components: {n:2d} | BIC: {bic:.2f}")

        if bic < best_bic:
            best_bic = bic
            best_gmm = gmm
            best_n = n

    print(f"   ✅ Selected N={best_n} for {domain_name}")
    return best_gmm


# ==========================================
# 3. MAIN PIPELINE
# ==========================================
def run_optimized_pipeline():
    print("\n" + "=" * 80)
    print("STARTING OPTIMIZED DOMAINNET PIPELINE")
    print("=" * 80)
    print(f"Detected CPUs: {TOTAL_CPUS}")
    print(f"Using DataLoader workers: {DATA_LOADER_WORKERS}")
    print(f"Using parallel GMM jobs: {GMM_N_JOBS}")
    print(f"Device: {device}")
    print(f"Domains: {DOMAINS}")
    print(f"Latent dim: {LATENT_DIM}")
    print(f"Temperature: {TEMPERATURE}")
    print(f"GMM reg_covar: {GMM_REG_COVAR}")
    print(f"Component options: {COMPONENT_OPTIONS}")
    print(f"Models dir: {MODELS_DIR}")
    print(f"Results dir: {RESULTS_DIR}")
    print("=" * 80)

    print("\n[CHECK] Looking for final cached D matrix...")
    final_d = maybe_load_final_d_matrix()
    if final_d is not None:
        print("[CHECK] Final D matrix already exists.")
        print(f"[CHECK] Loaded D shape: {final_d.shape}")
        print("[EXIT] Skipping full recomputation.")
        return final_d
    else:
        print("[CHECK] No final D matrix found. Continuing full pipeline.")

    print("\n[STEP 0] Building ResNet feature extractor...")
    resnet = ResNetFeatureExtractor().to(device)

    gpu_count = torch.cuda.device_count()
    print(f"[STEP 0] torch.cuda.device_count() = {gpu_count}")

    if gpu_count > 1:
        print(f"[STEP 0] Using DataParallel over {gpu_count} GPUs for feature extraction.")
        resnet = nn.DataParallel(resnet)

    resnet = resnet.to(device)
    resnet.eval()
    print("[STEP 0] ResNet ready and set to eval mode.")

    # STEP 1: Feature Extraction or Load
    print("\n" + "-" * 80)
    print("[STEP 1] FEATURE EXTRACTION / CACHE LOAD")
    print("-" * 80)

    features_by_domain = {}
    for dom in DOMAINS:
        cache_path = feature_cache_path(dom)
        print(f"\n[STEP 1] Domain: {dom}")
        print(f"[STEP 1] Feature cache path: {cache_path}")

        if os.path.exists(cache_path):
            print(f"[STEP 1] Cache exists for {dom}. Loading cached features...")
        else:
            print(f"[STEP 1] No cache found for {dom}. Extracting features from images...")

        features_by_domain[dom] = extract_or_load_features(dom, resnet)
        print(f"[STEP 1] {dom} features shape: {features_by_domain[dom].shape}")
        print(f"[STEP 1] {dom} features dtype: {features_by_domain[dom].dtype}")

    total_samples = sum(features_by_domain[d].shape[0] for d in DOMAINS)
    print(f"\n[STEP 1] Finished feature stage.")
    print(f"[STEP 1] Total number of samples across all domains: {total_samples}")

    split_lengths = {}
    for dom in DOMAINS:
        train_file = os.path.join(DOMAINNET_DIR, f"{dom}_train.txt")
        test_file = os.path.join(DOMAINNET_DIR, f"{dom}_test.txt")

        train_ds = Data.DomainNetSplitDataset(
            root=DOMAINNET_DIR,
            split_file=train_file,
            transform=None
        )
        test_ds = Data.DomainNetSplitDataset(
            root=DOMAINNET_DIR,
            split_file=test_file,
            transform=None
        )

        split_lengths[dom] = {
            "train": len(train_ds),
            "test": len(test_ds),
            "total": len(train_ds) + len(test_ds),
        }

    joblib.dump(split_lengths, os.path.join(MODELS_DIR, "domainnet_all_split_lengths.pkl"))
    print(f"[STEP 1] Saved split lengths to {os.path.join(MODELS_DIR, 'domainnet_all_split_lengths.pkl')}")
    print(f"[STEP 1] Split lengths: {split_lengths}")

    # STEP 2: Scaler + PCA or Load
    print("\n" + "-" * 80)
    print("[STEP 2] SCALER + PCA")
    print("-" * 80)
    print(f"[STEP 2] Scaler path: {scaler_cache_path()}")
    print(f"[STEP 2] PCA path: {pca_cache_path()}")

    if os.path.exists(scaler_cache_path()) and os.path.exists(pca_cache_path()):
        print("[STEP 2] Found cached scaler and PCA. Loading...")
    else:
        print("[STEP 2] Cached scaler/PCA not found. Fitting from scratch...")

    scaler, pca = fit_or_load_scaler_pca(features_by_domain)

    print("[STEP 2] Scaler and PCA ready.")
    print(f"[STEP 2] PCA n_components: {pca.n_components_ if hasattr(pca, 'n_components_') else LATENT_DIM}")
    if hasattr(pca, "explained_variance_ratio_"):
        print(f"[STEP 2] Sum explained variance ratio: {np.sum(pca.explained_variance_ratio_):.6f}")

    # STEP 3: Projection or Load
    print("\n" + "-" * 80)
    print("[STEP 3] PCA PROJECTION / CACHE LOAD")
    print("-" * 80)

    data_projected = {}
    for dom in DOMAINS:
        cache_path = projected_cache_path(dom)
        print(f"\n[STEP 3] Domain: {dom}")
        print(f"[STEP 3] Projection cache path: {cache_path}")

        if os.path.exists(cache_path):
            print(f"[STEP 3] Cache exists for {dom}. Loading projected features...")
        else:
            print(f"[STEP 3] No projection cache for {dom}. Projecting now...")

        data_projected[dom] = project_or_load_domain(dom, features_by_domain[dom], scaler, pca)
        print(f"[STEP 3] {dom} projected shape: {data_projected[dom].shape}")
        print(f"[STEP 3] {dom} projected dtype: {data_projected[dom].dtype}")

    print("\n[STEP 3] Finished projection stage.")

    # STEP 4: GMM fitting or Load
    print("\n" + "-" * 80)
    print("[STEP 4] GMM FITTING / CACHE LOAD")
    print("-" * 80)

    for dom in DOMAINS:
        path = gmm_cache_path(dom)
        exists_msg = "exists" if os.path.exists(path) else "missing"
        print(f"[STEP 4] {dom}: GMM path = {path} [{exists_msg}]")

    print(f"[STEP 4] Launching parallel GMM stage with n_jobs={GMM_N_JOBS} ...")
    gmm_results = joblib.Parallel(n_jobs=GMM_N_JOBS, backend="loky")(
        joblib.delayed(fit_domain_gmm)((dom, data_projected[dom]))
        for dom in DOMAINS
    )
    gmms = dict(gmm_results)

    print("[STEP 4] Finished GMM stage.")
    for dom in DOMAINS:
        gmm = gmms[dom]
        print(
            f"[STEP 4] {dom}: "
            f"n_components={getattr(gmm, 'n_components', 'N/A')}, "
            f"covariance_type={getattr(gmm, 'covariance_type', 'N/A')}"
        )

    # STEP 5: Probability Matrix & Diagnostics
    print("\n" + "-" * 80)
    print("[STEP 5] BUILDING PROBABILITY MATRIX")
    print("-" * 80)

    print("[STEP 5] Concatenating all projected features...")
    all_z = np.concatenate([data_projected[d] for d in DOMAINS], axis=0)
    print(f"[STEP 5] all_z shape: {all_z.shape}")

    print("[STEP 5] Computing raw log-probability matrix...")
    raw_log_p = np.array([gmms[d].score_samples(all_z) for d in DOMAINS]).T
    print(f"[STEP 5] raw_log_p shape: {raw_log_p.shape}")

    print("[STEP 5] Applying temperature-scaled normalization...")
    scaled_log_p = raw_log_p / TEMPERATURE
    print(f"[STEP 5] scaled_log_p shape: {scaled_log_p.shape}")

    p_unnorm = np.exp(scaled_log_p - np.max(scaled_log_p, axis=0))
    print(f"[STEP 5] p_unnorm shape: {p_unnorm.shape}")

    D_norm = p_unnorm / (np.sum(p_unnorm, axis=0) + 1e-12)
    print(f"[STEP 5] D_norm shape: {D_norm.shape}")

    save_path = d_matrix_cache_path()
    np.save(save_path, D_norm)
    print(f"[STEP 5] Saved final D matrix to: {save_path}")

    print("\n[STEP 5] Building true label vector for diagnostic accuracy...")
    true_labels = []
    for i, dom in enumerate(DOMAINS):
        n_dom = len(data_projected[dom])
        true_labels.extend([i] * n_dom)
        print(f"[STEP 5] {dom}: added {n_dom} labels with class index {i}")

    preds = np.argmax(D_norm, axis=1)
    print(f"[STEP 5] preds shape: {preds.shape}")
    print(f"[STEP 5] true_labels length: {len(true_labels)}")

    acc = accuracy_score(true_labels, preds)
    print(f"\n🎯 Final Optimized Accuracy: {acc * 100:.2f}%")

    print("\n" + "=" * 80)
    print("PIPELINE FINISHED SUCCESSFULLY")
    print("=" * 80)

    return D_norm


if __name__ == "__main__":
    run_optimized_pipeline()