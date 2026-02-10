import numpy as np
import os
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib

# ==========================================
# --- CONFIGURATION (OFFICE-HOME) ---
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_EXP_DIR = "/data/nogaz/Convex_bounds_optimization/LatentFlow_Pixel_Experiments"

# ✅ Change this to your Office-Home root
# Common layouts:
# 1) OFFICEHOME_DIR/Art/Alarm_Clock/*.jpg  (domain directly contains class folders)
# 2) OFFICEHOME_DIR/OfficeHomeDataset_10072016/Art/Alarm_Clock/*.jpg
OFFICEHOME_DIR = "/data/nogaz/Convex_bounds_optimization/OfficeHome"  # <-- EDIT IF NEEDED

MODELS_DIR = os.path.join(BASE_EXP_DIR, "models", "gmm_models_soft_officehome")
RESULTS_DIR = os.path.join(BASE_EXP_DIR, "results")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ✅ Office-Home domains
DOMAINS = ["Art", "Clipart", "Product", "Real World"]

LATENT_DIM = 64
FINAL_MATRIX_NAME = "D_Matrix_FINAL_GMM_Soft_OfficeHome.npy"

# === PARAMETERS FOR SOFTER DISTRIBUTION ===
TEMPERATURE = 5.0
GMM_REG_COVAR = 0.1
N_COMPONENTS = 5

# ==========================================
# 1. HELPERS
# ==========================================
class ResNetFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Identity()
        self.model.eval()

    def forward(self, x):
        return self.model(x)


def resolve_officehome_domain_path(officehome_dir: str, domain: str) -> str:
    """
    Tries to resolve the folder that ImageFolder should read for a given Office-Home domain.
    Supports common Office-Home directory layouts.
    """
    candidates = [
        os.path.join(officehome_dir, domain),
        os.path.join(officehome_dir, domain.lower()),
        os.path.join(officehome_dir, "OfficeHomeDataset_10072016", domain),
        os.path.join(officehome_dir, "OfficeHomeDataset_10072016", domain.lower()),
        os.path.join(officehome_dir, "OfficeHomeDataset", domain),
        os.path.join(officehome_dir, "OfficeHomeDataset", domain.lower()),
        os.path.join(officehome_dir, "office_home", domain),
        os.path.join(officehome_dir, "office_home", domain.lower()),
    ]
    for p in candidates:
        if os.path.isdir(p):
            # quick sanity: must contain at least one subfolder (class)
            subdirs = [d for d in os.listdir(p) if os.path.isdir(os.path.join(p, d))]
            if len(subdirs) > 0:
                return p

    # If nothing worked, raise with helpful message
    raise FileNotFoundError(
        f"Could not resolve Office-Home domain path for '{domain}'.\n"
        f"Tried candidates:\n" + "\n".join(candidates) + "\n\n"
        f"Fix by setting OFFICEHOME_DIR correctly (root that contains the domain folders)."
    )


def get_image_loader_officehome(domain, batch_size=128):
    domain_path = resolve_officehome_domain_path(OFFICEHOME_DIR, domain)

    tr = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    ds = datasets.ImageFolder(domain_path, transform=tr)
    print(f"   -> Loaded {domain}: {len(ds)} images from {domain_path}")
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)


# ==========================================
# 2. MAIN PIPELINE
# ==========================================
def run_full_pipeline():
    print(f"🚀 Starting Soft GMM Pipeline on Office-Home (Temp={TEMPERATURE}, Reg={GMM_REG_COVAR})...")

    resnet = ResNetFeatureExtractor().to(device)

    # ---------------------------------------------------------
    # STEP A: Extract Features & Fit Global PCA/Scaler
    # ---------------------------------------------------------
    print("\n[1/5] Extracting features & Fitting Balanced PCA...")
    features_by_domain = {}
    min_samples = float("inf")

    for dom in DOMAINS:
        print(f"   -> Extracting {dom}...")
        loader = get_image_loader_officehome(dom)

        feats = []
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(device, non_blocking=True)
                f = resnet(x).cpu().numpy()
                feats.append(f)

        feats_np = np.concatenate(feats, axis=0)
        features_by_domain[dom] = feats_np
        min_samples = min(min_samples, len(feats_np))

    print(f"   -> Balancing to {min_samples} samples per domain for PCA training.")
    rng = np.random.RandomState(42)
    balanced_data = []
    for dom in DOMAINS:
        idx = rng.choice(len(features_by_domain[dom]), min_samples, replace=False)
        balanced_data.append(features_by_domain[dom][idx])
    X_balanced = np.concatenate(balanced_data, axis=0)

    scaler = StandardScaler()
    X_balanced_scaled = scaler.fit_transform(X_balanced)

    pca = PCA(n_components=LATENT_DIM, whiten=True, random_state=42)
    pca.fit(X_balanced_scaled)

    joblib.dump(scaler, os.path.join(MODELS_DIR, "global_scaler.pkl"))
    joblib.dump(pca, os.path.join(MODELS_DIR, "global_pca.pkl"))
    print("   ✅ Saved Global Scaler & PCA")

    # ---------------------------------------------------------
    # STEP B: Fit GMMs (With Regularization) & Save
    # ---------------------------------------------------------
    print("\n[2/5] Fitting & Saving GMMs (Regularized)...")
    gmms = {}
    data_projected = {}

    for dom in DOMAINS:
        raw = features_by_domain[dom]
        scaled = scaler.transform(raw)
        z = pca.transform(scaled)
        data_projected[dom] = z

        gmm = GaussianMixture(
            n_components=N_COMPONENTS,
            covariance_type="full",
            reg_covar=GMM_REG_COVAR,
            random_state=42,
        )
        gmm.fit(z)
        gmms[dom] = gmm

        joblib.dump(gmm, os.path.join(MODELS_DIR, f"gmm_{dom}.pkl"))
        print(f"      ✅ Saved Regularized GMM for {dom}")

    # ---------------------------------------------------------
    # STEP C: Compute Log Likelihoods
    # ---------------------------------------------------------
    print("\n[3/5] Computing Log-Likelihoods on concatenated (domain-wise) data...")
    all_z_list = []
    true_labels = []
    for i, dom in enumerate(DOMAINS):
        z = data_projected[dom]
        all_z_list.append(z)
        true_labels.extend([i] * len(z))

    all_z_np = np.concatenate(all_z_list, axis=0)
    N = len(all_z_np)
    K = len(DOMAINS)
    print(f"   -> Total samples N={N}, domains K={K}")

    Raw_LogP = np.zeros((N, K), dtype=np.float64)
    for k, dom in enumerate(DOMAINS):
        Raw_LogP[:, k] = gmms[dom].score_samples(all_z_np)

    # ---------------------------------------------------------
    # STEP D: Normalize Columns (With Temperature)
    # ---------------------------------------------------------
    print(f"\n[4/5] Normalizing with Temperature T={TEMPERATURE}...")

    scaled_log_p = Raw_LogP / TEMPERATURE  # soften peaks
    cols_max = np.max(scaled_log_p, axis=0, keepdims=True)
    log_p_shifted = scaled_log_p - cols_max  # stability
    p_unnormalized = np.exp(log_p_shifted)

    col_sums = np.sum(p_unnormalized, axis=0, keepdims=True)
    D_norm = p_unnormalized / (col_sums + 1e-12)

    save_path = os.path.join(RESULTS_DIR, FINAL_MATRIX_NAME)
    np.save(save_path, D_norm)
    print(f"✅ Saved SOFT Matrix to: {save_path}")

    # ---------------------------------------------------------
    # STEP E: Diagnostics
    # ---------------------------------------------------------
    print("\n" + "=" * 40)
    print("DIAGNOSTICS")
    print("=" * 40)

    print(f"Column Sums (Must be ~1.0): {np.sum(D_norm, axis=0)}")

    pred_labels = np.argmax(D_norm, axis=1)
    acc = accuracy_score(true_labels, pred_labels)
    print(f"🎯 Density Accuracy: {acc * 100:.2f}%")

    print("\nSample Row:")
    print(D_norm[0])

    min_val = np.min(D_norm[D_norm > 0])
    print(f"Minimum non-zero value: {min_val:.2e}")

    if min_val < 1e-50:
        print("⚠️ Still very sharp! Consider increasing TEMPERATURE or increasing reg_covar / reducing N_COMPONENTS.")
    else:
        print("✅ Values look mathematically safe.")


if __name__ == "__main__":
    run_full_pipeline()
