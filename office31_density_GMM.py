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
# --- CONFIGURATION ---
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_EXP_DIR = "/data/nogaz/Convex_bounds_optimization/LatentFlow_Pixel_Experiments"
OFFICE_DIR = "/data/nogaz/Convex_bounds_optimization/Office-31"

MODELS_DIR = os.path.join(BASE_EXP_DIR, "models", "gmm_models_soft")
RESULTS_DIR = os.path.join(BASE_EXP_DIR, "results")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

DOMAINS = ['amazon', 'dslr', 'webcam']
LATENT_DIM = 64
FINAL_MATRIX_NAME = "D_Matrix_FINAL_GMM_Soft.npy"

# === PARAMETERS FOR SOFTER DISTRIBUTION ===
# 1. Temperature: מחלק את ה-LOG לפני האקספוננט. ערך גבוה = התפלגות שטוחה יותר.
TEMPERATURE = 5.0
# 2. Reg Covar: מוסיף "רוחב" לגאוסיאנים כדי למנוע קריסה לערכים אפסיים.
GMM_REG_COVAR = 0.1


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


def get_image_loader(domain, batch_size=128):
    path = os.path.join(OFFICE_DIR, domain, 'images')
    if not os.path.exists(path): path = os.path.join(OFFICE_DIR, domain)

    tr = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    try:
        ds = datasets.ImageFolder(path, transform=tr)
    except FileNotFoundError:
        print(f"❌ Error: Path not found for domain {domain}")
        return None
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


# ==========================================
# 2. MAIN PIPELINE
# ==========================================
def run_full_pipeline():
    print(f"🚀 Starting Soft GMM Pipeline (Temp={TEMPERATURE}, Reg={GMM_REG_COVAR})...")

    resnet = ResNetFeatureExtractor().to(device)

    # ---------------------------------------------------------
    # STEP A: Extract Features & Fit Global PCA/Scaler
    # ---------------------------------------------------------
    print("\n[1/5] Extracting features & Fitting Balanced PCA...")
    features_by_domain = {}
    min_samples = float('inf')

    for dom in DOMAINS:
        print(f"   -> Extracting {dom}...")
        loader = get_image_loader(dom)
        feats = []
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(device)
                f = resnet(x).cpu().numpy()
                feats.append(f)
        feats_np = np.concatenate(feats)
        features_by_domain[dom] = feats_np
        min_samples = min(min_samples, len(feats_np))

    print(f"   -> Balancing to {min_samples} samples per domain for PCA training.")
    balanced_data = []
    rng = np.random.RandomState(42)
    for dom in DOMAINS:
        indices = rng.choice(len(features_by_domain[dom]), min_samples, replace=False)
        balanced_data.append(features_by_domain[dom][indices])
    X_balanced = np.concatenate(balanced_data)

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

        # === HERE IS THE CHANGE: reg_covar ===
        gmm = GaussianMixture(
            n_components=5,
            covariance_type='full',
            reg_covar=GMM_REG_COVAR,  # <--- הקשחת הרגולריזציה
            random_state=42
        )
        gmm.fit(z)
        gmms[dom] = gmm

        joblib.dump(gmm, os.path.join(MODELS_DIR, f"gmm_{dom}.pkl"))
        print(f"      ✅ Saved Regularized GMM for {dom}")

    # ---------------------------------------------------------
    # STEP C: Compute Log Likelihoods
    # ---------------------------------------------------------
    print("\n[3/5] Computing Log-Likelihoods...")
    all_z_list = []
    true_labels = []
    for i, dom in enumerate(DOMAINS):
        z = data_projected[dom]
        all_z_list.append(z)
        true_labels.extend([i] * len(z))

    all_z_np = np.concatenate(all_z_list)
    N = len(all_z_np)
    K = len(DOMAINS)

    Raw_LogP = np.zeros((N, K))
    for k, dom in enumerate(DOMAINS):
        Raw_LogP[:, k] = gmms[dom].score_samples(all_z_np)

    # ---------------------------------------------------------
    # STEP D: Normalize Columns (With Temperature)
    # ---------------------------------------------------------
    print(f"\n[4/5] Normalizing with Temperature T={TEMPERATURE}...")

    # 1. Apply Temperature (Soften the peaks)
    #    log(P) / T  --->  P^(1/T)
    scaled_log_p = Raw_LogP / TEMPERATURE

    # 2. Shift for numeric stability
    cols_max = np.max(scaled_log_p, axis=0, keepdims=True)
    log_p_shifted = scaled_log_p - cols_max

    # 3. Exponentiate
    p_unnormalized = np.exp(log_p_shifted)

    # 4. Normalize Columns (Sum=1)
    col_sums = np.sum(p_unnormalized, axis=0, keepdims=True)
    D_norm = p_unnormalized / col_sums

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

    print(f"\nSample Row (Old was ~1e-160, Target is ~1e-5):")
    print(D_norm[0])

    # Check "Softness"
    min_val = np.min(D_norm[D_norm > 0])
    print(f"Minimum non-zero value: {min_val:.2e}")

    if min_val < 1e-50:
        print("⚠️ Still very sharp! Consider increasing TEMPERATURE.")
    else:
        print("✅ Values look mathematically safe.")


if __name__ == "__main__":
    run_full_pipeline()