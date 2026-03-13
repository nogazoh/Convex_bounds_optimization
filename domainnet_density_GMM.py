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
DOMAINNET_DIR = "/data/nogaz/Bi-ATEN/dataset/domainnet"

MODELS_DIR = os.path.join(BASE_EXP_DIR, "models", "gmm_optimized_domainnet")
RESULTS_DIR = os.path.join(BASE_EXP_DIR, "results")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

DOMAINS = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]
LATENT_DIM = 64
TEMPERATURE = 5.0
GMM_REG_COVAR = 0.1

# Define the range of components to test
COMPONENT_OPTIONS = [2, 5, 10, 20, 50]


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


def get_image_loader(domain, batch_size=64):
    path = os.path.join(DOMAINNET_DIR, domain)
    tr = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    ds = datasets.ImageFolder(path, transform=tr)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)


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
    resnet = ResNetFeatureExtractor().to(device)

    if torch.cuda.device_count() > 1:
        print(f"🚀 Using {torch.cuda.device_count()} GPUs for feature extraction!")
        resnet = nn.DataParallel(resnet)
    resnet = resnet.to(device)

    # STEP 1: Feature Extraction
    features_by_domain = {}
    for dom in DOMAINS:
        loader = get_image_loader(dom)
        feats = []
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(device)
                feats.append(resnet(x).cpu().numpy())
        features_by_domain[dom] = np.concatenate(feats, axis=0)

    # STEP 2: PCA & Scaling
    all_raw = np.concatenate(list(features_by_domain.values()), axis=0)
    scaler = StandardScaler()
    pca = PCA(n_components=LATENT_DIM, whiten=True, random_state=42)

    X_scaled = scaler.fit_transform(all_raw)
    pca.fit(X_scaled)

    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    joblib.dump(pca, os.path.join(MODELS_DIR, "pca.pkl"))

    # STEP 3: Optimized GMM Fitting
    gmms = {}
    data_projected = {}
    for dom in DOMAINS:
        z = pca.transform(scaler.transform(features_by_domain[dom]))
        data_projected[dom] = z
        gmms[dom] = find_best_gmm(z, dom)
        joblib.dump(gmms[dom], os.path.join(MODELS_DIR, f"gmm_{dom}.pkl"))

    # STEP 4: Probability Matrix & Diagnostics
    all_z = np.concatenate([data_projected[d] for d in DOMAINS], axis=0)
    raw_log_p = np.array([gmms[d].score_samples(all_z) for d in DOMAINS]).T

    # Softmax with Temperature
    scaled_log_p = raw_log_p / TEMPERATURE
    p_unnorm = np.exp(scaled_log_p - np.max(scaled_log_p, axis=0))
    D_norm = p_unnorm / (np.sum(p_unnorm, axis=0) + 1e-12)

    np.save(os.path.join(RESULTS_DIR, "D_Matrix_Optimized_DomainNet.npy"), D_norm)

    # Basic Accuracy Metric
    true_labels = []
    for i, dom in enumerate(DOMAINS):
        true_labels.extend([i] * len(data_projected[dom]))

    acc = accuracy_score(true_labels, np.argmax(D_norm, axis=1))
    print(f"\n🎯 Final Optimized Accuracy: {acc * 100:.2f}%")


if __name__ == "__main__":
    run_optimized_pipeline()