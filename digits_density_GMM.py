from __future__ import print_function
import os
import ssl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib

# =========================
# IMPORTANT: SSL PATCH (no change to data.py)
# =========================
ssl._create_default_https_context = ssl._create_unverified_context

# =========================
# Your loader module (unchanged)
# =========================
import data as Data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# CONFIG (DIGITS)
# =========================
BASE_EXP_DIR = "/data/nogaz/Convex_bounds_optimization/LatentFlow_Pixel_Experiments"
MODELS_DIR = os.path.join(BASE_EXP_DIR, "models", "gmm_models_soft_digits")
RESULTS_DIR = os.path.join(BASE_EXP_DIR, "results")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

DOMAINS = ["MNIST", "USPS", "SVHN"]   # same names as data.py expects
LATENT_DIM = 64
FINAL_MATRIX_NAME = "D_Matrix_FINAL_GMM_Soft_DIGITS.npy"

TEMPERATURE = 10.0
GMM_REG_COVAR = 0.1
N_COMPONENTS = 3

BATCH_SIZE = 64  # digits are small -> can be bigger


class ResNetFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        m.fc = nn.Identity()
        self.model = m.eval()

    def forward(self, x):
        return self.model(x)


@torch.no_grad()
def to_resnet_input(x):
    """
    x: (B,1,H,W) or (B,3,H,W), values in [0,1]
    Output: (B,3,224,224) normalized like ImageNet
    """
    if x.ndim != 4:
        raise ValueError(f"Expected 4D tensor, got {x.shape}")

    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)  # grayscale -> 3ch

    x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)

    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    x = (x - mean) / std
    return x


def extract_features_from_loader(resnet, loader):
    feats = []
    labels = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.cpu().numpy()
        x = to_resnet_input(x)
        f = resnet(x).detach().cpu().numpy()
        feats.append(f)
        labels.append(y)
    return np.concatenate(feats, axis=0), np.concatenate(labels, axis=0)


def run_full_pipeline():
    print(f"🚀 Starting Soft GMM Pipeline on DIGITS (Temp={TEMPERATURE}, Reg={GMM_REG_COVAR})...")

    resnet = ResNetFeatureExtractor().to(device)

    # ---------------------------------------------------------
    # STEP A: Extract TRAIN features + Fit balanced PCA/Scaler
    # ---------------------------------------------------------
    print("\n[1/5] Extracting features & Fitting Balanced PCA (using TRAIN split)...")
    features_by_domain = {}
    min_samples = float("inf")

    for dom in DOMAINS:
        print(f"   -> Extracting {dom} (train)...")
        tr_loader, te_loader, cfg = Data.get_data_loaders(dom, seed=42, batch_size=BATCH_SIZE, test_batch_size=BATCH_SIZE)

        X_tr, y_tr = extract_features_from_loader(resnet, tr_loader)
        features_by_domain[dom] = X_tr
        min_samples = min(min_samples, len(X_tr))

    print(f"   -> Balancing to {min_samples} samples per domain for PCA training.")
    rng = np.random.RandomState(42)
    balanced = []
    for dom in DOMAINS:
        idx = rng.choice(len(features_by_domain[dom]), min_samples, replace=False)
        balanced.append(features_by_domain[dom][idx])
    X_balanced = np.concatenate(balanced, axis=0)

    scaler = StandardScaler()
    X_balanced_scaled = scaler.fit_transform(X_balanced)

    pca = PCA(n_components=LATENT_DIM, whiten=True, random_state=42)
    pca.fit(X_balanced_scaled)

    joblib.dump(scaler, os.path.join(MODELS_DIR, "global_scaler.pkl"))
    joblib.dump(pca, os.path.join(MODELS_DIR, "global_pca.pkl"))
    print("   ✅ Saved Global Scaler & PCA")

    # ---------------------------------------------------------
    # STEP B: Fit GMM per domain (on TRAIN projected)
    # ---------------------------------------------------------
    print("\n[2/5] Fitting & Saving GMMs (Regularized) per domain...")
    gmms = {}
    projected_train = {}

    for dom in DOMAINS:
        X = features_by_domain[dom]
        Xs = scaler.transform(X)
        Z = pca.transform(Xs)
        projected_train[dom] = Z

        gmm = GaussianMixture(
            n_components=N_COMPONENTS,
            covariance_type="full",
            reg_covar=GMM_REG_COVAR,
            random_state=42,
        )
        gmm.fit(Z)
        gmms[dom] = gmm

        joblib.dump(gmm, os.path.join(MODELS_DIR, f"gmm_{dom}.pkl"))
        print(f"      ✅ Saved GMM for {dom}")

    # ---------------------------------------------------------
    # STEP C: Compute log-likelihoods on concatenated data
    # IMPORTANT: define the SAME row order you will assume later.
    # Here: concat TRAIN then TEST per domain, in DOMAINS order.
    # ---------------------------------------------------------
    print("\n[3/5] Computing Log-Likelihoods on concatenated data (train+test, domain-wise)...")
    all_Z = []
    true_labels = []
    lengths_meta = []

    for i, dom in enumerate(DOMAINS):
        print(f"   -> Projecting {dom} (train+test)...")
        tr_loader, te_loader, cfg = Data.get_data_loaders(dom, seed=42, batch_size=BATCH_SIZE, test_batch_size=BATCH_SIZE)

        X_tr, _ = extract_features_from_loader(resnet, tr_loader)
        X_te, _ = extract_features_from_loader(resnet, te_loader)

        X_all = np.concatenate([X_tr, X_te], axis=0)
        Xs = scaler.transform(X_all)
        Z_all = pca.transform(Xs)

        all_Z.append(Z_all)
        true_labels.extend([i] * len(Z_all))
        lengths_meta.append((dom, len(X_tr), len(X_te), len(Z_all)))

    all_Z_np = np.concatenate(all_Z, axis=0)
    N = len(all_Z_np)
    K = len(DOMAINS)
    true_labels = np.array(true_labels, dtype=int)

    print(f"   -> Total samples N={N}, domains K={K}")
    print("   -> Lengths meta (dom, n_train, n_test, n_total):")
    for row in lengths_meta:
        print("      ", row)
    joblib.dump(lengths_meta, os.path.join(RESULTS_DIR, "digits_D_row_order_meta.pkl"))

    Raw_LogP = np.zeros((N, K), dtype=np.float64)
    for k, dom in enumerate(DOMAINS):
        Raw_LogP[:, k] = gmms[dom].score_samples(all_Z_np)

    # ---------------------------------------------------------
    # STEP D: Column-normalize with temperature (sum_i D[i,k] = 1)
    # ---------------------------------------------------------
    print(f"\n[4/5] Normalizing with Temperature T={TEMPERATURE} (column-stochastic)...")

    scaled_logp = Raw_LogP / TEMPERATURE
    col_max = np.max(scaled_logp, axis=0, keepdims=True)
    logp_shift = scaled_logp - col_max
    p_unnorm = np.exp(logp_shift)

    col_sums = np.sum(p_unnorm, axis=0, keepdims=True)
    D_norm = p_unnorm / (col_sums + 1e-12)

    save_path = os.path.join(RESULTS_DIR, FINAL_MATRIX_NAME)
    np.save(save_path, D_norm)
    print(f"✅ Saved DIGITS matrix to: {save_path}")

    # ---------------------------------------------------------
    # STEP E: Diagnostics
    # ---------------------------------------------------------
    print("\n" + "=" * 40)
    print("DIAGNOSTICS")
    print("=" * 40)

    print(f"Column sums (must be ~1.0): {np.sum(D_norm, axis=0)}")

    pred = np.argmax(D_norm, axis=1)
    acc = accuracy_score(true_labels, pred)
    print(f"🎯 Density argmax accuracy vs (domain blocks): {acc * 100:.2f}%")

    min_nonzero = np.min(D_norm[D_norm > 0])
    print(f"Min non-zero value: {min_nonzero:.2e}")

    if min_nonzero < 1e-50:
        print("⚠️ Still very sharp → increase TEMPERATURE or reg_covar or reduce N_COMPONENTS.")
    else:
        print("✅ Values look numerically safe.")


if __name__ == "__main__":
    run_full_pipeline()
