import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms, models, datasets
import numpy as np
import os
import matplotlib.pyplot as plt
import ssl

# --- LIBRARIES ---
try:
    from torchcfm.conditional_flow_matching import TargetConditionalFlowMatcher
    from torchdiffeq import odeint
except ImportError:
    print("❌ Missing libraries. Please run: pip install torchcfm torchdiffeq")
    exit()

# --- CONFIG ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_EXP_DIR = "/data/nogaz/Convex_bounds_optimization/LatentFlow_Pixel_Experiments"
OFFICE_DIR = "/data/nogaz/Convex_bounds_optimization/Office-31"
MODELS_DIR = os.path.join(BASE_EXP_DIR, "models", "feature_flows")
RESULTS_DIR = os.path.join(BASE_EXP_DIR, "results")
PLOTS_DIR = os.path.join(BASE_EXP_DIR, "plots")

DOMAINS = ['amazon', 'dslr', 'webcam']
LATENT_DIM = 64
BATCH_SIZE = 64
TEMPERATURE = 1.0  # הורדתי טמפרטורה כדי לחדד הבדלים

ssl._create_default_https_context = ssl._create_unverified_context
plt.switch_backend('agg')


# ==========================================
# MODEL CLASSES
# ==========================================
class ResNetFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Identity()
        self.model.eval()

    def forward(self, x):
        return self.model(x)


class LinearAutoencoder(nn.Module):
    def __init__(self, input_dim=2048, latent_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512), nn.LeakyReLU(0.2),
            nn.Linear(512, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.BatchNorm1d(512), nn.LeakyReLU(0.2),
            nn.Linear(512, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z


class VectorFieldNet(nn.Module):
    def __init__(self, dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, t, x):
        if not torch.is_tensor(t): t = torch.tensor([t], dtype=x.dtype, device=x.device)
        if t.ndim == 0: t = t.unsqueeze(0)
        if t.shape[0] != x.shape[0]:
            t = t.view(-1, 1).expand(x.shape[0], 1)
        else:
            t = t.view(-1, 1)
        return self.net(torch.cat([x, t], dim=-1))


class FlowMatchingODE(nn.Module):
    def __init__(self, vector_field, dim):
        super().__init__()
        self.vf = vector_field
        self.dim = dim

    def forward(self, t, states):
        x = states[0]
        with torch.set_grad_enabled(True):
            x.requires_grad_(True)
            v = self.vf(t, x)
            div = 0.0
            for i in range(self.dim):
                grads = torch.autograd.grad(v[:, i].sum(), x, create_graph=True)[0]
                div += grads[:, i]
        return v, -div.unsqueeze(1)


def compute_ll(vector_field, z_target, dim):
    ode = FlowMatchingODE(vector_field, dim)
    bs = z_target.shape[0]
    traj = odeint(ode, (z_target, torch.zeros(bs, 1).to(z_target.device)),
                  torch.tensor([1.0, 0.0]).to(z_target.device), method='dopri5', atol=1e-4, rtol=1e-4)
    z0, dlogp = traj[0][-1], traj[1][-1]
    log_prior = -0.5 * dim * np.log(2 * np.pi) - 0.5 * (z0 ** 2).sum(1, keepdim=True)
    return log_prior + dlogp


def get_image_loader(domain):
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
        return None
    return DataLoader(ds, BATCH_SIZE, shuffle=True)


# ==========================================
# MAIN RUNNER
# ==========================================
def run_experiment():
    for p in [MODELS_DIR, RESULTS_DIR, PLOTS_DIR]: os.makedirs(p, exist_ok=True)

    print("Loading ResNet50 Feature Extractor...")
    resnet = ResNetFeatureExtractor().to(device)
    trained_models = {}

    # -------------------------------------------
    # PHASE 1: Extract, Normalize & Train
    # -------------------------------------------
    for dom in DOMAINS:
        print(f"\n🚀 Processing Domain: {dom}")
        img_loader = get_image_loader(dom)
        if img_loader is None: continue

        # A. Pre-Extract Features
        print("   [1/4] Extracting features...")
        features_list = []
        with torch.no_grad():
            for x, _ in img_loader:
                x = x.to(device)
                f = resnet(x)
                features_list.append(f.cpu())
        all_features = torch.cat(features_list).to(device)  # (N, 2048)

        # --- KEY FIX: INPUT STANDARDIZATION ---
        print("   [2/4] Normalizing Features...")
        mu_x = torch.mean(all_features, dim=0)
        std_x = torch.std(all_features, dim=0) + 1e-5  # Prevent div by zero

        # Save Scaler Stats!
        torch.save({'mu': mu_x, 'std': std_x}, os.path.join(MODELS_DIR, f"feature_scaler_{dom}.pt"))

        # Normalize
        features_norm = (all_features - mu_x) / std_x

        feat_ds = TensorDataset(features_norm)
        feat_loader = DataLoader(feat_ds, BATCH_SIZE, shuffle=True, drop_last=True)

        # B. Train Autoencoder
        print("   [3/4] Training Linear Autoencoder...")
        ae = LinearAutoencoder(2048, LATENT_DIM).to(device)
        opt_ae = optim.Adam(ae.parameters(), lr=1e-3)

        for ep in range(40):  # קצת יותר אפוקים
            for (f_batch,) in feat_loader:
                # הוספת רעש קטן כדי למנוע קריסה על האפסים
                f_noisy = f_batch + 0.01 * torch.randn_like(f_batch)

                rec, _ = ae(f_noisy)
                loss = nn.MSELoss()(rec, f_batch)
                opt_ae.zero_grad();
                loss.backward();
                opt_ae.step()
        ae.eval()

        # C. Train Flow Matching
        print("   [4/4] Training Flow Matching...")
        zs = []
        with torch.no_grad():
            for (f_batch,) in feat_loader:
                _, z = ae(f_batch)
                zs.append(z)
        zs = torch.cat(zs)
        mu_z, std_z = zs.mean(0), zs.std(0) + 1e-6
        zs_norm = (zs - mu_z) / std_z

        fm = VectorFieldNet(LATENT_DIM).to(device)
        opt_fm = optim.Adam(fm.parameters(), lr=1e-3)
        FM_loss = TargetConditionalFlowMatcher(sigma=0.0)
        z_loader = DataLoader(TensorDataset(zs_norm), BATCH_SIZE, shuffle=True, drop_last=True)

        for ep in range(50):
            for (z_batch,) in z_loader:
                t, xt, ut = FM_loss.sample_location_and_conditional_flow(torch.randn_like(z_batch), z_batch)
                vt = fm(t, xt)
                loss = torch.mean((vt - ut) ** 2)
                opt_fm.zero_grad();
                loss.backward();
                opt_fm.step()
        fm.eval()

        # Save Models
        torch.save(ae.state_dict(), os.path.join(MODELS_DIR, f"feature_ae_{dom}.pt"))
        torch.save(fm.state_dict(), os.path.join(MODELS_DIR, f"feature_fm_{dom}.pt"))
        torch.save({'mu': mu_z, 'std': std_z}, os.path.join(MODELS_DIR, f"feature_stats_{dom}.pt"))  # Latent stats

        trained_models[dom] = {'ae': ae, 'fm': fm, 'stats': (mu_z, std_z), 'scaler': (mu_x, std_x)}
        print(f"   ✅ Saved models + scaler for {dom}")

    # -------------------------------------------
    # PHASE 2: Check Matrix D
    # -------------------------------------------
    print(f"\n[ANALYSIS] Computing Matrix D for FULL DATASET...")

    # 1. Collect Raw Features
    all_feats = []
    labels = []

    for dom in DOMAINS:
        loader = get_image_loader(dom)
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(device)
                f = resnet(x)
                all_feats.append(f)
                labels.extend([dom] * x.size(0))
    X_all = torch.cat(all_feats)  # Raw 2048

    N = len(X_all)
    K = len(DOMAINS)
    Raw_LogP = np.zeros((N, K))

    with torch.no_grad():
        for k, dom_model in enumerate(DOMAINS):
            # Unpack
            ae = trained_models[dom_model]['ae']
            fm = trained_models[dom_model]['fm']
            mu_z, std_z = trained_models[dom_model]['stats']
            mu_x, std_x = trained_models[dom_model]['scaler']  # Get Scaler

            bs = 32
            scores = []
            for i in range(0, N, bs):
                f_batch = X_all[i:i + bs]

                # --- APPLY SAME SCALING ---
                f_norm = (f_batch - mu_x) / std_x

                _, z = ae(f_norm)
                z_norm = (z - mu_z) / std_z
                ll = compute_ll(fm, z_norm, LATENT_DIM)
                scores.append(ll.cpu().numpy())

            Raw_LogP[:, k] = np.concatenate(scores).flatten()

    # Smart Scaling
    global_max = np.max(Raw_LogP)
    print(f"   Max LogProb: {global_max:.2f}")

    scaled_log_p = (Raw_LogP - global_max) / TEMPERATURE
    # Clip extreme values
    scaled_log_p = np.clip(scaled_log_p, -50, 0)

    D = np.exp(scaled_log_p)

    # Optional: Row Normalize?
    # D = D / D.sum(axis=1, keepdims=True)

    np.save(os.path.join(RESULTS_DIR, "D_Matrix_Features_2048.npy"), D)
    print("✅ Matrix D saved. Check if rows are still uniform!")
    print("Sample row:", D[0])


if __name__ == "__main__":
    run_experiment()