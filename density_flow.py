import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms, datasets
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
DOMAINS = ['amazon', 'dslr', 'webcam']
IMG_SIZE = 64
LATENT_DIM = 64
BATCH_SIZE = 64
LR = 1e-3

# --- HYPER PARAMETER: TEMPERATURE ---
TEMPERATURE = 8.0

ssl._create_default_https_context = ssl._create_unverified_context
plt.switch_backend('agg')


# ==========================================
# 1. MODEL: CNN AUTOENCODER
# ==========================================
class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.BatchNorm2d(32), nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, latent_dim)
        )
        # Decoder
        self.decoder_fc = nn.Linear(latent_dim, 256 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64), nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32), nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(self.decoder_fc(z).view(-1, 256, 4, 4))
        return x_hat, z


# ==========================================
# 2. MODEL: FLOW MATCHING
# ==========================================
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
        if not torch.is_tensor(t):
            t = torch.tensor([t], dtype=x.dtype, device=x.device)
        if t.ndim == 0:
            t = t.unsqueeze(0)
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


# ==========================================
# 3. DATA UTILS
# ==========================================
def get_loaders_for_training(domain):
    """
    Used ONLY for Phase 1 (Training Generative Models).
    Splits 80/20 because we don't want to train the Flow on the Test set (Leakage).
    """
    path = os.path.join(OFFICE_DIR, domain, 'images')
    if not os.path.exists(path): path = os.path.join(OFFICE_DIR, domain)

    print(f"   -> Loading Training data from: {path}")
    tr = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor()
    ])

    try:
        ds = datasets.ImageFolder(path, transform=tr)
    except FileNotFoundError:
        return None, None

    N = len(ds)
    rng = np.random.RandomState(42)
    indices = rng.permutation(N)

    train_ds = torch.utils.data.Subset(ds, indices[:int(0.8 * N)])
    # We ignore test_ds here for training purposes
    return DataLoader(train_ds, BATCH_SIZE, True), None


# ==========================================
# 4. MAIN RUNNER
# ==========================================
def run_experiment():
    paths = {k: os.path.join(BASE_EXP_DIR, k) for k in ['models', 'plots', 'results']}
    for p in paths.values(): os.makedirs(p, exist_ok=True)

    ae_models = {}
    fm_models = {}
    stats_dict = {}

    # -------------------------------------------
    # PHASE 1: Train Models (on 80% split)
    # -------------------------------------------
    for dom in DOMAINS:
        print(f"\n🚀 Processing Domain: {dom}")
        train_loader, _ = get_loaders_for_training(dom)

        if train_loader is None: continue

        # A. Autoencoder
        print("   [1/2] Training Autoencoder...")
        ae = ConvAutoencoder(LATENT_DIM).to(device)
        opt_ae = optim.Adam(ae.parameters(), lr=1e-3)
        for ep in range(30):
            for x, _ in train_loader:
                x = x.to(device)
                x_hat, _ = ae(x)
                loss = nn.MSELoss()(x_hat, x)
                opt_ae.zero_grad();
                loss.backward();
                opt_ae.step()

        ae.eval()
        ae_models[dom] = ae

        # B. Flow Matching
        print("   [2/2] Training Flow...")
        zs = []
        with torch.no_grad():
            for x, _ in train_loader:
                _, z = ae(x.to(device))
                zs.append(z)
        zs = torch.cat(zs)
        mu, std = zs.mean(0), zs.std(0) + 1e-6
        stats_dict[dom] = (mu, std)
        zs_norm = (zs - mu) / std

        fm = VectorFieldNet(LATENT_DIM).to(device)
        opt_fm = optim.Adam(fm.parameters(), lr=1e-3)
        FM_loss = TargetConditionalFlowMatcher(sigma=0.0)
        dl_z = DataLoader(TensorDataset(zs_norm), BATCH_SIZE, True)

        for ep in range(50):
            for (z_batch,) in dl_z:
                t, xt, ut = FM_loss.sample_location_and_conditional_flow(torch.randn_like(z_batch), z_batch)
                vt = fm(t, xt)
                loss = torch.mean((vt - ut) ** 2)
                opt_fm.zero_grad();
                loss.backward();
                opt_fm.step()

        fm.eval()
        fm_models[dom] = fm

        # Save
        torch.save(ae.state_dict(), os.path.join(paths['models'], f"ae_{dom}.pt"))
        torch.save(fm.state_dict(), os.path.join(paths['models'], f"fm_{dom}.pt"))
        torch.save({'mu': mu, 'std': std}, os.path.join(paths['models'], f"stats_{dom}.pt"))

    # -------------------------------------------
    # PHASE 2: Analyze Matrix D on FULL DATASET
    # -------------------------------------------
    print(f"\n[ANALYSIS] Computing Matrix D for FULL DATASET (T={TEMPERATURE})...")

    # 1. Collect ALL Data (Deterministic Order)
    all_imgs = []
    labels = []

    # Standard transform (No Augmentation) for D computation
    tr = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor()
    ])

    for d in DOMAINS:
        path = os.path.join(OFFICE_DIR, d, 'images')
        if not os.path.exists(path): path = os.path.join(OFFICE_DIR, d)

        # Load FULL folder, NO shuffle, NO split
        ds = datasets.ImageFolder(path, transform=tr)
        dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

        print(f"   -> Collecting FULL data from: {d} ({len(ds)} images)")

        for x, _ in dl:
            all_imgs.append(x)
            labels.extend([d] * x.size(0))

    X_all = torch.cat(all_imgs)
    N = len(X_all)
    K = len(DOMAINS)

    print(f"   -> Total images for D matrix: {N}")

    # 2. Compute Likelihoods
    Raw_LogP = np.zeros((N, K))

    with torch.no_grad():
        for k, dom_model in enumerate(DOMAINS):
            print(f"   -> Evaluating Model: {dom_model} on full dataset...")
            ae = ae_models[dom_model]
            fm = fm_models[dom_model]
            mu, std = stats_dict[dom_model]

            bs = 32
            scores = []
            for i in range(0, N, bs):
                x_batch = X_all[i:i + bs].to(device)

                # Encode & Normalize
                _, z = ae(x_batch)
                z_norm = (z - mu) / std

                # Compute Likelihood
                ll = compute_ll(fm, z_norm, LATENT_DIM)
                scores.append(ll.cpu().numpy())

            Raw_LogP[:, k] = np.concatenate(scores).flatten()

    # 3. Scale and Save
    print(f"   -> Applying GLOBAL Likelihood scaling with T={TEMPERATURE}")
    Raw_LogP = Raw_LogP.astype(np.float64)
    global_max = np.max(Raw_LogP)
    scaled_log_p = (Raw_LogP - global_max) / TEMPERATURE
    min_clip = -100.0
    scaled_log_p = np.maximum(scaled_log_p, min_clip)
    D = np.exp(scaled_log_p)
    D = D / np.max(D)

    filename = f"D_Matrix_LatentFlow_T{int(TEMPERATURE)}_FULL_DATA.npy"
    save_path = os.path.join(paths['results'], filename)
    np.save(save_path, D)
    print(f"   -> Matrix saved to {save_path}")

    # 4. Plotting (Optional visual check)
    print(f"   -> Generating Plots...")
    D_plot = D / N
    fig, axes = plt.subplots(K, K, figsize=(15, 15))
    color_map = {d: plt.cm.tab10(idx) for idx, d in enumerate(DOMAINS)}

    for i in range(K):
        for j in range(K):
            ax = axes[i, j]
            if i == j:
                ax.set_title(f"Model: {DOMAINS[i]}")
                for d in DOMAINS:
                    idxs = [z for z, l in enumerate(labels) if l == d]
                    ax.hist(D_plot[idxs, i], bins=30, alpha=0.5, label=d, density=True, color=color_map[d])
                if i == 0: ax.legend()
            else:
                scatter_colors = [color_map[l] for l in labels]
                ax.scatter(D_plot[:, j], D_plot[:, i], c=scatter_colors, s=5, alpha=0.6)
                ax.set_xlabel(f"P({DOMAINS[j]})")
                ax.set_ylabel(f"P({DOMAINS[i]})")

    plt.suptitle(f"D Matrix Analysis (FULL DATA) - T={TEMPERATURE}", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(paths['plots'], f"LatentFlow_Analysis_T{int(TEMPERATURE)}_FULL.png"))
    print("✅ Done.")


if __name__ == "__main__":
    run_experiment()