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
# Value 8.0 chosen based on Log-Gap analysis to reduce sharpness.
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
            nn.Conv2d(3, 32, 4, 2, 1),  # 32x32
            nn.BatchNorm2d(32), nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, 2, 1),  # 16x16
            nn.BatchNorm2d(64), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),  # 8x8
            nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),  # 4x4
            nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, latent_dim)
        )
        # Decoder
        self.decoder_fc = nn.Linear(latent_dim, 256 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 8x8
            nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 16x16
            nn.BatchNorm2d(64), nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # 32x32
            nn.BatchNorm2d(32), nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),  # 64x64
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(self.decoder_fc(z).view(-1, 256, 4, 4))
        return x_hat, z


# ==========================================
# 2. MODEL: FLOW MATCHING (FIXED BROADCASTING)
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
        # --- FIX: Robust t broadcasting ---
        # Ensure t is a tensor
        if not torch.is_tensor(t):
            t = torch.tensor([t], dtype=x.dtype, device=x.device)

        # If t is scalar (0-dim), make it 1-dim
        if t.ndim == 0:
            t = t.unsqueeze(0)

        # Broadcast t to [Batch_Size, 1]
        # We handle cases where t might already be (B, 1) or just (1)
        if t.shape[0] != x.shape[0]:
            t = t.view(-1, 1).expand(x.shape[0], 1)
        else:
            t = t.view(-1, 1)

        # Concatenate
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
    # Solve backwards
    traj = odeint(ode, (z_target, torch.zeros(bs, 1).to(z_target.device)),
                  torch.tensor([1.0, 0.0]).to(z_target.device), method='dopri5', atol=1e-4, rtol=1e-4)
    z0, dlogp = traj[0][-1], traj[1][-1]
    log_prior = -0.5 * dim * np.log(2 * np.pi) - 0.5 * (z0 ** 2).sum(1, keepdim=True)
    return log_prior + dlogp


# ==========================================
# 3. DATA & UTILS
# ==========================================
def get_loaders(domain):
    path = os.path.join(OFFICE_DIR, domain, 'images')
    if not os.path.exists(path): path = os.path.join(OFFICE_DIR, domain)

    print(f"   -> Loading data from: {path}")
    tr = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor()
    ])

    try:
        ds = datasets.ImageFolder(path, transform=tr)
    except FileNotFoundError:
        print(f"CRITICAL ERROR: Could not find data at {path}")
        return None, None

    N = len(ds)
    # IMPORTANT: Fixed seed for consistency with Solver script
    rng = np.random.RandomState(42)
    indices = rng.permutation(N)

    train_ds = torch.utils.data.Subset(ds, indices[:int(0.8 * N)])
    test_ds = torch.utils.data.Subset(ds, indices[int(0.8 * N):])

    return DataLoader(train_ds, BATCH_SIZE, True), DataLoader(test_ds, BATCH_SIZE, False)


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
    # PHASE 1: Train Models
    # -------------------------------------------
    for dom in DOMAINS:
        print(f"\n🚀 Processing Domain: {dom}")
        train_loader, _ = get_loaders(dom)

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
    # PHASE 2: Analyze Matrix D
    # -------------------------------------------
    print(f"\n[ANALYSIS] Computing Matrix D with Temperature Scaling (T={TEMPERATURE})...")

    # 1. Collect All Test Data
    all_imgs = []
    labels = []
    for d in DOMAINS:
        _, tl = get_loaders(d)
        if tl is None: continue
        for x, _ in tl:
            all_imgs.append(x)
            labels.extend([d] * x.size(0))

    X_test_all = torch.cat(all_imgs)  # CPU to save GPU memory
    N = len(X_test_all)
    K = len(DOMAINS)

    # 2. Compute Raw Log Probabilities
    Raw_LogP = np.zeros((N, K))

    with torch.no_grad():
        for k, dom_model in enumerate(DOMAINS):
            print(f"   -> Evaluating Model: {dom_model}")
            ae = ae_models[dom_model]
            fm = fm_models[dom_model]
            mu, std = stats_dict[dom_model]

            # Batch processing for evaluation
            bs = 32
            scores = []
            for i in range(0, N, bs):
                x_batch = X_test_all[i:i + bs].to(device)

                # Encode & Normalize using specific model stats
                _, z = ae(x_batch)
                z_norm = (z - mu) / std

                # Compute Exact Log Likelihood
                ll = compute_ll(fm, z_norm, LATENT_DIM)
                scores.append(ll.cpu().numpy())

            Raw_LogP[:, k] = np.concatenate(scores).flatten()

    # 3. Normalization + Scaling with TEMPERATURE
    D = np.zeros((N, K))
    print(f"   -> Applying Softmax with T={TEMPERATURE}")

    for i in range(K):
        col_log_p = Raw_LogP[:, i]

        # Stability shift
        col_max = np.max(col_log_p)

        # Apply Temperature Scaling BEFORE exp
        # P ~ exp(logP / T)
        exp_p = np.exp((col_log_p - col_max) / TEMPERATURE)

        col_sum = np.sum(exp_p)
        if col_sum == 0: col_sum = 1e-10

        col_normalized = exp_p / col_sum
        D[:, i] = col_normalized * N

    # Filename includes Temperature for clarity
    filename = f"D_Matrix_LatentFlow_T{int(TEMPERATURE)}.npy"
    save_path = os.path.join(paths['results'], filename)
    np.save(save_path, D)
    print(f"   -> Matrix saved to {save_path}")

    # 4. Plotting
    print(f"   -> Generating Plots...")
    D_plot = D / N

    fig, axes = plt.subplots(K, K, figsize=(15, 15))
    color_map = {d: plt.cm.tab10(idx) for idx, d in enumerate(DOMAINS)}

    for i in range(K):  # Row (Target)
        for j in range(K):  # Col (Source)
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

    plt.suptitle(f"D Matrix Analysis - LatentFlow (T={TEMPERATURE})", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(paths['plots'], f"LatentFlow_Analysis_T{int(TEMPERATURE)}.png"))
    print("✅ Done.")


if __name__ == "__main__":
    run_experiment()