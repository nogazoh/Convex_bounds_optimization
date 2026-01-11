import torch
from torch.utils.data import DataLoader, TensorDataset, Subset
import ssl
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch import nn, optim
from torch.distributions.normal import Normal
import torch.nn.functional as F
from torchvision import models, transforms
import argparse
import itertools

# Import umap
try:
    import umap
except ImportError:
    pass

import data as Data

# Bypass SSL & Matplotlib Backend
ssl._create_default_https_context = ssl._create_unverified_context
plt.switch_backend('agg')

# --- Device & Config ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

# ==========================================
# --- GLOBAL PATHS ---
# ==========================================
N_JOBS = 1
BASE_EXP_DIR = "/data/nogaz/Convex_bounds_optimization/VAE_Renyi_Experiments"
SHARED_OLD_MODELS = os.path.join(BASE_EXP_DIR, "Models")

OFFICE_31_DOMAINS = ['amazon', 'dslr', 'webcam']
UMAP_DIM = 64


def configure_per_process_threads():
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    try:
        torch.set_num_threads(1)
    except Exception:
        pass


# ==========================================
# PART 1: RAW FEATURE EXTRACTION
# ==========================================
def get_classifier_path(domain):
    if domain in OFFICE_31_DOMAINS:
        return f"/data/nogaz/Convex_bounds_optimization/classifiers_new/{domain}_classifier.pt"
    return f"/data/nogaz/Convex_bounds_optimization/classifiers/{domain}_224.pt"


def extract_raw_features(domain, mode="GAP"):
    save_dir = os.path.join(BASE_EXP_DIR, f"Raw_Features_{mode}")
    os.makedirs(save_dir, exist_ok=True)

    save_path_train = os.path.join(save_dir, f"{domain}_train.pt")
    save_path_test = os.path.join(save_dir, f"{domain}_test.pt")

    if os.path.exists(save_path_train) and os.path.exists(save_path_test): return

    print(f"\n[EXTRACTION] Extracting {mode} features for '{domain}'...")
    num_classes = 31
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    try:
        state = torch.load(get_classifier_path(domain), map_location=device)
        model.load_state_dict(state)
    except:
        return

    if mode == "GAP":
        model.fc = nn.Identity()
    else:
        model.avgpool = nn.Identity()
        model.fc = nn.Identity()

    model = model.to(device).eval()
    train_loader, test_loader, config = Data.get_data_loaders(domain)

    target_size = config['size']
    clean_transform = transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if isinstance(train_loader.dataset, Subset):
        if hasattr(train_loader.dataset.dataset, 'transform'):
            train_loader.dataset.dataset.transform = clean_transform
    elif hasattr(train_loader.dataset, 'transform'):
        train_loader.dataset.transform = clean_transform

    def run_extraction(loader, path):
        feats_list, labels_list = [], []
        with torch.no_grad():
            for imgs, labels in loader:
                imgs = imgs.to(device)
                out = model(imgs)
                if mode == "Layer4": out = torch.flatten(out, 1)
                feats_list.append(out.cpu())
                labels_list.append(labels.cpu())
        torch.save((torch.cat(feats_list, dim=0), torch.cat(labels_list, dim=0)), path)

    run_extraction(train_loader, save_path_train)
    run_extraction(test_loader, save_path_test)


# ==========================================
# PART 2: UMAP
# ==========================================
def run_umap_pipeline(domain, mode, use_umap, paths):
    if not use_umap: return

    raw_dir = os.path.join(BASE_EXP_DIR, f"Raw_Features_{mode}")
    out_dir = paths['features']

    save_path_train = os.path.join(out_dir, f"{domain}_train.pt")
    save_path_test = os.path.join(out_dir, f"{domain}_test.pt")

    if os.path.exists(save_path_train) and os.path.exists(save_path_test): return

    print(f"\n[UMAP] Running UMAP on {domain} ({mode})...")
    train_x, train_y = torch.load(os.path.join(raw_dir, f"{domain}_train.pt"), weights_only=True)
    test_x, test_y = torch.load(os.path.join(raw_dir, f"{domain}_test.pt"), weights_only=True)

    reducer = umap.UMAP(n_components=UMAP_DIM, random_state=42, n_jobs=1)
    train_reduced = reducer.fit_transform(train_x.numpy())
    test_reduced = reducer.transform(test_x.numpy())

    torch.save((torch.tensor(train_reduced), train_y), save_path_train)
    torch.save((torch.tensor(test_reduced), test_y), save_path_test)
    print(f"   -> Saved UMAP features to {out_dir}")


def run_combined_tsne(domains_list, mode, use_umap, paths):
    # Only run once per folder (independent of model type)
    if os.path.exists(os.path.join(paths['plots'], f"Combined_tSNE_{'UMAP' if use_umap else 'NoUMAP'}_{mode}.png")):
        return

    print(f"\n[VISUALIZATION] Generating t-SNE for {mode}...")
    if use_umap:
        data_dir = paths['features']
    else:
        data_dir = os.path.join(BASE_EXP_DIR, f"Raw_Features_{mode}")

    all_feats = []
    domain_labels = []

    for dom in domains_list:
        f_path = os.path.join(data_dir, f"{dom}_test.pt")
        if not os.path.exists(f_path): continue
        feats, _ = torch.load(f_path, weights_only=True)

        if not use_umap and len(feats) > 500:
            idx = np.random.choice(len(feats), 500, replace=False)
            feats = feats[idx]

        all_feats.append(feats.numpy())
        domain_labels.extend([dom] * len(feats))

    if not all_feats: return
    X = np.concatenate(all_feats, axis=0)

    tsne = TSNE(n_components=2, perplexity=30, init='pca', learning_rate='auto', random_state=42)
    X_emb = tsne.fit_transform(X)

    plt.figure(figsize=(10, 8))
    unique_doms = list(set(domain_labels))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_doms)))

    for i, dom in enumerate(unique_doms):
        mask = np.array(domain_labels) == dom
        plt.scatter(X_emb[mask, 0], X_emb[mask, 1], label=dom, alpha=0.6, s=20, color=colors[i])

    suffix = "UMAP" if use_umap else "NoUMAP"
    plt.title(f"Combined t-SNE ({suffix}) - Source: {mode}")
    plt.legend()
    plt.grid(True, alpha=0.3)

    save_name = f"Combined_tSNE_{suffix}_{mode}.png"
    plt.savefig(os.path.join(paths['plots'], save_name))
    plt.close()
    print(f"   -> Plot saved to {paths['plots']}")


# ==========================================
# PART 3: VR MODEL (Supports VAE and VRS)
# ==========================================
class vr_model(nn.Module):
    def __init__(self, input_dim, alpha_pos, alpha_neg):
        super(vr_model, self).__init__()
        self.input_dim = input_dim
        hidden_dim = 512 if input_dim > 1000 else 256
        self.fc1, self.fc2 = nn.Linear(input_dim, hidden_dim), nn.Linear(hidden_dim, hidden_dim)
        self.fc31, self.fc32 = nn.Linear(hidden_dim, 50), nn.Linear(hidden_dim, 50)
        self.fc4, self.fc5 = nn.Linear(50, hidden_dim), nn.Linear(hidden_dim, hidden_dim)
        self.fc6 = nn.Linear(hidden_dim, input_dim)
        self.alpha_pos, self.alpha_neg = alpha_pos, alpha_neg

    def encode(self, x):
        h = torch.tanh(self.fc2(torch.tanh(self.fc1(x))))
        return self.fc31(h), self.fc32(h)

    def reparameterize(self, mu, logstd):
        return mu + torch.randn_like(logstd) * torch.exp(logstd)

    def decode(self, z):
        h = torch.tanh(self.fc5(torch.tanh(self.fc4(z))))
        return torch.sigmoid(self.fc6(h))

    def forward(self, x):
        mu, logstd = self.encode(x.view(-1, self.input_dim))
        z = self.reparameterize(mu, logstd)
        return self.decode(z), mu, logstd

    def compute_log_probabitility_bernoulli(self, obs, p):
        p = torch.clamp(p, 1e-10, 1.0 - 1e-10)
        return torch.sum(p * torch.log(obs + 1e-10) + (1 - p) * torch.log(1 - obs + 1e-10), 1)

    def compute_loss_for_batch(self, data, model, model_type, K, testing_mode=False):
        B = data.shape[0]
        x_rep = data.view(B, -1).repeat_interleave(K, dim=0)
        mu, logstd = model.encode(x_rep)
        z = model.reparameterize(mu, logstd)

        log_q = torch.mean(Normal(mu, torch.exp(logstd)).log_prob(z), 1)
        log_p_z = torch.mean(Normal(torch.zeros_like(z), torch.ones_like(z)).log_prob(z), 1)
        x_hat = model.decode(z)
        log_p_x = model.compute_log_probabitility_bernoulli(x_hat, x_rep)

        # VRS Logic vs VAE Logic
        alpha = model.alpha_pos if model_type == 'vr' else (model.alpha_neg if model_type == 'vrlu' else 1)
        log_w = (log_p_z + log_p_x - log_q).view(-1, K) * (1 - alpha)

        if model_type == 'vae':
            BCE = F.binary_cross_entropy(x_hat, x_rep, reduction='sum') / K
            KLD = 0.5 * torch.sum(torch.exp(logstd) - logstd - 1 + mu.pow(2)) / K
            return BCE + KLD, 0, 0, 0

        # Renyi Loss (VRS)
        loss = -torch.mean(log_w)
        return loss, 0, 0, 0


# ==========================================
# PART 4: TRAIN & ANALYZE
# ==========================================

def load_dataset_for_vae(domain, mode, use_umap, paths):
    if use_umap:
        dir_path = paths['features']
    else:
        dir_path = os.path.join(BASE_EXP_DIR, f"Raw_Features_{mode}")

    train_x, train_y = torch.load(os.path.join(dir_path, f"{domain}_train.pt"), weights_only=True)
    test_x, test_y = torch.load(os.path.join(dir_path, f"{domain}_test.pt"), weights_only=True)

    min_v, max_v = train_x.min(), train_x.max()
    train_x = (train_x - min_v) / (max_v - min_v + 1e-6)
    test_x = (test_x - min_v) / (max_v - min_v + 1e-6)

    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=32, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=32, shuffle=False)

    return train_loader, test_loader, (min_v, max_v)


def train_vae(domain, mode, use_umap, seed, paths, input_dim, model_arch="VAE"):
    # Determine settings based on Model Architecture
    if model_arch == "VRS":
        alpha_pos, alpha_neg = 0.5, -0.5
        loss_type = 'vr'  # Triggers Renyi Loss
    else:  # VAE
        alpha_pos, alpha_neg = 1, -1
        loss_type = 'vae'  # Triggers ELBO

    # Distinct filename for each model type
    model_name = f"vae_{mode}_{domain}_{model_arch}_seed{seed}.pt"
    model_path = os.path.join(paths['models'], model_name)

    if os.path.exists(model_path): return

    print(f"   [TRAIN] {model_arch} on {domain} ({mode}) Dim={input_dim}...")
    train_loader, test_loader, _ = load_dataset_for_vae(domain, mode, use_umap, paths)

    model = vr_model(input_dim=input_dim, alpha_pos=alpha_pos, alpha_neg=alpha_neg).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(101):
        model.train()
        for data, _ in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            # Pass correct loss type (vae or vr)
            loss, _, _, _ = model.compute_loss_for_batch(data, model, loss_type, K=5)
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), model_path)


def analyze_matrix_D(domains, mode, use_umap, seed, paths, input_dim, model_arch="VAE"):
    suffix = "UMAP" if use_umap else "NoUMAP"

    # Distinct filename for Matrix D
    filename = f"D_Matrix_{mode}_{model_arch}.npy"
    save_path = os.path.join(paths['root'], filename)

    print(f"\n[ANALYSIS] Computing Matrix D for {model_arch} ({suffix})...")
    print(f"           Output: {save_path}")

    models_dict = {}
    stats_dict = {}

    if use_umap:
        data_dir = paths['features']
    else:
        data_dir = os.path.join(BASE_EXP_DIR, f"Raw_Features_{mode}")

    # 1. Load Models
    for dom in domains:
        tr_x, _ = torch.load(os.path.join(data_dir, f"{dom}_train.pt"), weights_only=True)
        stats_dict[dom] = (tr_x.min(), tr_x.max())

        model_name = f"vae_{mode}_{dom}_{model_arch}_seed{seed}.pt"
        m_path = os.path.join(paths['models'], model_name)

        # Fallback for old VAE models only (Legacy support)
        if not os.path.exists(m_path) and not use_umap and model_arch == "VAE":
            print("[INFO] Fallback to legacy VAE model")
            m_path = os.path.join(SHARED_OLD_MODELS, f"vae_1_-1_{dom}_seed1.pt")

        if not os.path.exists(m_path):
            print(f"   [WARN] Model not found for {dom}: {m_path}")
            continue

        # Init model with correct alphas (though eval doesn't use alpha, good for consistency)
        ap, an = (0.5, -0.5) if model_arch == "VRS" else (1, -1)
        model = vr_model(input_dim=input_dim, alpha_pos=ap, alpha_neg=an).to(device)
        model.load_state_dict(torch.load(m_path, map_location=device))
        model.eval()
        models_dict[dom] = model

    # 2. Collect Data
    all_data = []
    labels = []
    for dom in domains:
        te_x, _ = torch.load(os.path.join(data_dir, f"{dom}_test.pt"), weights_only=True)
        all_data.append(te_x)
        labels.extend([dom] * len(te_x))

    X_all = torch.cat(all_data, dim=0).to(device)
    N = len(X_all)
    K = len(domains)

    # 3. Compute Raw P(x|k)
    Raw_LogP = np.zeros((N, K))

    with torch.no_grad():
        for i, dom in enumerate(domains):
            model = models_dict[dom]
            min_v, max_v = stats_dict[dom]
            x_in = (X_all - min_v.to(device)) / (max_v.to(device) - min_v.to(device) + 1e-6)

            x_hat, _, _ = model(x_in)
            log_p = model.compute_log_probabitility_bernoulli(x_hat, x_in).cpu().numpy()
            Raw_LogP[:, i] = log_p

    # 4. Normalization + Scaling
    D = np.zeros((N, K))
    for i in range(K):
        col_log_p = Raw_LogP[:, i]
        col_max = np.max(col_log_p)
        exp_p = np.exp(col_log_p - col_max)
        col_sum = np.sum(exp_p)
        col_normalized = exp_p / col_sum
        D[:, i] = col_normalized * N

    np.save(save_path, D)
    print(f"   -> Matrix saved.")

    # Plotting
    D_plot = D / N
    fig, axes = plt.subplots(K, K, figsize=(15, 15))
    unique_doms = list(set(labels))
    color_map = {d: plt.cm.tab10(idx) for idx, d in enumerate(domains)}

    for i in range(K):
        for j in range(K):
            ax = axes[i, j]
            if i == j:
                ax.set_title(f"Model: {domains[i]}")
                for d in domains:
                    idxs = [z for z, l in enumerate(labels) if l == d]
                    ax.hist(D_plot[idxs, i], bins=30, alpha=0.5, label=d, density=True, color=color_map[d])
                if i == 0: ax.legend()
            else:
                scatter_colors = [color_map[l] for l in labels]
                ax.scatter(D_plot[:, j], D_plot[:, i], c=scatter_colors, s=5, alpha=0.6)
                ax.set_xlabel(f"P({domains[j]})")
                ax.set_ylabel(f"P({domains[i]})")
                ax.set_xlim([0, 1.05])
                ax.set_ylim([0, 1.05])

    plt.suptitle(f"D Matrix - {model_arch} - {mode} ({suffix})", fontsize=16)
    plt.tight_layout()
    plot_filename = f"Matrix_Scatter_{mode}_{model_arch}.png"
    plt.savefig(os.path.join(paths['plots'], plot_filename))

    return save_path


# ==========================================
# --- EXPERIMENT RUNNER ---
# ==========================================

def run_experiment(use_umap, mode, model_arch):
    configure_per_process_threads()
    target_domains = OFFICE_31_DOMAINS

    # 1. Output Paths
    subfolder_name = f"{mode}_{'UMAP' if use_umap else 'NoUMAP'}"
    exp_root_dir = os.path.join(BASE_EXP_DIR, subfolder_name)

    paths = {
        'root': exp_root_dir,
        'models': os.path.join(exp_root_dir, "models"),
        'plots': os.path.join(exp_root_dir, "plots"),
        'features': os.path.join(exp_root_dir, "features")
    }
    for p in paths.values(): os.makedirs(p, exist_ok=True)

    print("\n" + "=" * 60)
    print(f"STARTING EXPERIMENT: {subfolder_name} | MODEL: {model_arch}")
    print("=" * 60)

    # 2. Extract Features (Shared)
    for dom in target_domains:
        extract_raw_features(dom, mode=mode)

    # 3. Run UMAP (Specific)
    for dom in target_domains:
        run_umap_pipeline(dom, mode=mode, use_umap=use_umap, paths=paths)

    # 4. Detect Dimension
    if use_umap:
        sample_path = os.path.join(paths['features'], f"{target_domains[0]}_train.pt")
        feats, _ = torch.load(sample_path, weights_only=True)
        input_dim = feats.shape[1]
    else:
        raw_path = os.path.join(BASE_EXP_DIR, f"Raw_Features_{mode}", f"{target_domains[0]}_train.pt")
        feats, _ = torch.load(raw_path, weights_only=True)
        input_dim = feats.shape[1]

    print(f"   -> Input Dimension: {input_dim}")

    # 5. Visualize
    run_combined_tsne(target_domains, mode=mode, use_umap=use_umap, paths=paths)

    # 6. Train Models (VAE or VRS)
    for dom in target_domains:
        train_vae(dom, mode=mode, use_umap=use_umap, seed=1, paths=paths, input_dim=input_dim, model_arch=model_arch)

    # 7. Analyze & Generate D
    final_d_path = analyze_matrix_D(domains=target_domains, mode=mode, use_umap=use_umap, seed=1,
                                    paths=paths, input_dim=input_dim, model_arch=model_arch)

    # 8. Inspection
    if os.path.exists(final_d_path):
        D = np.load(final_d_path)
        row_sums = np.sum(D, axis=1)
        print(f"   [INSPECT] {model_arch} Avg Row Sum: {np.mean(row_sums):.4f}")


# ==========================================
# --- MAIN LOOP ---
# ==========================================

if __name__ == "__main__":
    umap_options = [True, False]
    mode_options = ["GAP", "Layer4"]
    model_options = ["VRS"]  # Added Model Options "VAE",

    print("🚀 BATCH EXECUTION STARTED: All permutations (Mode x UMAP x Model)...")

    for mode in mode_options:
        for use_umap in umap_options:
            for model_arch in model_options:
                try:
                    run_experiment(use_umap=use_umap, mode=mode, model_arch=model_arch)
                except Exception as e:
                    print(f"\n❌ ERROR in {mode} | UMAP={use_umap} | Model={model_arch}")
                    print(e)
                    continue

    print("\n✅ BATCH EXECUTION COMPLETE.")