import torch
from torch.utils.data import DataLoader, TensorDataset, Subset
from joblib import Parallel, delayed
import ssl
import os
import numpy as np
from torch import nn, optim
from torch.distributions.normal import Normal
from torch.distributions.multinomial import Multinomial
import datetime
import torch.nn.functional as F
from torchvision import models, transforms

# Import your data loader file
import data as Data

# Bypass SSL verification for dataset downloads if needed
ssl._create_default_https_context = ssl._create_unverified_context

# --- Device Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"VAE running on: {device}")

# Keep it 1 for stability when using GPU feature extraction
N_JOBS = 1


def configure_per_process_threads():
    """Limit threads per process to prevent CPU oversubscription."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass


# ==========================================
# PART 1: FEATURE EXTRACTION LOGIC
# ==========================================

def extract_features_if_needed(domain):
    """
    Extracts features using the _224.pt classifiers.
    Saves results with '224' in the filename to avoid overwriting.
    """
    # New save directory to include 224 naming convention
    save_dir = f"./data_features_{domain}_224"
    train_save_path = os.path.join(save_dir, "train_224.pt")
    test_save_path = os.path.join(save_dir, "test_224.pt")

    if os.path.exists(train_save_path) and os.path.exists(test_save_path):
        return

    print(f"\n[EXTRACTION] Features (224) for '{domain}' not found. Extracting...")
    os.makedirs(save_dir, exist_ok=True)

    # --- 1. Detect Classes ---
    OFFICE_31_DOMAINS = ['amazon', 'dslr', 'webcam']
    OFFICE_HOME_DOMAINS = ['Art', 'Clipart', 'Product', 'Real World']

    if domain in OFFICE_31_DOMAINS:
        num_classes = 31
    elif domain in OFFICE_HOME_DOMAINS:
        num_classes = 65
    else:
        num_classes = 10  # Digits

    # Load ResNet architecture
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # 2. Updated Path to your specific classifier directory
    model_path = f"/data/nogaz/Convex_bounds_optimization/classifiers/{domain}_224.pt"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"[CRITICAL] Classifier model (224) not found at: {model_path}")

    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"   -> Loaded 224 weights from {model_path}")
    except Exception as e:
        print(f"[ERROR] Failed to load weights for {domain}: {e}")
        raise e

    model.fc = nn.Identity()
    model = model.to(device)
    model.eval()

    # 3. Get Data Loaders
    train_loader, test_loader, config = Data.get_data_loaders(domain)

    target_size = config['size']
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    clean_transform = transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor(),
        normalize
    ])

    # Swap to clean transform for stable feature extraction
    if isinstance(train_loader.dataset, Subset):
        train_loader.dataset.dataset.transform = clean_transform

    def process_and_save(loader, save_path):
        features_list = []
        labels_list = []
        if loader is None: return
        with torch.no_grad():
            for imgs, labels in loader:
                imgs = imgs.to(device)
                feats = model(imgs)
                features_list.append(feats.cpu())
                labels_list.append(labels.cpu())
        torch.save((torch.cat(features_list, dim=0), torch.cat(labels_list, dim=0)), save_path)

    process_and_save(train_loader, train_save_path)
    process_and_save(test_loader, test_save_path)
    print(f"[EXTRACTION] Completed 224 features for {domain}.\n")


def load_feature_dataset(domain, batch_size=32):
    extract_features_if_needed(domain)
    feature_dir = f"./data_features_{domain}_224"
    train_features, train_labels = torch.load(os.path.join(feature_dir, "train_224.pt"))
    test_features, test_labels = torch.load(os.path.join(feature_dir, "test_224.pt"))

    # Normalization [0, 1]
    min_val, max_val = train_features.min(), train_features.max()
    train_features = torch.clamp((train_features - min_val) / (max_val - min_val + 1e-6), 0.0, 1.0)
    test_features = torch.clamp((test_features - min_val) / (max_val - min_val + 1e-6), 0.0, 1.0)

    train_loader = DataLoader(TensorDataset(train_features, train_labels), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_features, test_labels), batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, train_features.shape[1]


# ==========================================
# PART 2: VAE/VRS MODEL & MATH (UNCHANGED)
# ==========================================

def elbo(model, x, z, mu, logstd, gamma=1):
    x_hat = model.decode(z)
    BCE = F.binary_cross_entropy(x_hat, x, reduction='sum')
    KLD = 0.5 * torch.sum(torch.exp(logstd) - logstd - 1 + mu.pow(2))
    return BCE + gamma * KLD


def renyi_bound(method, model, x, z, mu, logstd, alpha, K, testing_mode=False):
    log_q = model.compute_log_probabitility_gaussian(z, mu, logstd)
    log_p_z = model.compute_log_probabitility_gaussian(z, torch.zeros_like(z), torch.zeros_like(z))
    x_hat = model.decode(z)
    log_p = model.compute_log_probabitility_bernoulli(x_hat, x)
    log_w_matrix = (log_p_z + log_p - log_q).view(-1, K) * (1 - alpha)
    if alpha == 1: return elbo(model, x, z, mu, logstd)
    if method == 'vr':
        return compute_MC_approximation(log_w_matrix, alpha, testing_mode)
    elif method == 'vr_ub':
        return compute_approximation_for_negative_alpha(log_w_matrix, alpha)
    return 0


def compute_MC_approximation(log_w_matrix, alpha, testing_mode=False):
    log_w_minus_max = log_w_matrix - torch.max(log_w_matrix, 1, keepdim=True)[0]
    ws_matrix = torch.exp(log_w_minus_max)
    ws_norm = ws_matrix / torch.sum(ws_matrix, 1, keepdim=True)
    if not testing_mode:
        ws_sum_per_datapoint = log_w_matrix.gather(1, Multinomial(1, ws_norm).sample().argmax(1, keepdim=True))
    else:
        ws_sum_per_datapoint = torch.sum(log_w_matrix * ws_norm, 1)
    return -torch.sum(ws_sum_per_datapoint / (1 - alpha))


def compute_approximation_for_negative_alpha(log_w_matrix, alpha):
    norm_log_w_matrix = log_w_matrix.view(log_w_matrix.size(0), -1)
    min_v, max_v = norm_log_w_matrix.min(1, keepdim=True)[0], norm_log_w_matrix.max(1, keepdim=True)[0]
    norm_w_matrix = torch.exp((norm_log_w_matrix - min_v) / (max_v - min_v + 1e-6))
    approx = (norm_w_matrix - 1) * max_v + min_v
    ws_norm = approx / torch.sum(approx, 1, keepdim=True)
    return -torch.sum(torch.sum(approx * ws_norm, 1) / (1 - alpha))


def renyi_bound_sandwich(model, x, z, mu, logstd, alpha_pos, alpha_neg, K, testing_mode=False):
    return (renyi_bound('vr', model, x, z, mu, logstd, alpha_pos, K, testing_mode) +
            renyi_bound('vr_ub', model, x, z, mu, logstd, alpha_neg, K, testing_mode)) / 2


class vr_model(nn.Module):
    def __init__(self, input_dim, alpha_pos, alpha_neg):
        super(vr_model, self).__init__()
        self.input_dim = input_dim
        hidden_dim = 200
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

    def MSE_reconstruction_error(self, x_hat, x):
        return torch.sum(torch.mean(torch.pow(x - x_hat, 2), axis=1))

    def CE_reconstruction_error(self, x_hat, x):
        epsilon = 1e-10
        loss = -torch.sum(x * torch.log(x_hat + epsilon))
        return loss / x_hat.size(dim=0)

    def compute_log_probabitility_gaussian(self, obs, mu, logstd):
        return torch.mean(Normal(mu, torch.exp(logstd)).log_prob(obs), 1)

    def compute_log_probabitility_bernoulli(self, obs, p):
        p = torch.clamp(p, 1e-10, 1.0 - 1e-10)
        return torch.sum(p * torch.log(obs + 1e-10) + (1 - p) * torch.log(1 - obs + 1e-10), 1)

    def compute_loss_for_batch(self, data, model, model_type, K, testing_mode=False):
        B = data.shape[0]
        x_rep = data.view(B, -1).repeat_interleave(K, dim=0)
        mu, logstd = model.encode(x_rep)
        z = model.reparameterize(mu, logstd)
        if model_type == "vae":
            loss = elbo(model, x_rep, z, mu, logstd)
        elif model_type == "vr":
            loss = renyi_bound("vr", model, x_rep, z, mu, logstd, model.alpha_pos, K, testing_mode)
        elif model_type == "vrlu":
            loss = renyi_bound("vr_ub", model, x_rep, z, mu, logstd, model.alpha_neg, K, testing_mode)
        else:
            loss = renyi_bound_sandwich(model, x_rep, z, mu, logstd, model.alpha_pos, model.alpha_neg, K, testing_mode)
        x_hat = model.decode(z)
        return loss, torch.sum(torch.mean(torch.pow(x_rep - x_hat, 2), 1)), 0, torch.sum(
            torch.mean(model.compute_log_probabitility_bernoulli(x_hat, x_rep).view(B, K), 1))


# ==========================================
# PART 3: TRAINING LOOPS
# ==========================================

def train(model, optimizer, epoch, train_loader, model_type, losses):
    model.train()
    total_loss = 0
    for data, _ in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        loss, _, _, _ = model.compute_loss_for_batch(data, model, model_type, K=20)
        loss.backward()
        total_loss += loss.item()
        optimizer.step()
    losses.append(total_loss / len(train_loader.dataset))
    return losses


def test(model, test_loader, model_type, losses):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            loss, _, _, _ = model.compute_loss_for_batch(data, model, model_type, K=20, testing_mode=True)
            total_loss += loss.item()
    losses.append(total_loss / len(test_loader.dataset))
    return losses


def run(model_type, alpha_pos, alpha_neg, data_name, seed):
    # Directory naming to avoid overwriting and clarify 224 resolution
    path = f"./models_{data_name}_seed{seed}_224_features"
    os.makedirs(path, exist_ok=True)
    filename_prefix = f"{model_type}_{alpha_pos}_{alpha_neg}_{data_name}_seed{seed}_224"
    model_file_path = os.path.join(path, f"{filename_prefix}_model.pt")

    if os.path.exists(model_file_path):
        return

    train_loader, test_loader, input_dim = load_feature_dataset(data_name)
    model = vr_model(input_dim, alpha_pos, alpha_neg).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses, test_losses = [], []
    for epoch in range(201):
        train_losses = train(model, optimizer, epoch, train_loader, model_type, train_losses)
        test_losses = test(model, test_loader, model_type, test_losses)
        if len(test_losses) >= 3 and test_losses[-1] >= test_losses[-2] >= test_losses[-3]: break

    torch.save(model.state_dict(), model_file_path)


def run_all():
    configure_per_process_threads()

    # Combined Domains: Office-31 + Office-Home
    domains = ['Art', 'Clipart', 'Product', 'Real World']
    seeds = [1]
    tasks = [(domain, seed) for domain in domains for seed in seeds]

    def wrapped_run(domain, seed):
        torch.manual_seed(seed)
        run('vae', alpha_pos=1, alpha_neg=-1, data_name=domain, seed=seed)
        run('vrs', alpha_pos=2, alpha_neg=-2, data_name=domain, seed=seed)
        run('vrs', alpha_pos=0.5, alpha_neg=-0.5, data_name=domain, seed=seed)

    Parallel(n_jobs=N_JOBS, backend="loky", prefer="processes")(
        delayed(wrapped_run)(domain, seed) for domain, seed in tasks
    )


if __name__ == "__main__":
    configure_per_process_threads()
    run_all()