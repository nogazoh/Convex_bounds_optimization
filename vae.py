import torch
from torch.utils.data import DataLoader, TensorDataset
from joblib import Parallel, delayed
import ssl
import os
import numpy as np
from torch import nn, optim
from torch.distributions.normal import Normal
from torch.distributions.multinomial import Multinomial
import matplotlib.pyplot as plt
import datetime
import torch.nn.functional as F
from torchvision import models

# Assuming 'data.py' is in the same directory for loading raw images during extraction
import data as Data

# Bypass SSL verification for dataset downloads if needed
ssl._create_default_https_context = ssl._create_unverified_context

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device = ", device)

# Determine number of parallel jobs
N_JOBS = 2


def configure_per_process_threads():
    """
    Limit threads per process to prevent CPU oversubscription during parallel runs.
    """
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
    Checks if feature files exist. If not, loads the pre-trained classifier,
    extracts features (penultimate layer), and saves them to disk.
    """
    save_dir = f"./data_features/{domain}"
    train_save_path = os.path.join(save_dir, "train.pt")
    test_save_path = os.path.join(save_dir, "test.pt")

    # If files already exist, we skip extraction
    if os.path.exists(train_save_path) and os.path.exists(test_save_path):
        return

    print(f"\n[EXTRACTION] Features for '{domain}' not found. Extracting from classifier...")
    os.makedirs(save_dir, exist_ok=True)

    # 1. Define Model Structure (ResNet50 with 65 classes for Office-Home)
    OFFICE_HOME_CLASSES = 65
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, OFFICE_HOME_CLASSES)

    # 2. Load Your Pre-trained Weights
    # Based on your screenshot, the folder is "classifiers_new"
    model_path = f"./classifiers_new/{domain}_classifier.pt"

    if not os.path.exists(model_path):
        # Fallback: try looking in current dir just in case
        if os.path.exists(f"{domain}_classifier.pt"):
            model_path = f"{domain}_classifier.pt"
        else:
            raise FileNotFoundError(f"[CRITICAL] Classifier model not found at: {model_path}\n"
                                    f"Cannot extract features without the source model.")

    # Load weights
    try:
        # map_location ensures it loads on CPU if CUDA is not available
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        print(f"[ERROR] Failed to load weights for {domain}: {e}")
        raise e

    # 3. "Cut the Head": Replace final layer with Identity to get 2048-dim vectors
    model.fc = nn.Identity()
    model = model.to(device)
    model.eval()

    # 4. Helper to process a loader and save
    def process_and_save(loader, save_path):
        features_list = []
        labels_list = []

        with torch.no_grad():
            for batch_idx, (imgs, labels) in enumerate(loader):
                imgs = imgs.to(device)

                # Pass through ResNet backbone
                feats = model(imgs)

                features_list.append(feats.cpu())
                labels_list.append(labels.cpu())

        # Concatenate all batches
        all_feats = torch.cat(features_list, dim=0)
        all_labels = torch.cat(labels_list, dim=0)

        # Save to disk
        torch.save((all_feats, all_labels), save_path)
        print(f"   -> Saved to {save_path} (Shape: {all_feats.shape})")

    # 5. Load Raw Images using data.py and Process
    print(f"   -> Loading raw images for {domain}...")
    # Note: data.py usually returns (train, test, val) or (train, test). Adjust unpacking if needed.
    train_loader, test_loader, _ = Data.get_data_loaders(domain)

    process_and_save(train_loader, train_save_path)
    process_and_save(test_loader, test_save_path)
    print(f"[EXTRACTION] Completed for {domain}.\n")


def load_feature_dataset(domain, batch_size=32):
    """
    Loads pre-extracted ResNet features from .pt files.
    Normalizes features to [0, 1] for BCE loss compatibility.
    """
    # Ensure features exist before loading
    extract_features_if_needed(domain)

    feature_dir = f"./data_features/{domain}"
    train_path = os.path.join(feature_dir, "train.pt")
    test_path = os.path.join(feature_dir, "test.pt")

    # Load Tensors
    train_features, train_labels = torch.load(train_path)
    test_features, test_labels = torch.load(test_path)

    # --- NORMALIZATION [0, 1] ---
    # Essential because VAE output is Sigmoid (0-1) and Loss is BCE.
    min_val = train_features.min()
    max_val = train_features.max()

    # Normalize Train
    train_features = (train_features - min_val) / (max_val - min_val + 1e-6)

    # Normalize Test (using Train stats)
    test_features = (test_features - min_val) / (max_val - min_val + 1e-6)

    # *** FIX: CLAMP VALUES TO [0, 1] ***
    # This prevents the "RuntimeError: all elements of target should be between 0 and 1"
    # if test data has outliers outside the training min/max range.
    train_features = torch.clamp(train_features, 0.0, 1.0)
    test_features = torch.clamp(test_features, 0.0, 1.0)

    # Create DataLoaders
    train_dataset = TensorDataset(train_features, train_labels)
    test_dataset = TensorDataset(test_features, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Input dimension (should be 2048 for ResNet50)
    input_dim = train_features.shape[1]

    return train_loader, test_loader, input_dim


# ==========================================
# PART 2: VAE/VRS MODEL & MATH
# ==========================================

def elbo(model, x, z, mu, logstd, gamma=1):
    x_hat = model.decode(z)
    BCE = F.binary_cross_entropy(x_hat, x, reduction='sum')
    KLD = 0.5 * torch.sum(torch.exp(logstd) - logstd - 1 + mu.pow(2))
    loss = BCE + gamma * KLD
    return loss


def renyi_bound(method, model, x, z, mu, logstd, alpha, K, testing_mode=False):
    log_q = model.compute_log_probabitility_gaussian(z, mu, logstd)
    log_p_z = model.compute_log_probabitility_gaussian(z, torch.zeros_like(z), torch.zeros_like(z))
    x_hat = model.decode(z)
    log_p = model.compute_log_probabitility_bernoulli(x_hat, x)

    log_w_matrix = (log_p_z + log_p - log_q).view(-1, K) * (1 - alpha)

    loss = 0
    if alpha == 1:
        loss = elbo(model, x, z, mu, logstd)
    if method == 'vr':
        loss = compute_MC_approximation(log_w_matrix, alpha, testing_mode)
    elif method == 'vr_ub':
        loss = compute_approximation_for_negative_alpha(log_w_matrix, alpha)
    else:
        print("Invalid value of alpha")

    return loss


def compute_MC_approximation(log_w_matrix, alpha, testing_mode=False):
    log_w_minus_max = log_w_matrix - torch.max(log_w_matrix, 1, keepdim=True)[0]
    ws_matrix = torch.exp(log_w_minus_max)
    ws_norm = ws_matrix / torch.sum(ws_matrix, 1, keepdim=True)

    if not testing_mode:
        sample_dist = Multinomial(1, ws_norm)
        ws_sum_per_datapoint = log_w_matrix.gather(1, sample_dist.sample().argmax(1, keepdim=True))
    else:
        ws_sum_per_datapoint = torch.sum(log_w_matrix * ws_norm, 1)

    if alpha == 1:
        return 0

    ws_sum_per_datapoint /= (1 - alpha)
    loss = -torch.sum(ws_sum_per_datapoint)
    return loss


def compute_approximation_for_negative_alpha(log_w_matrix, alpha):
    norm_log_w_matrix = log_w_matrix.view(log_w_matrix.size(0), -1)
    min_val = norm_log_w_matrix.min(1, keepdim=True)[0]
    max_val = norm_log_w_matrix.max(1, keepdim=True)[0]

    norm_log_w_matrix -= min_val
    norm_log_w_matrix /= max_val
    norm_w_matrix = torch.exp(norm_log_w_matrix)

    approx = norm_w_matrix - 1
    approx *= max_val
    approx += min_val

    ws_norm = approx / torch.sum(approx, 1, keepdim=True)
    ws_sum_per_datapoint = torch.sum(approx * ws_norm, 1)

    ws_sum_per_datapoint /= (1 - alpha)
    loss = -torch.sum(ws_sum_per_datapoint)
    return loss


def renyi_bound_sandwich(model, x, z, mu, logstd, alpha_pos, alpha_neg, K, testing_mode=False):
    loss_pos = renyi_bound('vr', model, x, z, mu, logstd, alpha_pos, K, testing_mode)
    loss_neg = renyi_bound('vr_ub', model, x, z, mu, logstd, alpha_neg, K, testing_mode)
    loss = (loss_neg + loss_pos) / 2
    return loss


class vr_model(nn.Module):
    def __init__(self, input_dim, alpha_pos, alpha_neg):
        super(vr_model, self).__init__()
        self.input_dim = input_dim
        # Features are ~2048 dims, so 512 is a good latent/hidden size
        hidden_dim = 200 if input_dim < 1000 else 512

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc31 = nn.Linear(hidden_dim, 50)  # mu
        self.fc32 = nn.Linear(hidden_dim, 50)  # logstd
        self.fc4 = nn.Linear(50, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.fc6 = nn.Linear(hidden_dim, input_dim)

        self.alpha_pos = alpha_pos
        self.alpha_neg = alpha_neg

    def encode(self, x):
        h1 = torch.tanh(self.fc1(x))
        h2 = torch.tanh(self.fc2(h1))
        return self.fc31(h2), self.fc32(h2)

    def reparameterize(self, mu, logstd):
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = torch.tanh(self.fc4(z))
        h4 = torch.tanh(self.fc5(h3))
        return torch.sigmoid(self.fc6(h4))

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

    def compute_log_probabitility_gaussian(self, obs, mu, logstd, axis=1):
        std = torch.exp(logstd)
        n = Normal(mu, std)
        res = torch.mean(n.log_prob(obs), axis)
        return res

    def compute_log_probabitility_bernoulli(self, obs, p, axis=1):
        epsilon = 1e-10
        # Ensure p is clamped just in case, though it comes from dataset
        p = torch.clamp(p, 0.0, 1.0)
        return torch.sum(p * torch.log(obs + epsilon) + (1 - p) * torch.log(1 - obs + epsilon), axis)

    def compute_loss_for_batch(self, data, model, model_type, K, testing_mode=False):
        B = data.shape[0]
        x = data.view(B, -1)
        x_repeated = x.repeat_interleave(K, dim=0)

        mu, logstd = model.encode(x_repeated)
        z = model.reparameterize(mu, logstd)

        loss = 0
        if model_type == "vae":
            loss = elbo(model, x_repeated, z, mu, logstd)
        elif model_type == "vr":
            loss = renyi_bound("vr", model, x_repeated, z, mu, logstd, model.alpha_pos, K, testing_mode)
        elif model_type == "vrlu":
            loss = renyi_bound("vr_ub", model, x_repeated, z, mu, logstd, model.alpha_neg, K, testing_mode)
        elif model_type == "vrs":
            loss = renyi_bound_sandwich(model, x_repeated, z, mu, logstd, model.alpha_pos, model.alpha_neg, K,
                                        testing_mode)

        x_hat = model.decode(z)
        recon_loss_MSE = model.MSE_reconstruction_error(x_hat, x_repeated)
        recon_loss_CE = model.CE_reconstruction_error(x_hat, x_repeated)
        log_p = model.compute_log_probabitility_bernoulli(x_hat, x_repeated)

        tmp1 = log_p.view(B, K)
        tmp2 = torch.mean(tmp1, 1)

        return loss, recon_loss_MSE, recon_loss_CE, torch.sum(tmp2)


# ==========================================
# PART 3: TRAINING LOOPS & MAIN
# ==========================================

def train(model, optimizer, epoch, train_loader, model_type, losses, recon_losses, log_p_vals):
    model.train()
    train_loss = 0
    train_recon_loss_MSE = 0
    train_recon_loss_CE = 0
    train_log_p = 0

    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()

        loss, recon_loss_MSE, recon_loss_CE, log_p = model.compute_loss_for_batch(data, model, model_type, K=20)
        loss.backward()

        train_loss += loss.item()
        train_recon_loss_MSE += recon_loss_MSE.item()
        train_recon_loss_CE += recon_loss_CE.item()
        train_log_p += log_p.item()

        optimizer.step()

    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))

    losses.append(train_loss / len(train_loader.dataset))
    recon_losses.append((train_recon_loss_MSE / len(train_loader.dataset),
                         train_recon_loss_CE / len(train_loader.dataset)))
    log_p_vals.append(train_log_p / len(train_loader.dataset))
    return losses, recon_losses, log_p_vals


def test(model, epoch, test_loader, model_type, losses, recon_losses, log_p_vals, img_shape):
    model.eval()
    test_loss = 0
    test_recon_loss_MSE = 0
    test_recon_loss_CE = 0
    test_log_p = 0

    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)

            loss, recon_loss_MSE, recon_loss_CE, log_p = model.compute_loss_for_batch(
                data, model, model_type, K=20, testing_mode=True
            )

            test_loss += loss.item()
            test_recon_loss_MSE += recon_loss_MSE.item()
            test_recon_loss_CE += recon_loss_CE.item()
            test_log_p += log_p.item()

    test_loss /= len(test_loader.dataset)
    test_recon_loss_MSE /= len(test_loader.dataset)
    test_recon_loss_CE /= len(test_loader.dataset)
    test_log_p /= len(test_loader.dataset)

    print('====> Test set loss: {:.4f}, alpha_pos: {:.4f}, alpha_neg: {:.4f}'.format(
        test_loss, model.alpha_pos, model.alpha_neg))
    print(f"  --> Distribution Estimation (Marginal Log-Likelihood): {test_log_p:.4f}")

    losses.append(test_loss)
    recon_losses.append((test_recon_loss_MSE, test_recon_loss_CE))
    log_p_vals.append(test_log_p)
    return losses, recon_losses, log_p_vals


def run(model_type, alpha_pos, alpha_neg, data_name, seed):
    learning_rate = 0.001
    testing_frequency = 1

    # Folder for results
    path = f"./models_{data_name}_seed{seed}_density_features"
    os.makedirs(path, exist_ok=True)

    filename_prefix = f"{model_type}_{alpha_pos}_{alpha_neg}_{data_name}_seed{seed}"
    model_file_path = os.path.join(path, f"{filename_prefix}_model.pt")

    if os.path.exists(model_file_path):
        print(f"[SKIP] {filename_prefix} already exists. Skipping.")
        return

    print(f"[START] {filename_prefix}")

    # Load Features
    try:
        train_loader, test_loader, input_dim = load_feature_dataset(data_name)
    except Exception as e:
        print(f"[ERROR] Could not load features for {data_name}: {e}")
        return

    print(f"[{data_name}] Loaded Features. Input Dim: {input_dim}")

    train_losses, test_losses = [], []
    train_recon_losses, test_recon_losses = [], []
    train_log_p_vals, test_log_p_vals = [], []

    model = vr_model(input_dim, alpha_pos, alpha_neg).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(datetime.datetime.now())
    eps = 1e-4
    epoch = 0
    testing_cnt = 0

    while True:
        if epoch > 200:
            break

        if testing_cnt >= 3 and test_losses[-1] >= test_losses[-2] and test_losses[-2] >= test_losses[-3]:
            print("Early stopping: Test loss increased 3 times.")
            break

        if len(train_losses) >= 5 and \
                np.abs(train_losses[-1] - train_losses[-2]) <= eps and \
                np.abs(train_losses[-2] - train_losses[-3]) <= eps:
            print("Early stopping: Train loss converged.")
            break

        train_losses, train_recon_losses, train_log_p_vals = train(
            model, optimizer, epoch, train_loader,
            model_type, train_losses, train_recon_losses, train_log_p_vals
        )

        if epoch % testing_frequency == 0:
            test_losses, test_recon_losses, test_log_p_vals = test(
                model, epoch, test_loader,
                model_type, test_losses, test_recon_losses, test_log_p_vals,
                img_shape=(1, 1, input_dim)
            )
            testing_cnt += 1

        epoch += 1

    print(datetime.datetime.now())
    print(f"[DONE] {filename_prefix}")

    # Save
    torch.save(train_losses, os.path.join(path, f"{filename_prefix}_train_losses.pt"))
    torch.save(train_recon_losses, os.path.join(path, f"{filename_prefix}_train_recon_losses.pt"))
    torch.save(train_log_p_vals, os.path.join(path, f"{filename_prefix}_train_log_p_vals.pt"))
    torch.save(test_losses, os.path.join(path, f"{filename_prefix}_test_losses.pt"))
    torch.save(test_recon_losses, os.path.join(path, f"{filename_prefix}_test_recon_losses.pt"))
    torch.save(test_log_p_vals, os.path.join(path, f"{filename_prefix}_test_log_p_vals.pt"))
    torch.save(model.state_dict(), model_file_path)


def run_all():
    configure_per_process_threads()

    domains = [
        'Art', 'Clipart', 'Product', 'Real World'
    ]

    seeds = [1]
    tasks = [(domain, seed) for domain in domains for seed in seeds]

    def wrapped_run(domain, seed):
        torch.manual_seed(seed)
        run('vae', alpha_pos=1, alpha_neg=-1, data_name=domain, seed=seed)
        # run('vr', alpha_pos=2, alpha_neg=-1, data_name=domain, seed=seed)
        # run('vr', alpha_pos=0.5, alpha_neg=-1, data_name=domain, seed=seed)
        # run('vr', alpha_pos=5, alpha_neg=-1, data_name=domain, seed=seed)
        run('vrs', alpha_pos=0.5, alpha_neg=-0.5, data_name=domain, seed=seed)
        run('vrs', alpha_pos=2, alpha_neg=-2, data_name=domain, seed=seed)
        # run('vrlu', alpha_pos=0.5, alpha_neg=-0.5, data_name=domain, seed=seed)
        # --- Active Configurations (Sandwich Bounds) ---
        # run('vrs', alpha_pos=2, alpha_neg=-0.5, data_name=domain, seed=seed)
        # run('vrs', alpha_pos=0.5, alpha_neg=-2, data_name=domain, seed=seed)

    print(f"Starting parallel training on {N_JOBS} cores...")

    Parallel(
        n_jobs=N_JOBS,
        backend="loky",
        prefer="processes",
        batch_size=1
    )([delayed(wrapped_run)(domain, seed) for domain, seed in tasks])


if __name__ == "__main__":
    configure_per_process_threads()
    run_all()