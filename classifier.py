import glob
import os
from PIL import Image

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import ssl
from joblib import Parallel, delayed

# Fix for some network environments preventing download of pre-trained weights
ssl._create_default_https_context = ssl._create_unverified_context

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import numpy as np
import torchvision.models as models

# Import your data loader file
import data as Data

TOTAL_CPUS = os.cpu_count() or 1

def save_checkpoint(path, model, optimizer, epoch, best_acc, scheduler=None, train_losses=None):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_acc": best_acc,
        "train_losses": train_losses if train_losses is not None else [],
    }
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(checkpoint, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, device="cpu"):
    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    start_epoch = checkpoint.get("epoch", 0) + 1
    best_acc = checkpoint.get("best_acc", 0)
    train_losses = checkpoint.get("train_losses", [])

    return model, optimizer, scheduler, start_epoch, best_acc, train_losses

# --- 1. The ORIGINAL Model (Strictly for Digits) ---
class Grey_32_64_128_gp(nn.Module):
    def __init__(self, n_classes=10):
        super(Grey_32_64_128_gp, self).__init__()

        self.conv1_1 = nn.Conv2d(1, 32, (3, 3), padding=1)
        self.conv1_1_bn = nn.BatchNorm2d(32)
        self.conv1_2 = nn.Conv2d(32, 32, (3, 3), padding=1)
        self.conv1_2_bn = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d((2, 2))

        self.conv2_1 = nn.Conv2d(32, 64, (3, 3), padding=1)
        self.conv2_1_bn = nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(64, 64, (3, 3), padding=1)
        self.conv2_2_bn = nn.BatchNorm2d(64)
        self.conv2_3 = nn.Conv2d(64, 64, (3, 3), padding=1)
        self.conv2_3_bn = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d((2, 2))
        self.conv3_1 = nn.Conv2d(64, 128, (3, 3), padding=1)
        self.conv3_1_bn = nn.BatchNorm2d(128)
        self.conv3_2 = nn.Conv2d(128, 128, (3, 3), padding=1)
        self.conv3_2_bn = nn.BatchNorm2d(128)
        self.conv3_3 = nn.Conv2d(128, 128, (3, 3), padding=1)
        self.conv3_3_bn = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d((2, 2))
        self.drop1 = nn.Dropout()

        self.fc4 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, n_classes)

    def forward(self, x):
        x = F.relu(self.conv1_1_bn(self.conv1_1(x)))
        x = self.pool1(F.relu(self.conv1_2_bn(self.conv1_2(x))))

        x = F.relu(self.conv2_1_bn(self.conv2_1(x)))
        x = F.relu(self.conv2_2_bn(self.conv2_2(x)))
        x = self.pool2(F.relu(self.conv2_3_bn(self.conv2_3(x))))

        x = F.relu(self.conv3_1_bn(self.conv3_1(x)))
        x = F.relu(self.conv3_2_bn(self.conv3_2(x)))
        x = F.relu(self.conv3_3_bn(self.conv3_3(x)))

        x = F.avg_pool2d(x, 4)
        x = x.view(-1, 128)
        x = self.drop1(x)

        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x


# --- 2. The Model Factory (Selector) ---

def build_network(domain):
    """
    Returns the appropriate model architecture based on the domain name.
    """
    if domain in ['MNIST', 'USPS', 'SVHN']:
        return Grey_32_64_128_gp(n_classes=10)

    elif domain in ['Art', 'Clipart', 'Product', 'Real World']:
        model = models.resnet50(weights='IMAGENET1K_V1')
        n_classes = 65
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, n_classes)
        return model

    elif domain in ['amazon', 'dslr', 'webcam']:
        model = models.resnet50(weights='IMAGENET1K_V1')
        n_classes = 31
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, n_classes)
        return model

    elif domain in ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']:
        model = models.resnet50(weights='IMAGENET1K_V1')
        n_classes = 345
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, n_classes)
        return model

    else:
        raise ValueError(f"Unknown domain for network factory: {domain}")


# --- Training & Testing Loops ---

def train(network, optimizer, train_loader, epoch):
    train_loss = 0
    classification_criterion = nn.CrossEntropyLoss()

    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if data.dim() == 3:
            data = data.unsqueeze(1)

        data = data.to(Data.device)

        if isinstance(target, int):
            target = torch.tensor(target)
        target = target.long().to(Data.device)

        optimizer.zero_grad()

        output = network(data)
        loss = classification_criterion(output, target)

        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

        train_loss += loss.item()

    return train_loss / len(train_loader)


def test(network, test_loader):
    network.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            if data.dim() == 3:
                data = data.unsqueeze(1)

            data = data.to(Data.device)
            target = target.to(Data.device)

            output = network(data)
            pred = output.argmax(dim=1, keepdim=True)

            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

    acc = 100. * correct / total
    print('\nTest set Accuracy: {}/{} ({:.2f}%)\n'.format(correct, total, acc))
    return acc


def run_classifier(domain):
    train_loader, test_loader, config = Data.get_data_loaders(domain)
    network = build_network(domain).to(Data.device)

    if domain in ['MNIST', 'USPS', 'SVHN']:
        lr = 0.005
        max_epochs = 200
        use_scheduler = False
    else:
        lr = 0.0001
        max_epochs = 50
        use_scheduler = True

    optimizer = optim.Adam(network.parameters(), lr=lr, weight_decay=1e-5)

    scheduler = None
    if use_scheduler:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

    train_losses = []
    best_acc = 0
    epsilon = 1e-4
    start_epoch = 1

    checkpoint_path = f"./classifiers/{domain}_checkpoint.pt"
    best_model_path = f"./classifiers/{domain}_224.pt"

    # Resume if checkpoint exists
    if os.path.exists(checkpoint_path):
        print(f"[{domain}] Found checkpoint. Resuming training from {checkpoint_path}")
        try:
            network, optimizer, scheduler, start_epoch, best_acc, train_losses = load_checkpoint(
                checkpoint_path,
                network,
                optimizer=optimizer,
                scheduler=scheduler,
                device=Data.device
            )
            print(f"[{domain}] Resumed from epoch {start_epoch} with best_acc={best_acc:.2f}%")
        except Exception as e:
            print(f"[{domain}] Failed to load checkpoint: {e}. Starting from scratch.")
            start_epoch = 1
            best_acc = 0
            train_losses = []

    print(f"STARTING TRAIN: {domain} | Max Epochs: {max_epochs} | LR: {lr} | Device: {Data.device}")

    for epoch in range(start_epoch, max_epochs + 1):
        avg_loss = train(network, optimizer, train_loader, epoch)
        train_losses.append(avg_loss)

        if scheduler:
            scheduler.step()

        acc = test(network, test_loader)

        # save best model weights
        if acc > best_acc:
            best_acc = acc
            torch.save(network.state_dict(), best_model_path)
            print(f"*** New Best Model Saved (Acc: {best_acc:.2f}%) ***")

        # save full checkpoint every epoch
        save_checkpoint(
            checkpoint_path,
            network,
            optimizer,
            epoch,
            best_acc,
            scheduler=scheduler,
            train_losses=train_losses
        )

        if len(train_losses) > 5 and \
                np.abs(train_losses[-1] - train_losses[-2]) <= epsilon and \
                np.abs(train_losses[-2] - train_losses[-3]) <= epsilon:
            print("Loss Converged.")
            break

    # if no best model was ever saved, save current model
    if not os.path.exists(best_model_path):
        torch.save(network.state_dict(), best_model_path)

    print(f"=== Finished {domain}. Best Accuracy: {best_acc:.2f}% ===")
    return best_acc



# --- Parallel Execution Wrapper ---

def parallel_wrapper(domain):
    print(f"--- Process starting for {domain} on PID {os.getpid()} ---")

    checkpoint_path = f"./classifiers/{domain}_checkpoint.pt"
    best_model_path = f"./classifiers/{domain}_224.pt"
    legacy_model_path = f"./classifiers/{domain}_classifier.pt"

    # If checkpoint exists -> resume training
    if os.path.exists(checkpoint_path):
        print(f"[{domain}] Checkpoint found. Resuming training.")
        return run_classifier(domain)

    # If only final model exists -> just evaluate it
    save_path = best_model_path if os.path.exists(best_model_path) else legacy_model_path
    if os.path.exists(save_path):
        print(f"[{domain}] Found existing final model at {save_path}. Evaluating...")
        try:
            model = build_network(domain).to(Data.device)
            model.load_state_dict(torch.load(save_path, map_location=Data.device))
            _, test_loader, _ = Data.get_data_loaders(domain)
            acc = test(model, test_loader)
            print(f"[{domain}] Existing model accuracy: {acc:.2f}%")
            return acc
        except Exception as e:
            print(f"[{domain}] Error loading final model: {e}. Re-training...")

    return run_classifier(domain)


# --- Cross-Domain Evaluation Matrix ---

def evaluate_cross_domain(domains):
    print("\n" + "#" * 60)
    print(">>> STARTING CROSS-DOMAIN EVALUATION MATRIX <<<")
    print("#" * 60)

    print("--> Pre-loading test datasets...")
    test_loaders = {}
    for d in domains:
        _, test_loader, _ = Data.get_data_loaders(d)
        test_loaders[d] = test_loader

    results_matrix = {s: {} for s in domains}

    for source in domains:
        print(f"\n[Source Model: {source}] Loading weights...")

        model = build_network(source).to(Data.device)

        p1 = f"./classifiers/{source}_224.pt"
        p2 = f"./classifiers/{source}_classifier.pt"
        model_path = p1 if os.path.exists(p1) else p2

        if not os.path.exists(model_path):
            print(f"[WARNING] Model for {source} not found in ./classifiers/. Skipping.")
            continue

        try:
            model.load_state_dict(torch.load(model_path, map_location=Data.device))
            model.eval()
        except Exception as e:
            print(f"[ERROR] Could not load {source}: {e}")
            continue

        for target in domains:
            loader = test_loaders[target]
            correct = 0
            total = 0

            with torch.no_grad():
                for i, (data, lbl) in enumerate(loader):
                    if data.dim() == 3: data = data.unsqueeze(1)
                    data = data.to(Data.device)
                    lbl = lbl.to(Data.device)

                    output = model(data)
                    preds = output.argmax(dim=1)
                    correct += preds.eq(lbl).sum().item()
                    total += lbl.size(0)

                    if i % 20 == 0:
                        print(f"   [Eval] {source} -> {target}: Batch {i}/{len(loader)}", end='\r')

            accuracy = correct / total
            error = 1.0 - accuracy
            results_matrix[source][target] = error

            print(f"   -> Tested on {target:<10} | Acc: {accuracy:.2%} | Err: {error:.4f}")

    print("\n" + "=" * 80)
    print(f"{'SOURCE (Model) / TARGET (Data)':<30} | " + " | ".join([f"{d:<10}" for d in domains]))
    print("-" * 80)

    for source in domains:
        row_str = f"{source:<30} | "
        for target in domains:
            err = results_matrix[source].get(target, -1)
            if err == -1:
                row_str += f"{'N/A':<10} | "
            else:
                row_str += f"{err:.4f}      | "
        print(row_str)
    print("=" * 80 + "\n")

def sanity_check_domainnet(domains, classifiers_dir="./classifiers"):
    """
    Sanity checks for DomainNet:
    1. Inspect train/test label ranges and unique-label counts per domain
    2. Check whether saved model files exist
    3. Evaluate each source model on its own target domain (self-domain only)

    This helps detect:
    - wrong label ranges
    - missing classes
    - missing / stale model files
    - suspiciously low self-domain accuracy
    """
    print("\n" + "#" * 80)
    print("SANITY CHECK: DOMAINNET LABELS + SAVED MODELS + SELF-DOMAIN ACCURACY")
    print("#" * 80)

    # ----------------------------
    # Part 1: dataset label sanity
    # ----------------------------
    print("\n[1] DATASET LABEL SANITY")
    print("-" * 80)

    for d in domains:
        train_file = os.path.join(Data.DOMAINNET_PATH, f"{d}_train.txt")
        test_file = os.path.join(Data.DOMAINNET_PATH, f"{d}_test.txt")

        train_ds = Data.DomainNetSplitDataset(
            root=Data.DOMAINNET_PATH,
            split_file=train_file,
            transform=None
        )
        test_ds = Data.DomainNetSplitDataset(
            root=Data.DOMAINNET_PATH,
            split_file=test_file,
            transform=None
        )

        train_labels = [label for _, label in train_ds.samples]
        test_labels = [label for _, label in test_ds.samples]

        train_unique = sorted(set(train_labels))
        test_unique = sorted(set(test_labels))

        train_missing = sorted(set(range(345)) - set(train_unique))
        test_missing = sorted(set(range(345)) - set(test_unique))

        print(f"\nDomain: {d}")
        print(
            f"  TRAIN | n={len(train_labels):7d} | "
            f"min={min(train_labels):3d} | max={max(train_labels):3d} | "
            f"unique={len(train_unique):3d}"
        )
        print(
            f"  TEST  | n={len(test_labels):7d} | "
            f"min={min(test_labels):3d} | max={max(test_labels):3d} | "
            f"unique={len(test_unique):3d}"
        )

        if train_unique != test_unique:
            only_train = sorted(set(train_unique) - set(test_unique))
            only_test = sorted(set(test_unique) - set(train_unique))
            print("  [WARNING] Train/Test unique-label sets differ!")
            print(f"    labels only in TRAIN: {only_train[:20]}{' ...' if len(only_train) > 20 else ''}")
            print(f"    labels only in TEST : {only_test[:20]}{' ...' if len(only_test) > 20 else ''}")

        if min(train_labels) < 0 or max(train_labels) >= 345 or min(test_labels) < 0 or max(test_labels) >= 345:
            print("  [WARNING] Labels are outside expected DomainNet range [0, 344].")

        if train_missing:
            print(f"  Missing in TRAIN (first 20): {train_missing[:20]}{' ...' if len(train_missing) > 20 else ''}")
        if test_missing:
            print(f"  Missing in TEST  (first 20): {test_missing[:20]}{' ...' if len(test_missing) > 20 else ''}")

    # ---------------------------------
    # Part 2: saved model existence/info
    # ---------------------------------
    print("\n[2] SAVED MODEL FILE CHECK")
    print("-" * 80)

    for d in domains:
        best_model_path = os.path.join(classifiers_dir, f"{d}_224.pt")
        legacy_model_path = os.path.join(classifiers_dir, f"{d}_classifier.pt")
        checkpoint_path = os.path.join(classifiers_dir, f"{d}_checkpoint.pt")

        print(f"\nDomain: {d}")
        print(f"  best_model   : {best_model_path} | exists={os.path.exists(best_model_path)}")
        print(f"  legacy_model : {legacy_model_path} | exists={os.path.exists(legacy_model_path)}")
        print(f"  checkpoint   : {checkpoint_path} | exists={os.path.exists(checkpoint_path)}")

        for p in [best_model_path, legacy_model_path, checkpoint_path]:
            if os.path.exists(p):
                try:
                    mtime = os.path.getmtime(p)
                    size_mb = os.path.getsize(p) / (1024 ** 2)
                    print(f"    -> {os.path.basename(p)} | size={size_mb:.2f} MB | mtime={mtime}")
                except Exception as e:
                    print(f"    -> could not inspect {p}: {e}")

    # ------------------------------------
    # Part 3: self-domain evaluation only
    # ------------------------------------
    print("\n[3] SELF-DOMAIN EVALUATION")
    print("-" * 80)

    for d in domains:
        print(f"\nEvaluating self-domain model for: {d}")

        model = build_network(d).to(Data.device)

        best_model_path = os.path.join(classifiers_dir, f"{d}_224.pt")
        legacy_model_path = os.path.join(classifiers_dir, f"{d}_classifier.pt")
        model_path = best_model_path if os.path.exists(best_model_path) else legacy_model_path

        if not os.path.exists(model_path):
            print(f"  [WARNING] No saved model found for {d}. Skipping.")
            continue

        try:
            model.load_state_dict(torch.load(model_path, map_location=Data.device))
            _, test_loader, _ = Data.get_data_loaders(d)
            acc = test(model, test_loader)
            print(f"  SELF ACCURACY ({d} -> {d}) = {acc:.2f}%")
        except Exception as e:
            print(f"  [ERROR] Failed loading/evaluating model for {d}: {e}")

    print("\n" + "#" * 80)
    print("END OF DOMAINNET SANITY CHECK")
    print("#" * 80 + "\n")

# --- Main Execution Block ---

if __name__ == '__main__':
    # Ensure the directory matches the path used in the functions
    save_dir = "./classifiers"
    os.makedirs(save_dir, exist_ok=True)

    # CASE 2: Office-Home--------
    # domains_to_run = ['Art', 'Clipart', 'Product', 'Real World']
    # domains_to_run = ['amazon', 'dslr', 'webcam']
    domains_to_run = ['clipart', 'infograph', 'painting' , 'quickdraw', 'real', 'sketch']
    print(f"Running for domains: {domains_to_run}")

    num_jobs = 1 if torch.cuda.is_available() else min(len(domains_to_run), max(1, TOTAL_CPUS // 4))
    print(f"Using {num_jobs} parallel jobs out of {TOTAL_CPUS} CPUs")

    # results = Parallel(n_jobs=num_jobs, backend="loky")(
    #     delayed(parallel_wrapper)(d) for d in domains_to_run
    # )

    print("\n" + "=" * 30)
    print("INDIVIDUAL TRAINING COMPLETED")
    print("=" * 30)

    evaluate_cross_domain(domains_to_run)