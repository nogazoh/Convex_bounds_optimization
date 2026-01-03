import glob
import os
import ssl
from joblib import Parallel, delayed

ssl._create_default_https_context = ssl._create_unverified_context

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torchvision.models as models

import data as Data


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
        self.conv3_1 = nn.Conv2d(64, 128, (3, 3), padding=1)
        self.conv3_1_bn = nn.BatchNorm2d(128)
        self.conv3_2 = nn.Conv2d(128, 128, (3, 3), padding=1)
        self.conv3_2_bn = nn.BatchNorm2d(128)
        self.conv3_3 = nn.Conv2d(128, 128, (3, 3), padding=1)
        self.conv3_3_bn = nn.BatchNorm2d(128)
        # self.pool3 = nn.MaxPool2d((2, 2))

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
    Returns the original Grey model for digits,
    and a powerful ResNet50 for Office-Home.
    """
    # Group A: Digits - Use the original code
    if domain in ['MNIST', 'USPS', 'SVHN']:
        # print(f"--> [Factory] Selected Original Grey Model for {domain}")
        return Grey_32_64_128_gp(n_classes=10)

    # Group B: Office-Home - Use ResNet50
    else:
        # print(f"--> [Factory] Selected ResNet-50 (Pretrained) for {domain}")
        # Load ResNet50 with ImageNet weights
        model = models.resnet50(weights='IMAGENET1K_V1')

        # Office-Home has 65 classes
        n_classes = 65

        # Replace the final layer
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, n_classes)

        return model


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
    # 1. Get Data
    train_loader, test_loader, config = Data.get_data_loaders(domain)

    # 2. Build Network
    network = build_network(domain).to(Data.device)

    # 3. Configure Hyperparameters per Domain Type
    if domain in ['MNIST', 'USPS', 'SVHN']:
        lr = 0.005
        max_epochs = 200
        use_scheduler = False
    else:
        lr = 0.001
        max_epochs = 50
        use_scheduler = True

    optimizer = optim.Adam(network.parameters(), lr=lr)

    scheduler = None
    if use_scheduler:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

    train_losses = []
    best_acc = 0
    epsilon = 1e-4

    print(f"STARTING TRAIN: {domain} | Max Epochs: {max_epochs} | LR: {lr}")

    for epoch in range(1, max_epochs + 1):

        avg_loss = train(network, optimizer, train_loader, epoch)
        train_losses.append(avg_loss)

        if scheduler:
            scheduler.step()

        if epoch % 1 == 0:
            acc = test(network, test_loader)

            if acc > best_acc:
                best_acc = acc
                save_path = f"./classifiers_new/{domain}_classifier.pt"
                torch.save(network.state_dict(), save_path)
                print(f"*** New Best Model Saved (Acc: {best_acc:.2f}%) ***")

        if len(train_losses) > 5 and \
                np.abs(train_losses[-1] - train_losses[-2]) <= epsilon and \
                np.abs(train_losses[-2] - train_losses[-3]) <= epsilon:
            print("Loss Converged.")
            break

    if best_acc == 0:
        save_path = f"./classifiers_new/{domain}_classifier.pt"
        torch.save(network.state_dict(), save_path)

    print(f"=== Finished {domain}. Best Accuracy: {best_acc:.2f}% ===")
    return best_acc


# --- Parallel Execution Wrapper (UPDATED) ---

def parallel_wrapper(domain):
    """
    Wrapper function for parallel execution.
    If model exists -> Loads and Evaluates it.
    If model missing -> Trains it.
    """
    # torch.set_num_threads(1)
    # torch.set_num_interop_threads(1)

    print(f"--- Process starting for {domain} on PID {os.getpid()} ---")

    save_path = f"./classifiers_new/{domain}_classifier.pt"

    # === CHANGE STARTS HERE ===
    # If the model already exists, we LOAD and TEST it instead of returning 0
    if os.path.exists(save_path):
        print(f"[{domain}] Found existing model at {save_path}. Evaluating...")

        try:
            # 1. Build empty architecture
            model = build_network(domain).to(Data.device)

            # 2. Load weights
            model.load_state_dict(torch.load(save_path, map_location=Data.device))

            # 3. Get test data
            _, test_loader, _ = Data.get_data_loaders(domain)

            # 4. Run test
            acc = test(model, test_loader)
            print(f"[{domain}] Existing model accuracy: {acc:.2f}%")
            return acc

        except Exception as e:
            print(f"[{domain}] Error loading existing model: {e}. Will re-train.")
            # If loading failed, we fall through to training below
            pass
    # === CHANGE ENDS HERE ===

    # If we are here, either model didn't exist or failed to load
    return run_classifier(domain)


# --- New Function: Cross-Domain Evaluation ---

def evaluate_cross_domain(domains):
    print("\n" + "#" * 60)
    print(">>> STARTING CROSS-DOMAIN EVALUATION MATRIX <<<")
    print("#" * 60)

    # 1. Pre-load all Test Data Loaders to save time
    # (Loading data once is faster than reloading it for every model loop)
    print("--> Pre-loading test datasets...")
    test_loaders = {}
    for d in domains:
        # We only need the test loader
        _, test_loader, _ = Data.get_data_loaders(d)
        test_loaders[d] = test_loader

    # Dictionary to store results: results[Source][Target] = Error
    results_matrix = {s: {} for s in domains}

    # 2. Iterate over each Source Domain (Load Model)
    for source in domains:
        print(f"\n[Source Model: {source}] Loading weights...")

        # Build model and load weights
        model = build_network(source).to(Data.device)
        model_path = f"./classifiers_new/{source}_classifier.pt"

        if not os.path.exists(model_path):
            print(f"[WARNING] Model for {source} not found at {model_path}. Skipping.")
            continue

        try:
            model.load_state_dict(torch.load(model_path, map_location=Data.device))
            model.eval()
        except Exception as e:
            print(f"[ERROR] Could not load {source}: {e}")
            continue

        # 3. Test against ALL Target Domains
        for target in domains:
            loader = test_loaders[target]

            correct = 0
            total = 0

            with torch.no_grad():
                for data, lbl in loader:
                    if data.dim() == 3: data = data.unsqueeze(1)
                    data = data.to(Data.device)
                    lbl = lbl.to(Data.device)

                    output = model(data)
                    preds = output.argmax(dim=1)
                    correct += preds.eq(lbl).sum().item()
                    total += lbl.size(0)

            # Calculate Error (1 - Accuracy)
            accuracy = correct / total
            error = 1.0 - accuracy
            results_matrix[source][target] = error

            print(f"   -> Tested on {target:<10} | Acc: {accuracy:.2%} | Err: {error:.4f}")

    # 4. Print Final Formatted Matrix
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
                # Highlight diagonal (Self-Error) with brackets if you want, or just print
                row_str += f"{err:.4f}     | "
        print(row_str)
    print("=" * 80 + "\n")


# --- Update Main Block ---

if __name__ == '__main__':
    # ... (Your existing code for creating directories and running parallel jobs) ...
    save_dir = os.path.join(os.path.dirname(__file__), "classifiers_new")
    os.makedirs(save_dir, exist_ok=True)

    office_domains = ['Art', 'Clipart', 'Product', 'Real World']
    # office_domains = ['MNIST', 'USPS', 'SVHN'] # Use this if creating Digits models

    num_jobs = 1 #min(len(office_domains), os.cpu_count() or 1)

    print(f"Starting parallel execution on {num_jobs} CPU cores...")

    # This runs the training/testing for each individual model
    results = Parallel(n_jobs=num_jobs, backend="loky")(
        delayed(parallel_wrapper)(d) for d in office_domains
    )

    print("\n" + "=" * 30)
    print("INDIVIDUAL TRAINING COMPLETED")
    print("=" * 30)

    # === NEW: Run the Cross-Domain Matrix Evaluation ===
    # This runs sequentially in the main process to avoid GPU memory conflicts
    evaluate_cross_domain(office_domains)