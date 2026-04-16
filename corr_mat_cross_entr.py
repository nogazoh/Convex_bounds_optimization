# corr_mat_cross_entr.py

import os
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from scipy.stats import spearmanr
from matplotlib.colors import TwoSlopeNorm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

import classifier as ClSFR
from msa_all_summer import FeatureExtractor
from helpers import load_global_D_matrix


# ============================================================
# GLOBAL CONFIG
# ============================================================

SEED = 1
BATCH_SIZE = 64
TEST_BATCH_SIZE = 64
RESULTS_DIR = "./correlation_matrices_results"
CLASSIFIERS_DIR = "./classifiers"

HEATMAP_VMIN = -0.5
HEATMAP_VMAX = 0.5
HEATMAP_CENTER = 0.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

DATASET_CONFIGS = {
    "OFFICE31": {
        "domains": ["amazon", "dslr", "webcam"],
        "num_classes": 31,
        "dataset_root": "/data/nogaz/Convex_bounds_optimization/Office-31",
        "d_precomp_path": "/data/nogaz/Convex_bounds_optimization/LatentFlow_Pixel_Experiments/results/D_Matrix_FINAL_GMM_Soft.npy",
        "target_size": 224,
    },
    "OFFICE224": {
        "domains": ["Art", "Clipart", "Product", "Real World"],
        "num_classes": 65,
        "dataset_root": "/data/nogaz/Convex_bounds_optimization/OfficeHome",
        "d_precomp_path": "/data/nogaz/Convex_bounds_optimization/LatentFlow_Pixel_Experiments/results/D_Matrix_FINAL_GMM_Soft_OfficeHome.npy",
        "target_size": 224,
    },
}


# ============================================================
# HELPERS
# ============================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def safe_torch_load(path, map_location=device):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def safe_pearsonr(x, y):
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)

    if len(x) != len(y):
        raise ValueError(f"Length mismatch: {len(x)} vs {len(y)}")

    if len(x) < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan

    return float(np.corrcoef(x, y)[0, 1])


def safe_spearmanr(x, y):
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)

    if len(x) != len(y):
        raise ValueError(f"Length mismatch: {len(x)} vs {len(y)}")

    if len(x) < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan

    return float(spearmanr(x, y).correlation)


def find_classifier_path(domain):
    p1 = os.path.join(CLASSIFIERS_DIR, f"{domain}_224.pt")
    p2 = os.path.join(CLASSIFIERS_DIR, f"{domain}_classifier.pt")

    if os.path.exists(p1):
        return p1
    if os.path.exists(p2):
        return p2

    raise FileNotFoundError(f"Could not find classifier for {domain} in {CLASSIFIERS_DIR}")


def get_domain_path(dataset_mode, domain_name):
    root = DATASET_CONFIGS[dataset_mode]["dataset_root"]
    p1 = os.path.join(root, domain_name, "images")
    p2 = os.path.join(root, domain_name)

    if os.path.exists(p1):
        return p1
    if os.path.exists(p2):
        return p2

    raise FileNotFoundError(
        f"Could not find domain folder for dataset_mode={dataset_mode}, domain={domain_name}. "
        f"Tried:\n  {p1}\n  {p2}"
    )


class TransformedSubset(torch.utils.data.Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform is not None:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


def make_rgb_transforms(target_size=224):
    train_trans = transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    test_trans = transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return train_trans, test_trans


def make_loader(dataset, batch_size, shuffle=False, num_workers=4, pin_memory=None):
    if pin_memory is None:
        pin_memory = (device.type == "cuda")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


# ============================================================
# DATASET LOCAL LOGIC
# ============================================================

def compute_domain_lengths_local(dataset_mode, domains):
    lengths = {}
    for d in domains:
        path = get_domain_path(dataset_mode, d)
        ds_full = datasets.ImageFolder(path)
        lengths[d] = len(ds_full)
    return lengths


def get_train_test_loaders_and_indices_local(dataset_mode, domain, seed, batch_size=64, test_batch_size=64):
    cfg = DATASET_CONFIGS[dataset_mode]
    target_size = cfg["target_size"]

    train_trans, test_trans = make_rgb_transforms(target_size)
    path = get_domain_path(dataset_mode, domain)
    ds_full = datasets.ImageFolder(path)

    N = len(ds_full)
    rng = np.random.RandomState(seed)
    indices = rng.permutation(N)
    split_point = int(0.8 * N)

    train_idx = indices[:split_point]
    test_idx = indices[split_point:]

    train_ds = TransformedSubset(Subset(ds_full, train_idx), train_trans)
    test_ds = TransformedSubset(Subset(ds_full, test_idx), test_trans)

    train_loader = make_loader(train_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = make_loader(test_ds, batch_size=test_batch_size, shuffle=False, num_workers=4)

    return train_loader, test_loader, train_idx, test_idx, N


def get_domain_offsets(domain_lengths, domains):
    offsets = {}
    start = 0
    for d in domains:
        end = start + domain_lengths[d]
        offsets[d] = (start, end)
        start = end
    return offsets


def slice_global_D_for_domain(Global_D, domains, domain_lengths, target_domain):
    offsets = get_domain_offsets(domain_lengths, domains)
    start, end = offsets[target_domain]
    return Global_D[start:end, :]


def row_normalize_scores(D_block, eps=1e-12):
    row_sums = D_block.sum(axis=1, keepdims=True)
    return D_block / np.clip(row_sums, eps, None)


# ============================================================
# MODEL LOADING
# ============================================================

def load_source_models(domains):
    models = {}

    for d in domains:
        print(f"Loading classifier for source: {d}")
        model_path = find_classifier_path(d)

        base_model = ClSFR.build_network(d).to(device)
        state_dict = safe_torch_load(model_path, map_location=device)
        base_model.load_state_dict(state_dict)

        extractor = FeatureExtractor(base_model).to(device).eval()
        models[d] = extractor

    return models


# ============================================================
# ERROR COLLECTION
# ============================================================

def collect_metrics_for_target(dataset_mode, target_domain, domains, source_models):
    print(f"\nCollecting metrics on target={target_domain}")

    _, test_loader, _, test_idx, _ = get_train_test_loaders_and_indices_local(
        dataset_mode=dataset_mode,
        domain=target_domain,
        seed=SEED,
        batch_size=BATCH_SIZE,
        test_batch_size=TEST_BATCH_SIZE,
    )

    err01_by_source = {src: [] for src in domains}

    for batch_i, (data, labels) in enumerate(test_loader):
        data = data.to(device)
        labels = labels.long().to(device)

        with torch.no_grad():
            for src in domains:
                _, logits = source_models[src](data)
                preds = torch.argmax(logits, dim=1)
                err01 = (preds != labels).float()
                err01_by_source[src].append(err01.detach().cpu().numpy())

        if batch_i % 20 == 0:
            print(f"  target={target_domain}: batch {batch_i}/{len(test_loader)}", end="\r")

    for src in domains:
        err01_by_source[src] = np.concatenate(err01_by_source[src], axis=0)

    print(f"\nFinished target={target_domain}. N_test={len(test_idx)}")
    return err01_by_source, test_idx


# ============================================================
# CORRELATION LOGIC
# ============================================================

def compute_all_correlations(score_vec, err01_vec):
    return {
        "pearson_err01": safe_pearsonr(score_vec, err01_vec),
        "spearman_err01": safe_spearmanr(score_vec, err01_vec),
    }


def init_result_matrices(domains):
    keys = [
        "pearson_err01",
        "spearman_err01",
    ]
    return {k: np.full((len(domains), len(domains)), np.nan, dtype=float) for k in keys}


def save_one_matrix(matrix, domains, dataset_mode, score_type, metric_name, save_dir):
    ensure_dir(save_dir)

    df = pd.DataFrame(matrix, index=domains, columns=domains)

    stem = f"{dataset_mode}_{score_type}_{metric_name}"
    csv_path = os.path.join(save_dir, f"{stem}.csv")
    npy_path = os.path.join(save_dir, f"{stem}.npy")
    png_path = os.path.join(save_dir, f"{stem}.png")

    df.to_csv(csv_path)
    np.save(npy_path, matrix)

    fig, ax = plt.subplots(figsize=(1.8 * len(domains) + 2, 1.6 * len(domains) + 2))

    norm = TwoSlopeNorm(vmin=HEATMAP_VMIN, vcenter=HEATMAP_CENTER, vmax=HEATMAP_VMAX)
    im = ax.imshow(matrix, aspect="auto", cmap="bwr", norm=norm)

    ax.set_xticks(np.arange(len(domains)))
    ax.set_yticks(np.arange(len(domains)))
    ax.set_xticklabels(domains, rotation=45, ha="right")
    ax.set_yticklabels(domains)

    ax.set_xlabel("Target domain")
    ax.set_ylabel("Source domain")
    ax.set_title(f"{dataset_mode}: {score_type} vs {metric_name}")

    for i in range(len(domains)):
        for j in range(len(domains)):
            val = matrix[i, j]
            txt = "nan" if np.isnan(val) else f"{val:.3f}"
            ax.text(j, i, txt, ha="center", va="center")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Correlation")

    plt.tight_layout()
    plt.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_metadata(dataset_mode, domains, save_dir):
    meta = {
        "dataset_mode": dataset_mode,
        "domains": domains,
        "score_types": {
            "col_score": (
                "Original D as saved in the soft-GMM pipeline. "
                "Columns are normalized to sum to 1, so each source column is normalized separately."
            ),
            "row_score": (
                "Row-normalized version of D over active sources. "
                "Rows sum to 1, so each sample gets a relative score over sources."
            ),
        },
        "metrics": {
            "pearson_err01": "Pearson correlation between score and per-sample 0/1 error",
            "spearman_err01": "Spearman correlation between score and per-sample 0/1 error",
        },
        "heatmap": {
            "cmap": "bwr",
            "vmin": HEATMAP_VMIN,
            "vcenter": HEATMAP_CENTER,
            "vmax": HEATMAP_VMAX,
            "meaning": "negative=blue, zero=white, positive=red",
        },
        "interpretation": (
            "More negative values mean that when a source gives a sample a higher score, "
            "that source tends to make fewer 0/1 mistakes on that sample."
        ),
    }

    with open(os.path.join(save_dir, f"{dataset_mode}_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


# ============================================================
# MAIN ANALYSIS
# ============================================================

def build_correlation_outputs_for_dataset(dataset_mode):
    cfg = DATASET_CONFIGS[dataset_mode]
    domains = cfg["domains"]
    d_precomp_path = cfg["d_precomp_path"]

    print("\n" + "=" * 80)
    print(f"Building correlation outputs for {dataset_mode}")
    print("=" * 80)

    source_models = load_source_models(domains)

    if not os.path.exists(d_precomp_path):
        raise FileNotFoundError(f"D matrix not found: {d_precomp_path}")

    print(f"Loading Global_D from: {d_precomp_path}")
    Global_D = load_global_D_matrix(d_precomp_path)

    domain_lengths = compute_domain_lengths_local(dataset_mode, domains)

    expected_total = sum(domain_lengths[d] for d in domains)
    if Global_D.shape[0] != expected_total:
        raise ValueError(f"Global_D rows mismatch: {Global_D.shape[0]} vs {expected_total}")

    if Global_D.shape[1] != len(domains):
        raise ValueError(f"Global_D cols mismatch: {Global_D.shape[1]} vs {len(domains)}")

    results = {
        "col_score": init_result_matrices(domains),
        "row_score": init_result_matrices(domains),
    }

    for target_j, target_domain in enumerate(domains):
        err01_by_source, test_idx = collect_metrics_for_target(
            dataset_mode=dataset_mode,
            target_domain=target_domain,
            domains=domains,
            source_models=source_models,
        )

        D_target_full = slice_global_D_for_domain(
            Global_D=Global_D,
            domains=domains,
            domain_lengths=domain_lengths,
            target_domain=target_domain,
        )

        D_test_col = D_target_full[test_idx, :]
        D_test_row = row_normalize_scores(D_test_col)

        for source_i, src in enumerate(domains):
            err01_vec = err01_by_source[src]

            score_variants = {
                "col_score": D_test_col[:, source_i],
                "row_score": D_test_row[:, source_i],
            }

            for score_type, score_vec in score_variants.items():
                out = compute_all_correlations(score_vec, err01_vec)

                for metric_name, val in out.items():
                    results[score_type][metric_name][source_i, target_j] = val

                print(
                    f"  [{score_type}] source={src:<12} target={target_domain:<12} "
                    f"pearson_err01={out['pearson_err01']:.4f} "
                    f"spearman_err01={out['spearman_err01']:.4f}"
                )

    save_dir = os.path.join(RESULTS_DIR, dataset_mode)
    ensure_dir(save_dir)
    save_metadata(dataset_mode, domains, save_dir)

    for score_type, mats in results.items():
        for metric_name, matrix in mats.items():
            save_one_matrix(matrix, domains, dataset_mode, score_type, metric_name, save_dir)

    return results


def print_summary_tables(results, dataset_mode):
    domains = DATASET_CONFIGS[dataset_mode]["domains"]
    for score_type, mats in results.items():
        print("\n" + "-" * 80)
        print(f"{dataset_mode} | {score_type}")
        print("-" * 80)
        for metric_name, matrix in mats.items():
            print(f"\n{metric_name}:")
            print(pd.DataFrame(matrix, index=domains, columns=domains))


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    ensure_dir(RESULTS_DIR)

    office31_results = build_correlation_outputs_for_dataset("OFFICE31")
    office224_results = build_correlation_outputs_for_dataset("OFFICE224")

    print_summary_tables(office31_results, "OFFICE31")
    print_summary_tables(office224_results, "OFFICE224")


if __name__ == "__main__":
    main()