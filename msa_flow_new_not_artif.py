from __future__ import print_function
import time
import torch.utils.data
import logging
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
import os
from joblib import Parallel, delayed
import torch
import torch.nn.functional as F
import glob
import sys
import io
import itertools
from torchvision import models, transforms, datasets
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import traceback
import pickle
from datetime import datetime
from dc import *
from dc_fast import *
from vae import *
import data as Data
import classifier as ClSFR

# --- SOLVERS ---
try:
    from old_cvxpy_solver import solve_convex_problem_mosek
    from cvxpy_solver_per_domain import solve_convex_problem_per_domain
except ImportError:
    pass
if "solve_convex_problem_mosek" not in globals():
    raise ImportError("solve_convex_problem_mosek not imported. Check old_cvxpy_solver import/path.")

from cvxpy_3_21 import solve_convex_problem_smoothed_kl_321
from cvxpy_3_22 import solve_convex_problem_domain_anchored_smoothed
from cvxpy_3_23 import solve_convex_problem_smoothed_original_p
from cvxpy_3_31 import solve_convex_problem_smoothed_kl_331
from cvxpy_3_32 import solve_convex_problem_domain_anchored_smoothed_332
from cvxpy_3_33 import solve_convex_problem_smoothed_original_p_333
from cvxpy_6 import solve_convex_problem_soft_kl_diagonal_fast
from cvxpy_6a import solve_convex_problem_soft_kl_diagonal_fast_6A

from helpers import *
from loss_functions import (
    evaluate_solution_with_w,
    evaluate_solution_with_q,
    evaluate_predictions_accuracy,
    evaluate_predictions_error_rate,
    evaluate_predictions_cross_entropy,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["OMP_NUM_THREADS"] = "1"
torch.set_num_threads(1)


# ==========================================
# --- CONFIGURATION ---
# ==========================================

# SEED = 1
SEED = 101
# SEED = 201
# SEED = 301
# SEED = 401

USE_ALL_SOURCES_FOR_SOLVER = False

CONSTRAINT_LOSS_TYPE = "01"   # options: "01", "ce"
if CONSTRAINT_LOSS_TYPE == "01":
    EVAL_METRIC = "accuracy"
else:
    EVAL_METRIC = "ce"

# NOTE:
# This file is now configured for SINGLE-TARGET evaluation:
# each run picks one target domain, and all remaining domains are sources.
# We keep the rest of the pipeline as close as possible to the original file.
FILENAME = f'Sweep_Results_SINGLE_TARGET_{CONSTRAINT_LOSS_TYPE}_6_all_sources_{USE_ALL_SOURCES_FOR_SOLVER}.txt'
DATASET_MODE = "OFFICE224"  # "DIGITS" / "OFFICE224" / "OFFICE31" / "DOMAINNET"
USE_ORACLE_EPSILON = True
USE_PRECOMPUTED_D = True
USE_ARTIFICIAL_RATIOS = False   # retained for compatibility; unused in single-target mode

RUN_TAG = f"seed_{SEED}"
CLASSIFIERS_DIR = f"./classifiers_{RUN_TAG}"

FORBIDDEN_EXACT_PAIRS = {
    # retained from original file; not used in single-target mode
    tuple(sorted(['infograph', 'sketch'])),
    tuple(sorted(['painting', 'quickdraw'])),
    tuple(sorted(['painting', 'real'])),
    tuple(sorted(['painting', 'sketch'])),
    tuple(sorted(['quickdraw', 'real'])),
    tuple(sorted(['quickdraw', 'sketch'])),
    tuple(sorted(['real', 'sketch'])),
}

CONFIGS = {
    # "DIGITS": {
    #     "DOMAINS": ['MNIST', 'USPS', 'SVHN'],
    #     "CLASSES": 10,
    #     "INPUT_DIM": 784,
    #     "SOURCE_LOSSES": {'MNIST': 0.005, 'USPS': 0.027, 'SVHN': 0.05},
    #     "TEST_SET_SIZES": {'MNIST': 10000, 'USPS': 2007, 'SVHN': 26032},
    #     "D_PRECOMP_PATH": "/data/nogaz/Convex_bounds_optimization/LatentFlow_Pixel_Experiments/results/D_Matrix_FINAL_GMM_Soft_DIGITS.npy"
    # },
    "OFFICE224": {  # Office-Home
        "DOMAINS": ['Art', 'Clipart', 'Product', 'Real World'],
        "CLASSES": 65,
        "INPUT_DIM": 2048,
        "SOURCE_LOSSES_01": {
            'Art': 0.0139,
            'Clipart': 0.06943,
            'Product': 0.00383,
            'Real World': 0.04968
        },
        "SOURCE_LOSSES_CR_ENT": {
            'Art': 0.05358,
            'Clipart': 0.22513,
            'Product': 0.01883,
            'Real World': 0.132
        },
        "TEST_SET_SIZES": {'Art': 486, 'Clipart': 873, 'Product': 888, 'Real World': 872},
        "D_PRECOMP_PATH": "/data/nogaz/Convex_bounds_optimization/LatentFlow_Pixel_Experiments/results/D_Matrix_FINAL_GMM_Soft_OfficeHome.npy"
    },
    "OFFICE31": {
        "DOMAINS": ['amazon', 'dslr', 'webcam'],
        "CLASSES": 31,
        "INPUT_DIM": 2048,
        "SOURCE_LOSSES_01": {'amazon': 0.02855, 'dslr': 1e-05, 'webcam': 0.00018},
        "SOURCE_LOSSES_CR_ENT": {'amazon': 0.07293, 'dslr': 0.13897, 'webcam': 0.02134},
        "TEST_SET_SIZES": {'amazon': 2197, 'dslr': 281, 'webcam': 578},
        "D_PRECOMP_PATH": "/data/nogaz/Convex_bounds_optimization/LatentFlow_Pixel_Experiments/results/D_Matrix_FINAL_GMM_Soft.npy"
    },
    "DOMAINNET": {
        "DOMAINS": ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"],
        "CLASSES": 345,
        "INPUT_DIM": 2048,
        "SOURCE_LOSSES_01": {
            'clipart': 0.41511,
            'infograph': 0.78587,
            'painting': 0.32533,
            'quickdraw': 0.30616,
            'real': 0.13799,
            'sketch': 0.37851
        },
        "SOURCE_LOSSES_CR_ENT": {
            'clipart': 2.30216,
            'infograph': 5.11777,
            'painting': 1.90294,
            'quickdraw': 1.26597,
            'real': 0.77012,
            'sketch': 2.13864
        },
        "TEST_SET_SIZES": {
            'clipart': 14604,
            'infograph': 15582,
            'painting': 21850,
            'quickdraw': 51750,
            'real': 52041,
            'sketch': 20916
        },
        "D_PRECOMP_PATH": "/data/nogaz/Convex_bounds_optimization/LatentFlow_Pixel_Experiments/results/D_Matrix_Optimized_DomainNet_ALL.npy"
    }
}

# Retained from original file for compatibility / future reuse.
# Not used in the new single-target setting.
TARGET_RATIOS_CONFIG = {
    ('amazon', 'dslr'): {'amazon': 0.20, 'dslr': 0.80},
    ('amazon', 'webcam'): {'amazon': 0.75, 'webcam': 0.25},
    ('dslr', 'webcam'): {'dslr': 0.30, 'webcam': 0.70},
    ('amazon', 'dslr', 'webcam'): {'amazon': 0.15, 'dslr': 0.65, 'webcam': 0.20},

    ('Art', 'Clipart'): {'Art': 0.80, 'Clipart': 0.20},
    ('Art', 'Product'): {'Art': 0.25, 'Product': 0.75},
    ('Art', 'Real World'): {'Art': 0.70, 'Real World': 0.30},
    ('Clipart', 'Product'): {'Clipart': 0.15, 'Product': 0.85},
    ('Clipart', 'Real World'): {'Clipart': 0.35, 'Real World': 0.65},
    ('Product', 'Real World'): {'Product': 0.80, 'Real World': 0.20},
    ('Art', 'Clipart', 'Product'): {'Art': 0.50, 'Clipart': 0.10, 'Product': 0.40},
    ('Art', 'Clipart', 'Real World'): {'Art': 0.15, 'Clipart': 0.70, 'Real World': 0.15},
    ('Art', 'Product', 'Real World'): {'Art': 0.60, 'Product': 0.20, 'Real World': 0.20},
    ('Clipart', 'Product', 'Real World'): {'Clipart': 0.10, 'Product': 0.30, 'Real World': 0.60},
    ('Art', 'Clipart', 'Product', 'Real World'): {'Art': 0.40, 'Clipart': 0.10, 'Product': 0.10, 'Real World': 0.40},

    ('clipart', 'infograph'): {'clipart': 0.3745, 'infograph': 0.6255},
    ('clipart', 'painting'): {'clipart': 0.7320, 'painting': 0.2680},
    ('clipart', 'quickdraw'): {'clipart': 0.1560, 'quickdraw': 0.8440},
    ('clipart', 'real'): {'clipart': 0.0581, 'real': 0.9419},
    ('clipart', 'sketch'): {'clipart': 0.8662, 'sketch': 0.1338},
    ('infograph', 'painting'): {'infograph': 0.0206, 'painting': 0.9794},
    ('infograph', 'quickdraw'): {'infograph': 0.8324, 'quickdraw': 0.1676},
    ('infograph', 'real'): {'infograph': 0.2123, 'real': 0.7877},
    ('infograph', 'sketch'): {'infograph': 0.1818, 'sketch': 0.8182},
    ('painting', 'quickdraw'): {'painting': 0.1834, 'quickdraw': 0.8166},
    ('painting', 'real'): {'painting': 0.3042, 'real': 0.6958},
    ('painting', 'sketch'): {'painting': 0.5247, 'sketch': 0.4753},
    ('quickdraw', 'real'): {'quickdraw': 0.4320, 'real': 0.5680},
    ('quickdraw', 'sketch'): {'quickdraw': 0.2912, 'sketch': 0.7088},
    ('real', 'sketch'): {'real': 0.6119, 'sketch': 0.3881},

    ('clipart', 'infograph', 'painting'): {'clipart': 0.0772, 'infograph': 0.8039, 'painting': 0.1189},
    ('clipart', 'infograph', 'quickdraw'): {'clipart': 0.5488, 'infograph': 0.2809, 'quickdraw': 0.1703},
    ('clipart', 'infograph', 'real'): {'clipart': 0.2140, 'infograph': 0.5300, 'real': 0.2560},
    ('clipart', 'infograph', 'sketch'): {'clipart': 0.1441, 'infograph': 0.6559, 'sketch': 0.2000},
    ('clipart', 'painting', 'quickdraw'): {'clipart': 0.2010, 'painting': 0.3340, 'quickdraw': 0.4650},
    ('clipart', 'painting', 'real'): {'clipart': 0.1111, 'painting': 0.1111, 'real': 0.7778},
    ('clipart', 'painting', 'sketch'): {'clipart': 0.4000, 'painting': 0.3000, 'sketch': 0.3000},
    ('clipart', 'quickdraw', 'real'): {'clipart': 0.2500, 'quickdraw': 0.6000, 'real': 0.1500},
    ('clipart', 'quickdraw', 'sketch'): {'clipart': 0.3500, 'quickdraw': 0.4500, 'sketch': 0.2000},
    ('clipart', 'real', 'sketch'): {'clipart': 0.1000, 'real': 0.8000, 'sketch': 0.1000},
    ('infograph', 'painting', 'quickdraw'): {'infograph': 0.0500, 'painting': 0.0500, 'quickdraw': 0.9000},
    ('infograph', 'painting', 'real'): {'infograph': 0.7000, 'painting': 0.2000, 'real': 0.1000},
    ('infograph', 'painting', 'sketch'): {'infograph': 0.3334, 'painting': 0.3333, 'sketch': 0.3333},
    ('infograph', 'quickdraw', 'real'): {'infograph': 0.8000, 'quickdraw': 0.1500, 'real': 0.0500},
    ('infograph', 'quickdraw', 'sketch'): {'infograph': 0.2000, 'quickdraw': 0.2000, 'sketch': 0.6000},
    ('infograph', 'real', 'sketch'): {'infograph': 0.1500, 'real': 0.7500, 'sketch': 0.1000},
    ('painting', 'quickdraw', 'real'): {'painting': 0.4000, 'quickdraw': 0.1000, 'real': 0.5000},
    ('painting', 'quickdraw', 'sketch'): {'painting': 0.5000, 'quickdraw': 0.2500, 'sketch': 0.2500},
    ('painting', 'real', 'sketch'): {'painting': 0.1000, 'real': 0.6000, 'sketch': 0.3000},
    ('quickdraw', 'real', 'sketch'): {'quickdraw': 0.6000, 'real': 0.2000, 'sketch': 0.2000},

    ('clipart', 'infograph', 'painting', 'quickdraw'): {'clipart': 0.0573, 'infograph': 0.0478, 'painting': 0.6738, 'quickdraw': 0.2211},
    ('clipart', 'infograph', 'painting', 'real'): {'clipart': 0.1000, 'infograph': 0.2000, 'painting': 0.5000, 'real': 0.2000},
    ('clipart', 'infograph', 'painting', 'sketch'): {'clipart': 0.4000, 'infograph': 0.1000, 'painting': 0.3000, 'sketch': 0.2000},
    ('clipart', 'infograph', 'quickdraw', 'real'): {'clipart': 0.1500, 'infograph': 0.4500, 'quickdraw': 0.1500, 'real': 0.2500},
    ('clipart', 'infograph', 'quickdraw', 'sketch'): {'clipart': 0.3000, 'infograph': 0.3000, 'quickdraw': 0.3000, 'sketch': 0.1000},
    ('clipart', 'infograph', 'real', 'sketch'): {'clipart': 0.2000, 'infograph': 0.2000, 'real': 0.4000, 'sketch': 0.2000},
    ('clipart', 'painting', 'quickdraw', 'real'): {'clipart': 0.0500, 'painting': 0.8500, 'quickdraw': 0.0500, 'real': 0.0500},
    ('clipart', 'painting', 'quickdraw', 'sketch'): {'clipart': 0.1250, 'painting': 0.1250, 'quickdraw': 0.5000, 'sketch': 0.2500},
    ('clipart', 'painting', 'real', 'sketch'): {'clipart': 0.2500, 'painting': 0.2500, 'real': 0.2500, 'sketch': 0.2500},
    ('clipart', 'quickdraw', 'real', 'sketch'): {'clipart': 0.6000, 'quickdraw': 0.1000, 'real': 0.1000, 'sketch': 0.2000},
    ('infograph', 'painting', 'quickdraw', 'real'): {'infograph': 0.1000, 'painting': 0.1000, 'quickdraw': 0.1000, 'real': 0.7000},
    ('infograph', 'painting', 'quickdraw', 'sketch'): {'infograph': 0.1500, 'painting': 0.3500, 'quickdraw': 0.3500, 'sketch': 0.1500},
    ('infograph', 'painting', 'real', 'sketch'): {'infograph': 0.4000, 'painting': 0.1000, 'real': 0.1000, 'sketch': 0.4000},
    ('infograph', 'quickdraw', 'real', 'sketch'): {'infograph': 0.3000, 'quickdraw': 0.4000, 'real': 0.2000, 'sketch': 0.1000},
    ('painting', 'quickdraw', 'real', 'sketch'): {'painting': 0.2000, 'quickdraw': 0.2000, 'real': 0.3000, 'sketch': 0.3000},

    ('clipart', 'infograph', 'painting', 'quickdraw', 'real'): {'clipart': 0.0768, 'infograph': 0.1340, 'painting': 0.0764, 'quickdraw': 0.3346, 'real': 0.3782},
    ('clipart', 'infograph', 'painting', 'quickdraw', 'sketch'): {'clipart': 0.1500, 'infograph': 0.2500, 'painting': 0.1000, 'quickdraw': 0.1500, 'sketch': 0.3500},
    ('clipart', 'infograph', 'painting', 'real', 'sketch'): {'clipart': 0.1000, 'infograph': 0.4000, 'painting': 0.2000, 'real': 0.1000, 'sketch': 0.2000},
    ('clipart', 'infograph', 'quickdraw', 'real', 'sketch'): {'clipart': 0.3000, 'infograph': 0.1500, 'quickdraw': 0.2000, 'real': 0.2000, 'sketch': 0.1500},
    ('clipart', 'painting', 'quickdraw', 'real', 'sketch'): {'clipart': 0.2500, 'painting': 0.1000, 'quickdraw': 0.1000, 'real': 0.2500, 'sketch': 0.3000},
    ('infograph', 'painting', 'quickdraw', 'real', 'sketch'): {'infograph': 0.0500, 'painting': 0.4500, 'quickdraw': 0.2000, 'real': 0.1500, 'sketch': 0.1500},

    ('clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch'): {'clipart': 0.1023, 'infograph': 0.1528, 'painting': 0.0805, 'quickdraw': 0.3688, 'real': 0.1292, 'sketch': 0.1664}
}

CURRENT_CFG = CONFIGS[DATASET_MODE]
TEST_SET_SIZES = CURRENT_CFG["TEST_SET_SIZES"]
ALL_DOMAINS_LIST = CURRENT_CFG["DOMAINS"]
NUM_CLASSES = CURRENT_CFG["CLASSES"]
INPUT_DIM = CURRENT_CFG["INPUT_DIM"]

if CONSTRAINT_LOSS_TYPE == "01":
    SOURCE_CONSTRAINT_VALUES = CURRENT_CFG["SOURCE_LOSSES_01"]
elif CONSTRAINT_LOSS_TYPE == "ce":
    SOURCE_CONSTRAINT_VALUES = CURRENT_CFG["SOURCE_LOSSES_CR_ENT"]
else:
    raise ValueError(f"Unknown CONSTRAINT_LOSS_TYPE: {CONSTRAINT_LOSS_TYPE}")

# ==========================================
# --- OPTIONAL: Use precomputed Global D ---
# ==========================================
D_PRECOMP_PATH = CURRENT_CFG["D_PRECOMP_PATH"]

# Root directories (adjust if needed)
OFFICEHOME_DIR = "/data/nogaz/Convex_bounds_optimization/OfficeHome"


class FeatureExtractor(nn.Module):
    def __init__(self, original_model):
        super(FeatureExtractor, self).__init__()
        self.backbone = nn.Sequential(*list(original_model.children())[:-1])
        self.head = original_model.fc

    def forward(self, x):
        feats = torch.flatten(self.backbone(x), 1)
        return feats, self.head(feats)


def compute_domain_lengths(domains):
    """
    Computes the TOTAL number of samples (Train + Test) in each domain.
    Needed for correctly slicing the Global_D matrix.
    """
    lengths = {}

    for d in domains:
        if DATASET_MODE == 'DIGITS':
            try:
                ret = Data.get_data_loaders(d, seed=1)
                if isinstance(ret, tuple) or isinstance(ret, list):
                    l_train = len(ret[0].dataset)
                    l_test = len(ret[1].dataset)
                    lengths[d] = l_train + l_test
                else:
                    lengths[d] = len(ret.dataset)
            except Exception as e:
                print(f"Error computing length for {d}: {e}")
                lengths[d] = 0

        elif DATASET_MODE == "DOMAINNET":
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

            lengths[d] = len(train_ds) + len(test_ds)

        else:
            path = get_domain_path(d)
            if os.path.exists(path):
                ds = datasets.ImageFolder(path)
                lengths[d] = len(ds)
            else:
                lengths[d] = 0

    return lengths


def get_domain_path(domain_name: str) -> str:
    """
    Build path for a given domain based on DATASET_MODE.
    Adjust if your dataset layout differs.
    """
    if DATASET_MODE in ["OFFICE", "OFFICE224"]:
        p1 = os.path.join(OFFICEHOME_DIR, domain_name, "images")
        p2 = os.path.join(OFFICEHOME_DIR, domain_name)
        return p1 if os.path.exists(p1) else p2
    elif DATASET_MODE == "OFFICE31":
        OFFICE31_DIR = "/data/nogaz/Convex_bounds_optimization/Office-31"
        p1 = os.path.join(OFFICE31_DIR, domain_name, "images")
        p2 = os.path.join(OFFICE31_DIR, domain_name)
        return p1 if os.path.exists(p1) else p2
    elif DATASET_MODE == "DOMAINNET":
        DOMAINNET_DIR = "/data/nogaz/Bi-ATEN/dataset/domainnet"
        return os.path.join(DOMAINNET_DIR, domain_name)
    else:
        raise ValueError("Precomputed D slicing is implemented for OFFICE/OFFICE224/OFFICE31/DOMAINNET only.")


class TransformedSubset(torch.utils.data.Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
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


def make_loader(dataset, batch_size, shuffle=False, num_workers=4, pin_memory=True):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def evaluate_oracle_any_ce_lower_bound(H, Y, eps=1e-12):
    """
    Cross-entropy lower bound based on per-sample oracle-any:
    for each sample, take the maximal probability assigned by any source
    to the true class, then compute CE.

    H: (N, C, K)
    Y: (N, C) one-hot
    """
    H = np.asarray(H)
    Y = np.asarray(Y)

    y_true = Y.argmax(axis=1)
    true_class_probs = H[np.arange(len(y_true)), y_true, :]
    best_true_prob = np.max(true_class_probs, axis=1)

    ce_lb = -np.mean(np.log(np.clip(best_true_prob, eps, 1.0)))
    return ce_lb


def evaluate_preds_with_metric(preds, Y, metric):
    metric = metric.lower()

    if metric == "accuracy":
        return evaluate_predictions_accuracy(preds, Y)
    if metric == "error":
        return evaluate_predictions_error_rate(preds, Y)
    if metric == "ce":
        return evaluate_predictions_cross_entropy(preds, Y)

    raise ValueError(f"Unsupported metric='{metric}'")


def get_train_test_loaders_and_indices(domain: str, seed: int, batch_size=32, test_batch_size=16):
    if DATASET_MODE == 'DIGITS':
        train_loader, test_loader, _ = Data.get_data_loaders(
            domain,
            seed=seed,
            batch_size=batch_size,
            test_batch_size=test_batch_size,
        )
        n_train = len(train_loader.dataset)
        n_test = len(test_loader.dataset)
        return (
            train_loader,
            test_loader,
            np.arange(0, n_train),
            np.arange(n_train, n_train + n_test),
            n_train + n_test,
        )

    train_trans, test_trans = make_rgb_transforms(
        Data.get_config(domain)["size"] if DATASET_MODE == "DOMAINNET" else 224
    )

    if DATASET_MODE == "DOMAINNET":
        train_file = os.path.join(Data.DOMAINNET_PATH, f"{domain}_train.txt")
        test_file = os.path.join(Data.DOMAINNET_PATH, f"{domain}_test.txt")

        train_ds = Data.DomainNetSplitDataset(
            root=Data.DOMAINNET_PATH,
            split_file=train_file,
            transform=train_trans
        )
        test_ds = Data.DomainNetSplitDataset(
            root=Data.DOMAINNET_PATH,
            split_file=test_file,
            transform=test_trans
        )

        n_train = len(train_ds)
        n_test = len(test_ds)

        train_idx = np.arange(0, n_train)
        test_idx = np.arange(n_train, n_train + n_test)
        total_n = n_train + n_test

    elif DATASET_MODE == "OFFICE31":
        path = get_domain_path(domain)
        ds_full = datasets.ImageFolder(path)

        train_subset, test_subset = Data.get_office31_split(ds_full, domain, seed)

        train_idx = np.array(train_subset.indices)
        test_idx = np.array(test_subset.indices)

        train_ds = TransformedSubset(train_subset, train_trans)
        test_ds = TransformedSubset(test_subset, test_trans)

        n_train = len(train_ds)
        n_test = len(test_ds)
        total_n = len(ds_full)

    else:
        path = get_domain_path(domain)
        ds_full = datasets.ImageFolder(path)

        N = len(ds_full)
        rng = np.random.RandomState(seed)
        indices = rng.permutation(N)
        split_point = int(0.8 * N)

        train_idx = indices[:split_point]
        test_idx = indices[split_point:]

        train_ds = TransformedSubset(Subset(ds_full, train_idx), train_trans)
        test_ds = TransformedSubset(Subset(ds_full, test_idx), test_trans)

        n_train = len(train_ds)
        n_test = len(test_ds)
        total_n = n_train + n_test

    train_loader = make_loader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Data.DATA_LOADER_WORKERS if DATASET_MODE == "DOMAINNET" else 4,
        pin_memory=(Data.device.type == "cuda") if DATASET_MODE == "DOMAINNET" else True,
    )
    test_loader = make_loader(
        test_ds,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=Data.DATA_LOADER_WORKERS if DATASET_MODE == "DOMAINNET" else 4,
        pin_memory=(Data.device.type == "cuda") if DATASET_MODE == "DOMAINNET" else True,
    )

    return train_loader, test_loader, train_idx, test_idx, total_n


# ==========================================
# --- Core matrix builder (KDE path) ---
# ==========================================

def build_DP_model_Classes(data_loaders, data_size, source_domains, models, classifiers, normalize_factors,
                           vae_norm_stats):
    """
    This function is retained almost unchanged from the original file.

    IMPORTANT in the new single-target setting:
    - data_loaders will contain ONLY the target-domain loader(s)
    - source_domains are the remaining domains (the experts)
    """
    print(f"--- Building Matrices (Size: {data_size}) ---")
    C, K = NUM_CLASSES, len(source_domains)
    Y = np.zeros((data_size, C))
    D = np.zeros((data_size, C, K))
    H = np.zeros((data_size, C, K))

    i = 0
    for dom_name, loader in data_loaders:
        print(f" [DATA] Processing {dom_name} batch extraction...")
        for data, label in loader:
            data = data.to(device)
            data = fix_batch_resnet(data, DATASET_MODE)
            N = len(data)

            one_hot = np.zeros((N, C))
            one_hot[np.arange(N)[label < C], label[label < C]] = 1
            Y[i:i + N] = one_hot

            with torch.no_grad():
                for k, src in enumerate(source_domains):
                    if DATASET_MODE == 'DIGITS':
                        logits = classifiers[src](data)
                        vae_in = data.view(data.size(0), -1)
                    else:
                        feats, logits = classifiers[src](data)
                        min_v, max_v = vae_norm_stats[src]
                        vae_in = torch.clamp((feats - min_v) / (max_v - min_v + 1e-6), 0, 1)

                    H[i:i + N, :, k] = F.softmax(logits, dim=1).cpu().numpy()

                    x_hat, _, _ = models[src](vae_in)
                    log_p = models[src].compute_log_probabitility_bernoulli(x_hat, vae_in)
                    D[i:i + N, :, k] = np.tile(log_p[:, None].cpu().numpy(), (1, C))

            i += N
            torch.cuda.empty_cache()

    for k, src in enumerate(source_domains):
        print(f" [KDE] Starting GridSearch for {src}...")
        raw = D[:, 0, k].reshape(-1, 1)
        mean_v, std_v = normalize_factors[src]
        proc = (raw - mean_v.item()) / max(std_v.item(), 1e-4)
        bw_range = np.linspace(0.1, 0.5, 10)
        kde = GridSearchCV(KernelDensity(kernel='gaussian'), {'bandwidth': bw_range}, cv=3)
        kde.fit(proc[np.random.permutation(len(proc))[:min(2000, len(proc))]])
        print(f" [KDE DEBUG] {src:<10} | Best BW: {kde.best_estimator_.bandwidth:.3f}")
        scores = np.exp(kde.best_estimator_.score_samples(proc))
        D[:, :, k] = np.tile(scores.reshape(-1, 1), (1, C))
        print(f" -> Score Range: {scores.min():.4e} to {scores.max():.4e} | Mean: {scores.mean():.4f}")

    print("--- Normalizing Matrices ---")
    dom_idx = np.argmax(D[:, 0, :], axis=1)
    unique, counts = np.unique(dom_idx, return_counts=True)
    print("\n[DOMAIN DOMINANCE CHECK]")
    for idx, count in zip(unique, counts):
        print(f" Source {source_domains[idx]:<10} is strongest for {count:>5} samples ({100 * count / data_size:.1f}%)")
    print("-" * 50)

    D[D < (D.max(axis=2, keepdims=True) * 0.2)] = 0.0
    for k in range(K):
        if D[:, :, k].max() > 1e-9:
            D[:, :, k] /= D[:, :, k].max()

    return Y, D, H


# ==========================================
# --- Baselines / Solvers ---
# ==========================================

def evaluate_accuracy_wd(w, D, H, Y):
    preds = ((D * H) * w.reshape(1, 1, -1)).sum(axis=2)
    return accuracy_score(Y.argmax(axis=1), preds.argmax(axis=1)) * 100.0


def evaluate_accuracy_q(Q, H, Y):
    """
    Supports:
    1) Q shape (N, K)  -> sample-level weights
    2) Q shape (N*C, K)-> (sample,class)-level weights, reshape to (N, C, K)
    """
    Q = np.asarray(Q)
    N, C, K = H.shape
    if Q.shape == (N, K):
        preds = (H * Q[:, None, :]).sum(axis=2)
    elif Q.shape == (N * C, K):
        Q3 = Q.reshape(N, C, K)
        preds = (H * Q3).sum(axis=2)
    else:
        raise ValueError(f"Unexpected Q shape {Q.shape}, expected {(N, K)} or {(N * C, K)}")
    return accuracy_score(Y.argmax(axis=1), preds.argmax(axis=1)) * 100.0


def _simplex_grid_recursive(total_steps, dim, prefix=None):
    """
    Helper to enumerate all simplex grid points with coordinates multiples of 1/total_steps.
    Returns integer tuples summing to total_steps.
    """
    if prefix is None:
        prefix = []

    if dim == 1:
        yield tuple(prefix + [total_steps])
        return

    for v in range(total_steps + 1):
        yield from _simplex_grid_recursive(total_steps - v, dim - 1, prefix + [v])


def estimate_empirical_oracle_weights(Y, D, H, metric, grid_step=0.05):
    """
    SINGLE-TARGET oracle baseline:
    in the old mixture setting, ORACLE used the true target mixture weights.
    In the new single-target setting there is no true target mixture.
    Therefore we define ORACLE as the BEST convex mixture on the target labels
    among the source experts, evaluated on the current target split.

    This is a privileged baseline, analogous in spirit to the old ORACLE.
    """
    K = H.shape[2]

    if K == 1:
        return np.array([1.0]), evaluate_solution_with_w(np.array([1.0]), D, H, Y, metric=metric)

    step_count = int(round(1.0 / grid_step))
    candidates = []

    # exact simplex grid
    for ints in _simplex_grid_recursive(step_count, K):
        w = np.array(ints, dtype=float) / step_count
        candidates.append(w)

    # extra candidates
    candidates.append(np.ones(K) / K)
    for k in range(K):
        w = np.zeros(K)
        w[k] = 1.0
        candidates.append(w)

    best_w = None
    best_score = None

    for w in candidates:
        score = evaluate_solution_with_w(w, D, H, Y, metric=metric)
        if best_score is None:
            best_score = score
            best_w = w.copy()
        else:
            if metric == "accuracy":
                if score > best_score:
                    best_score = score
                    best_w = w.copy()
            else:
                if score < best_score:
                    best_score = score
                    best_w = w.copy()

    return best_w, best_score


def run_baselines(Y, D, H, source_domains, target_domain, all_source_domains, seed):
    print(" [Baseline] Running Oracle, Uniform, and Best Single Source...")
    buf = io.StringIO()

    metric_label = {
        "accuracy": "acc",
        "error": "err",
        "ce": "ce",
    }[EVAL_METRIC]

    # =====================================================
    # 1. ORACLE (single-target version) + UNIFORM
    # =====================================================
    uniform_w = np.ones(len(source_domains)) / len(source_domains)

    oracle_w, oracle_score = estimate_empirical_oracle_weights(
        Y=Y,
        D=D,
        H=H,
        metric=EVAL_METRIC,
        grid_step=0.05
    )
    uniform_score = evaluate_solution_with_w(uniform_w, D, H, Y, metric=EVAL_METRIC)

    buf.write(
        f"[ORACLE_INFO] single_target_mode=True | "
        f"oracle_definition=best_empirical_convex_mixture_on_target_labels\n"
    )

    if CONSTRAINT_LOSS_TYPE == "01":
        oracle_epsilon = 1.0 - (oracle_score / 100.0)
    else:
        oracle_epsilon = oracle_score

    buf.write(f"[ORACLE_EPSILON] constraint_type={CONSTRAINT_LOSS_TYPE} | oracle_epsilon={oracle_epsilon:.6f}\n")

    oracle_w_full = map_weights_to_full_source_list(oracle_w, source_domains, all_source_domains)
    uniform_w_full = map_weights_to_full_source_list(uniform_w, source_domains, all_source_domains)

    buf.write(
        f"{'ORACLE':<18} | {'N/A':<15} | {'N/A':<15} | "
        f"{oracle_score:<12.2f} | {str(np.round(oracle_w_full, 4))}\n"
    )
    buf.write(
        f"{'UNIFORM':<18} | {'N/A':<15} | {'N/A':<15} | "
        f"{uniform_score:<12.2f} | {str(np.round(uniform_w_full, 4))}\n"
    )

    print(f" >>> [Baseline] ORACLE  | {metric_label}: {oracle_score:.2f}")
    print(f" >>> [Baseline] UNIFORM | {metric_label}: {uniform_score:.2f}")

    print("BASELINE DEBUG")
    print("target_domain:", target_domain)
    print("source_domains:", source_domains)
    print("uniform_w local:", uniform_w)
    print("oracle_w local:", oracle_w)
    print("uniform_ce direct:", evaluate_solution_with_w(uniform_w, D, H, Y, metric="ce"))
    print("oracle_ce direct:", evaluate_solution_with_w(oracle_w, D, H, Y, metric="ce"))

    # =====================================================
    # 2. BEST_SINGLE_SRC
    # =====================================================
    # --- 2. Per-Source Expert Performance on Current Target ---
    for k, src_name in enumerate(source_domains):
        src_preds = H[:, :, k]  # (N, C)
        src_score = evaluate_preds_with_metric(src_preds, Y, EVAL_METRIC)

        w_single = np.zeros(len(source_domains))
        w_single[k] = 1.0
        w_f_single = map_weights_to_full_source_list(
            w_single, source_domains, all_source_domains
        )

        buf.write(
            f"{('SRC_' + src_name):<18} | {src_name:<15} | {'N/A':<15} | "
            f"{src_score:<12.2f} | {str(np.round(w_f_single, 4))}\n"
        )
        print(f" >>> [Baseline] SRC_{src_name} | {metric_label}: {src_score:.2f}")

    # =====================================================
    # 3. ORACLE_ANY_CORRECT / CE lower bound
    # =====================================================
    y_true = Y.argmax(axis=1)
    pred_per_source = H.argmax(axis=1)  # (N, K)
    any_correct = (pred_per_source == y_true[:, None]).any(axis=1)

    oracle_any_acc = any_correct.mean() * 100.0
    oracle_any_err = 1.0 - any_correct.mean()
    oracle_any_ce_lb = evaluate_oracle_any_ce_lower_bound(H, Y)

    if EVAL_METRIC == "accuracy":
        oracle_any_name = "ORACLE_ANY_CORRECT"
        oracle_any_score = oracle_any_acc
    elif EVAL_METRIC == "error":
        oracle_any_name = "ORACLE_ANY_CORRECT"
        oracle_any_score = oracle_any_err
    elif EVAL_METRIC == "ce":
        oracle_any_name = "ORACLE_ANY_CE_LB"
        oracle_any_score = oracle_any_ce_lb
    else:
        raise ValueError(f"Unsupported EVAL_METRIC={EVAL_METRIC}")

    buf.write(
        f"{oracle_any_name:<18} | {'N/A':<15} | {'N/A':<15} | "
        f"{oracle_any_score:<12.2f} | N/A\n"
    )
    buf.write(
        f"[ORACLE_ANY_DETAILS] acc={oracle_any_acc:.4f} | "
        f"err={oracle_any_err:.6f} | ce_lb={oracle_any_ce_lb:.6f}\n"
    )

    if EVAL_METRIC == "ce":
        print(f" >>> [Baseline] ORACLE_ANY_CE_LB | ce: {oracle_any_ce_lb:.6f}")
    else:
        print(f" >>> [Baseline] ORACLE_ANY_CORRECT | {metric_label}: {oracle_any_score:.2f}")

    # =====================================================
    # 4. DC Solver
    # =====================================================
    print(" [Baseline] Running DC Solver...")

    DC_NUM_JOBS = 2

    def _run_single_dc_restart(i):
        try:
            dp = init_problem_from_model_fast(Y, D, H, p=len(source_domains), C=NUM_CLASSES)
            slv = ConvexConcaveSolverFast(
                ConvexConcaveProblemFast(dp),
                seed + (i * 100),
                "err"
            )
            z_dc, _, _ = slv.solve()

            if z_dc is None:
                return None

            score = evaluate_solution_with_w(z_dc, D, H, Y, metric=EVAL_METRIC)
            return {
                "seed_idx": i,
                "score": score,
                "z": z_dc,
            }

        except Exception:
            return None

    dc_results = Parallel(n_jobs=DC_NUM_JOBS, backend="loky")(
        delayed(_run_single_dc_restart)(i) for i in range(5)
    )

    dc_results = [r for r in dc_results if r is not None]

    if dc_results:
        dc_scores = [r["score"] for r in dc_results]

        if EVAL_METRIC == "accuracy":
            best_result = max(dc_results, key=lambda r: r["score"])
        else:
            best_result = min(dc_results, key=lambda r: r["score"])

        best_dc_score = best_result["score"]
        best_z_dc = best_result["z"]

        avg_res = f"{np.mean(dc_scores):.2f}±{np.std(dc_scores):.2f}"
        w_f = map_weights_to_full_source_list(best_z_dc, source_domains, all_source_domains)
        buf.write(
            f"{'DC (5-Seeds)':<18} | {'N/A':<15} | {'N/A':<15} | "
            f"{avg_res:<12} | {str(np.round(w_f, 4))}\n"
        )

    return buf.getvalue(), oracle_epsilon, oracle_w


def run_solver_sweep_worker(
    Y, D, H, eps_mult, source_domains, all_source_domains, save_dir,
    strategy_name=None, target_domain=None, oracle_epsilon=None,
):
    print(f" [Worker] Starting sweep for Epsilon Mult: {eps_mult}")
    buf = io.StringIO()

    errors = np.array([
        (SOURCE_CONSTRAINT_VALUES.get(d, 0.1) + 0.05) * eps_mult
        for d in source_domains
    ])

    old_epsilon = float(np.max(errors))

    if USE_ORACLE_EPSILON and oracle_epsilon is not None:
        epsilon_used = float(oracle_epsilon) * eps_mult
    else:
        epsilon_used = old_epsilon

    oracle_eps_str = f"{oracle_epsilon:.6f}" if oracle_epsilon is not None else "None"

    buf.write(
        f"[EPSILON_INFO] eps_mult={eps_mult} | "
        f"old_epsilon={old_epsilon:.6f} | "
        f"oracle_epsilon={oracle_eps_str} | "
        f"epsilon_used={epsilon_used:.6f}\n"
    )

    max_ent = np.log(len(source_domains)) if len(source_domains) > 1 else 0.1

    SOLVERS = ["PER_DOMAIN", "6", "6A"]  # retained from your current config
    BACKENDS = ["SCS"]
    ETAS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    DELTA_MULTS = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    BETAS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]

    metric_label = {
        "accuracy": "acc",
        "error": "err",
        "ce": "ce",
    }[EVAL_METRIC]

    # =========================================================
    # Solvers that depend on DELTA
    # =========================================================
    delta_solvers = [s for s in SOLVERS if s in ["CVXPY_GLOBAL", "PER_DOMAIN"]]

    for solver in delta_solvers:
        for delta_mult in DELTA_MULTS:
            delta = float(delta_mult * max_ent)

            for backend in BACKENDS:
                try:
                    if solver == "CVXPY_GLOBAL":
                        w, Q = solve_convex_problem_mosek(
                            Y, D, H,
                            delta=delta,
                            epsilon=float(epsilon_used),
                            solver_type=backend,
                            loss_type=CONSTRAINT_LOSS_TYPE,
                        )

                    elif solver == "PER_DOMAIN":
                        w, Q = solve_convex_problem_per_domain(
                            Y, D, H,
                            delta=np.full(len(source_domains), delta),
                            epsilon=epsilon_used,
                            solver_type=backend,
                            loss_type=CONSTRAINT_LOSS_TYPE,
                        )
                    else:
                        continue

                    if w is None or Q is None:
                        buf.write(
                            f"[{solver}/{backend}] Returned None (likely infeasible) "
                            f"| delta_mult={delta_mult} | delta={delta:.6f}\n"
                        )
                        continue

                    score_w = evaluate_solution_with_w(w, D, H, Y, metric=EVAL_METRIC)
                    score_q = evaluate_solution_with_q(Q, H, Y, metric=EVAL_METRIC)
                    w_f = map_weights_to_full_source_list(w, source_domains, all_source_domains)

                    buf.write(
                        f"{solver:<12} | {backend:<5} | m:{eps_mult:<4} "
                        f"| delta_mult:{delta_mult:<5} | delta:{delta:<8.6f} "
                        f"| {metric_label}_W: {score_w:>6.2f} | {metric_label}_Q: {score_q:>6.2f} "
                        f"| W: {np.array2string(w_f, precision=6)}\n"
                    )

                except Exception as e:
                    tb = traceback.format_exc()
                    msg = (
                        f"[{solver}/{backend}] ERROR eps_mult={eps_mult} "
                        f"delta_mult={delta_mult} delta={delta:.6f}: {repr(e)}\n{tb}\n"
                    )
                    print(msg)
                    buf.write(msg)

    # =========================================================
    # Solvers that depend on ETA
    # =========================================================
    eta_solver_map = {
        "6": solve_convex_problem_soft_kl_diagonal_fast,
        "6A": solve_convex_problem_soft_kl_diagonal_fast_6A,
    }
    beta_solvers = {"6", "6A"}

    eta_solvers = [s for s in SOLVERS if s in eta_solver_map]

    for solver in eta_solvers:
        solver_fn = eta_solver_map[solver]

        for eta in ETAS:
            for backend in BACKENDS:
                if solver in beta_solvers:
                    for beta in BETAS:
                        try:
                            out = solver_fn(
                                Y, D, H,
                                epsilon=float(epsilon_used),
                                eta=eta,
                                beta=beta,
                                solver_type=backend,
                                loss_type=CONSTRAINT_LOSS_TYPE,
                            )

                            if solver == "6A":
                                w, Q, R, s_val = out
                                if w is None or Q is None or R is None:
                                    msg = (
                                        f"[{solver}/{backend}] Returned None (likely infeasible) "
                                        f"| eps_mult={eps_mult} | eta={eta} | beta={beta}\n"
                                    )
                                    print(msg, end="")
                                    buf.write(msg)
                                    continue
                            else:
                                w, Q, R = out
                                s_val = None
                                if w is None or Q is None or R is None:
                                    msg = (
                                        f"[{solver}/{backend}] Returned None (likely infeasible) "
                                        f"| eps_mult={eps_mult} | eta={eta} | beta={beta}\n"
                                    )
                                    print(msg, end="")
                                    buf.write(msg)
                                    continue

                            score_w = evaluate_solution_with_w(w, D, H, Y, metric=EVAL_METRIC)
                            score_q = evaluate_solution_with_q(Q, H, Y, metric=EVAL_METRIC)
                            w_f = map_weights_to_full_source_list(w, source_domains, all_source_domains)

                            extra = f" | s:{s_val:.6f}" if s_val is not None else ""
                            msg = (
                                f"{solver:<12} | {backend:<5} | m:{eps_mult:<4} "
                                f"| eta:{eta:<8} | beta:{beta:<8} "
                                f"| {metric_label}_W: {score_w:>6.2f} | {metric_label}_Q: {score_q:>6.2f}"
                                f"{extra} | W: {np.array2string(w_f, precision=6)}\n"
                            )
                            print(msg, end="")
                            buf.write(msg)

                        except Exception as e:
                            tb = traceback.format_exc()
                            msg = (
                                f"[{solver}/{backend}] ERROR "
                                f"eps_mult={eps_mult} eta={eta} beta={beta}: "
                                f"{repr(e)}\n{tb}\n"
                            )
                            print(msg, end="")
                            buf.write(msg)
    return buf.getvalue()


# ============================================================
# --- Build Y,H,D using precomputed D for SINGLE TARGET ---
# ============================================================

def build_YDH_single_target_with_precomputed_D(
    target_domain,
    source_domains,
    seed,
    classifiers,
    Global_D,
    domain_lengths,
):
    """
    SINGLE-TARGET version:
    Build Y,H from the target-domain TEST loader only.
    D is sliced from the global precomputed matrix using:
      - rows from the target domain
      - columns corresponding to source_domains
    """
    C = NUM_CLASSES
    K = len(source_domains)

    _, teL, tr_idx, te_idx, Ndom = get_train_test_loaders_and_indices(
        target_domain,
        seed=seed,
        batch_size=32,
        test_batch_size=16
    )

    data_size = len(teL.dataset)
    Y = np.zeros((data_size, C))
    H = np.zeros((data_size, C, K))
    D_loaded = np.zeros((data_size, K))

    D_dom_full = slice_global_D_for_domain(Global_D, ALL_DOMAINS_LIST, domain_lengths, target_domain)
    col_idx = [ALL_DOMAINS_LIST.index(s) for s in source_domains]
    D_dom_full = D_dom_full[:, col_idx]
    D_dom = D_dom_full[te_idx]

    i = 0
    cursor = 0
    for data, label in teL:
        data = data.to(device)
        data = fix_batch_resnet(data, DATASET_MODE)
        N = len(data)

        one_hot = np.zeros((N, C))
        one_hot[np.arange(N)[label < C], label[label < C]] = 1
        Y[i:i + N] = one_hot

        with torch.no_grad():
            for k, src in enumerate(source_domains):
                if DATASET_MODE == 'DIGITS':
                    logits = classifiers[src](data)
                else:
                    feats, logits = classifiers[src](data)
                H[i:i + N, :, k] = F.softmax(logits, dim=1).cpu().numpy()

        D_loaded[i:i + N, :] = D_dom[cursor:cursor + N, :]
        cursor += N
        i += N
        torch.cuda.empty_cache()

    D = np.tile(D_loaded[:, None, :], (1, C, 1))
    return Y, D, H


# ============================================================
# --- Retained helper from original file ---
# ============================================================

def build_YDH_with_precomputed_D(target_domains, seed, classifiers, Global_D, domain_lengths):
    """
    Retained from the original file for compatibility.
    Not used in the new single-target setting.
    """
    C = NUM_CLASSES
    K = len(target_domains)

    loaders = []
    split_info = {}
    for d in target_domains:
        _, teL, tr_idx, te_idx, Ndom = get_train_test_loaders_and_indices(d, seed=seed, batch_size=64, test_batch_size=64)
        loaders.append((d, teL))
        split_info[d] = {"te_idx": te_idx, "N": Ndom}

    data_size = sum(len(l.dataset) for _, l in loaders)
    Y = np.zeros((data_size, C))
    H = np.zeros((data_size, C, K))
    D_loaded = np.zeros((data_size, K))

    i = 0
    for dom_name, loader in loaders:
        D_dom_full = slice_global_D_for_domain(Global_D, ALL_DOMAINS_LIST, domain_lengths, dom_name)
        col_idx = [ALL_DOMAINS_LIST.index(s) for s in target_domains]
        D_dom_full = D_dom_full[:, col_idx]
        te_idx = split_info[dom_name]["te_idx"]
        D_dom = D_dom_full[te_idx]

        cursor = 0
        for data, label in loader:
            data = data.to(device)
            data = fix_batch_resnet(data, DATASET_MODE)
            N = len(data)

            one_hot = np.zeros((N, C))
            one_hot[np.arange(N)[label < C], label[label < C]] = 1
            Y[i:i + N] = one_hot

            with torch.no_grad():
                for k, src in enumerate(target_domains):
                    if DATASET_MODE == 'DIGITS':
                        logits = classifiers[src](data)
                    else:
                        feats, logits = classifiers[src](data)
                    H[i:i + N, :, k] = F.softmax(logits, dim=1).cpu().numpy()

            D_loaded[i:i + N, :] = D_dom[cursor:cursor + N, :]
            cursor += N
            i += N
            torch.cuda.empty_cache()

    D = np.tile(D_loaded[:, None, :], (1, C, 1))
    return Y, D, H


def apply_custom_ratios(Y, D, H, target_domains, custom_ratios_dict):
    """
    Retained from the original file for compatibility.
    Not used in the new single-target setting.
    """
    current_sizes = [TEST_SET_SIZES[d] for d in target_domains]
    requested_ratios = [custom_ratios_dict[d] for d in target_domains]

    max_total_N = int(min([actual / ratio for actual, ratio in zip(current_sizes, requested_ratios)]))
    max_total_N = min(max_total_N, 10000)
    new_indices = []
    offset = 0
    for i, dom in enumerate(target_domains):
        n_to_take = int(max_total_N * requested_ratios[i])
        new_indices.extend(range(offset, offset + n_to_take))
        offset += current_sizes[i]

    print(f" [Subsample] Adjusting: New N={len(new_indices)} | Ratios: {requested_ratios}")
    return Y[new_indices], D[new_indices], H[new_indices]


def run_solver_job_flat(
    Y, D, H,
    eps_mult,
    source_domains,
    target_domain,
    all_source_domains,
    qstar_save_dir,
    strategy_name,
    oracle_epsilon,
):
    result_text = run_solver_sweep_worker(
        Y=Y,
        D=D,
        H=H,
        eps_mult=eps_mult,
        source_domains=source_domains,
        all_source_domains=all_source_domains,
        save_dir=qstar_save_dir,
        strategy_name=strategy_name,
        target_domain=target_domain,
        oracle_epsilon=oracle_epsilon,
    )
    return strategy_name, eps_mult, result_text


def task_run(classifiers, all_source_domains):
    seed = SEED
    torch.manual_seed(seed)
    np.random.seed(seed)
    import gc

    test_path = f'./results_{DATASET_MODE}_mosek/seed_{seed}/'
    os.makedirs(test_path, exist_ok=True)

    qstar_save_dir = os.path.join(test_path, "saved_qstars")
    os.makedirs(qstar_save_dir, exist_ok=True)

    if USE_PRECOMPUTED_D is False:
        models, normalize_factors, vae_norm_stats = handle_vae_models(
            all_source_domains, classifiers, seed
        )

    Global_D = None
    domain_lengths = None
    if USE_PRECOMPUTED_D and os.path.exists(D_PRECOMP_PATH):
        Global_D = load_global_D_matrix(D_PRECOMP_PATH)
        if Global_D is not None:
            domain_lengths = compute_domain_lengths(all_source_domains)

    with open(os.path.join(test_path, FILENAME), 'a') as fp:
        # =====================================================
        # NEW SINGLE-TARGET LOOP:
        # target = one domain
        # sources = all remaining domains
        # =====================================================
        for target_domain in all_source_domains:

            if USE_ALL_SOURCES_FOR_SOLVER:
                source_domains = list(all_source_domains)
                source_mode = "ALL_SOURCES_INCLUDING_TARGET"
            else:
                source_domains = [d for d in all_source_domains if d != target_domain]
                source_mode = "LEAVE_TARGET_OUT"

            if len(source_domains) == 0:
                print(f"⚠️ Skipping target={target_domain}: no source domains available.")
                continue

            print(f"\n[TARGET] Processing single target: {target_domain}")
            print(f"[SOURCE_MODE] {source_mode}")
            print(f"[SOURCES] {source_domains}")
            # -------------------------------------------------
            # Build full tensors ONCE per target
            # -------------------------------------------------
            if Global_D is not None and domain_lengths is not None and USE_PRECOMPUTED_D:
                Y_full, D_full, H_full = build_YDH_single_target_with_precomputed_D(
                    target_domain=target_domain,
                    source_domains=source_domains,
                    seed=seed,
                    classifiers=classifiers,
                    Global_D=Global_D,
                    domain_lengths=domain_lengths,
                )
            else:
                _, target_test_loader, _, _, _ = get_train_test_loaders_and_indices(
                    target_domain,
                    seed=seed,
                    batch_size=32,
                    test_batch_size=16
                )

                el = [(target_domain, target_test_loader)]
                Y_full, D_full, H_full = build_DP_model_Classes(
                    el,
                    sum(len(l.dataset) for _, l in el),
                    source_domains,
                    models,
                    classifiers,
                    normalize_factors,
                    vae_norm_stats,
                )

            torch.cuda.empty_cache()
            gc.collect()

            # -------------------------------------------------
            # Keep the "3 strategy blocks" structure for minimal
            # changes to the rest of the file, but in single-target
            # mode there is no target-mixture subsampling.
            # So all three strategies use the same tensors.
            # -------------------------------------------------
            weight_sets = [
                ("SINGLE_TARGET", None),
            ]

            flat_jobs = []
            strategy_blocks = {}
            strategy_order = []

            for strategy_name, _ in weight_sets:
                print(f"  -> Strategy: {strategy_name}")
                strategy_order.append(strategy_name)

                Y, D, H = Y_full, D_full, H_full

                header_buf = io.StringIO()
                header_buf.write(
                    f"\n{'=' * 100}\n"
                    f"TARGET: {target_domain} | STRATEGY: {strategy_name}\n"
                    f"SOURCE_MODE: {source_mode}\n"
                    f"SOURCES: {source_domains}\n"
                    f"{'=' * 100}\n"
                )
                baseline_text, oracle_epsilon, oracle_w = run_baselines(
                    Y=Y,
                    D=D,
                    H=H,
                    source_domains=source_domains,
                    target_domain=target_domain,
                    all_source_domains=all_source_domains,
                    seed=seed
                )
                header_buf.write(baseline_text)

                strategy_blocks[strategy_name] = {
                    "prefix_text": header_buf.getvalue(),
                    "solver_texts": {},
                }

                for eps_mult in [1.0, 2.0, 3.0]:
                    flat_jobs.append(
                        delayed(run_solver_job_flat)(
                            Y=Y,
                            D=D,
                            H=H,
                            eps_mult=eps_mult,
                            source_domains=source_domains,
                            target_domain=target_domain,
                            all_source_domains=all_source_domains,
                            qstar_save_dir=qstar_save_dir,
                            strategy_name=strategy_name,
                            oracle_epsilon=oracle_epsilon,
                        )
                    )

            del Y_full, D_full, H_full
            torch.cuda.empty_cache()
            gc.collect()

            results = Parallel(n_jobs=4, verbose=5)(flat_jobs)

            for strategy_name, eps_mult, result_text in results:
                strategy_blocks[strategy_name]["solver_texts"][eps_mult] = result_text

            for strategy_name in strategy_order:
                fp.write(strategy_blocks[strategy_name]["prefix_text"])
                for eps_mult in [1.0, 2.0, 3.0]:
                    fp.write(strategy_blocks[strategy_name]["solver_texts"][eps_mult])

            fp.flush()
            gc.collect()

            print(f"✅ Finished single-target run for: {target_domain}")


def handle_vae_models(all_source_domains, classifiers, seed):
    models, normalize_factors, vae_norm_stats = {}, {}, {}
    ap, an = (0.5, -0.5) if DATASET_MODE == "OFFICE31" else (2.0, -2.0)
    print("\n--- Initializing Models for all Domains ---")
    for d in all_source_domains:
        m = vr_model(INPUT_DIM, ap, an).to(device)
        pattern = f"./models_{d}_seed{seed}_224_features/vrs_*_model.pt"
        matches = glob.glob(pattern)
        if not matches:
            pattern = f"./models_{d}_seed{seed}_*/vrs_*_model.pt"
            matches = glob.glob(pattern)
        path = matches[0]
        m.load_state_dict(torch.load(path, map_location=device))
        models[d] = m.eval()
        loader, _, _ = Data.get_data_loaders(d, seed=seed)
        feats = []
        with torch.no_grad():
            for j, (imgs, _) in enumerate(loader):
                if j > 10:
                    break
                if DATASET_MODE == 'DIGITS':
                    f = imgs.view(imgs.size(0), -1).to(device)
                else:
                    batch_fixed = fix_batch_resnet(imgs.to(device))
                    f, _ = classifiers[d](batch_fixed)
                feats.append(f.cpu())

        cat_f = torch.cat(feats, 0).to(device)
        vae_norm_stats[d] = (cat_f.min(), cat_f.max())
        vae_in = torch.clamp((cat_f - cat_f.min()) / (cat_f.max() - cat_f.min() + 1e-6), 0, 1)
        out, _, _ = m(vae_in)
        lp = m.compute_log_probabitility_bernoulli(out, vae_in)
        normalize_factors[d] = (lp.mean(), lp.std())
        print(f"✅ Loaded VRS for {d}")
    return models, normalize_factors, vae_norm_stats


def get_actual_test_set_sizes(domains):
    """
    Iterates through the DomainNet test split files and counts the
    actual number of samples to fill the TEST_SET_SIZES dictionary.
    """
    actual_sizes = {}
    print("--> Calculating actual test set sizes for DomainNet...")
    for d in domains:
        test_file = os.path.join(Data.DOMAINNET_PATH, f"{d}_test.txt")
        if os.path.exists(test_file):
            test_ds = Data.DomainNetSplitDataset(
                root=Data.DOMAINNET_PATH,
                split_file=test_file,
                transform=None
            )
            actual_sizes[d] = len(test_ds)
        else:
            print(f"  [WARNING] Test file not found for {d}: {test_file}")
            actual_sizes[d] = 0
    return actual_sizes


def main():
    print(f"Starting Main Process | Dataset Mode: {DATASET_MODE}")
    classifiers = {}
    n_gpus = torch.cuda.device_count()
    if n_gpus > 1:
        print(f"🚀 Detected {n_gpus} GPUs! Enabling DataParallel for feature extraction.")

    # for d in ALL_DOMAINS_LIST:
    #     p1 = f"./classifiers/{d}_224.pt"
    #     p2 = f"./classifiers/{d}_classifier.pt"
    #     path = p1 if os.path.exists(p1) else p2
    #
    #     if not os.path.exists(path):
    #         print(f"❌ Warning: Classifier for {d} not found in ./classifiers/")
    #         continue

    for d in ALL_DOMAINS_LIST:
        p1 = os.path.join(CLASSIFIERS_DIR, f"{d}_224.pt")
        p2 = os.path.join(CLASSIFIERS_DIR, f"{d}_classifier.pt")
        path = p1 if os.path.exists(p1) else p2

        if not os.path.exists(path):
            print(f"❌ Warning: Classifier for {d} not found in {CLASSIFIERS_DIR}/")
            print(f"   Tried: {p1}")
            print(f"   Tried: {p2}")
            continue

        if DATASET_MODE == 'DIGITS':
            m = ClSFR.Grey_32_64_128_gp()
            print(f"Loading DIGITS model ({d}) from: {path}")
            m.load_state_dict(torch.load(path, map_location=device))
            if n_gpus > 1:
                m = nn.DataParallel(m)
            m = m.to(device).eval()
            classifiers[d] = m
        else:
            m = models.resnet50(weights=None)
            m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
            m.load_state_dict(torch.load(path, map_location=device))
            extractor = FeatureExtractor(m)
            if n_gpus > 1:
                extractor = nn.DataParallel(extractor)
            classifiers[d] = extractor.to(device).eval()

        print(f"✅ Loaded Classifier: {d}")

    missing = [d for d in ALL_DOMAINS_LIST if d not in classifiers]
    if missing:
        raise RuntimeError(
            f"Missing classifiers for domains={missing} in {CLASSIFIERS_DIR}. "
            f"Do not run MSA with partial classifiers."
        )

    task_run(classifiers, ALL_DOMAINS_LIST)


# --------------------------------------------
# Retained debugging utility from original file
# --------------------------------------------
import numpy as np

from loss_functions import (
    combine_predictions_with_w,
    evaluate_predictions_cross_entropy,
)

def debug_ce_mixture_case(
    Y,
    D,
    H,
    source_domains,
    oracle_w,
    uniform_w,
    dc_w,
    solver_w=None,
    single_source_name="webcam",
    top_k_samples=10,
    eps=1e-12,
):
    """
    Explicit sanity-check / explanation for CE gaps between:
    - BEST_SINGLE_SRC (raw source only)
    - BEST_SINGLE via WD with one-hot w
    - ORACLE
    - UNIFORM
    - DC
    - optional solver W
    """

    Y = np.asarray(Y)
    H = np.asarray(H)

    if D.ndim == 3:
        D2 = D[:, 0, :]
    else:
        D2 = np.asarray(D)

    N, K = D2.shape
    y_true = Y.argmax(axis=1)
    single_idx = source_domains.index(single_source_name)

    def ce_of_probs(probs):
        return evaluate_predictions_cross_entropy(probs, Y)

    def effective_alpha(D2, w, eps=1e-12):
        w = np.asarray(w, dtype=float).reshape(1, -1)
        numer = D2 * w
        denom = numer.sum(axis=1, keepdims=True)
        alpha = numer / np.clip(denom, eps, None)
        return alpha

    def summarize_w_case(name, w):
        probs = combine_predictions_with_w(w, D2, H)
        ce = ce_of_probs(probs)
        alpha = effective_alpha(D2, w, eps=eps)

        print(f"\n{name}")
        print("-" * len(name))
        print(f"w = {np.round(w, 6)}")
        print(f"CE = {ce:.6f}")
        print("Mean effective alpha per source:")
        for j, dom in enumerate(source_domains):
            print(f"  {dom:<10}: mean={alpha[:, j].mean():.6f}, "
                  f"min={alpha[:, j].min():.6f}, max={alpha[:, j].max():.6f}")

        return probs, ce, alpha

    raw_single_probs = H[:, :, single_idx]
    raw_single_ce = ce_of_probs(raw_single_probs)

    print("\nBEST_SINGLE_SRC (RAW)")
    print("---------------------")
    print(f"source = {single_source_name}")
    print(f"one-hot w = {[1.0 if j == single_idx else 0.0 for j in range(K)]}")
    print(f"CE(raw source only) = {raw_single_ce:.6f}")

    one_hot_w = np.zeros(K)
    one_hot_w[single_idx] = 1.0

    onehot_probs, onehot_ce, onehot_alpha = summarize_w_case(
        name=f"BEST_SINGLE_WD ({single_source_name})",
        w=one_hot_w
    )

    max_abs_diff = np.max(np.abs(raw_single_probs - onehot_probs))
    print(f"\nSanity check: max |RAW - WD_onehot| = {max_abs_diff:.12f}")
    print(f"Sanity check: CE difference = {abs(raw_single_ce - onehot_ce):.12f}")

    oracle_probs, oracle_ce, oracle_alpha = summarize_w_case("ORACLE", oracle_w)
    uniform_probs, uniform_ce, uniform_alpha = summarize_w_case("UNIFORM", uniform_w)
    dc_probs, dc_ce, dc_alpha = summarize_w_case("DC", dc_w)

    solver_probs = solver_ce = solver_alpha = None
    if solver_w is not None:
        solver_probs, solver_ce, solver_alpha = summarize_w_case("SOLVER_W", solver_w)

    if "amazon" in source_domains:
        amazon_idx = source_domains.index("amazon")
        webcam_idx = source_domains.index("webcam") if "webcam" in source_domains else None

        order = np.argsort(-dc_alpha[:, amazon_idx])
        top_idx = order[:top_k_samples]

        print("\nTop samples where DC still gives effective weight to amazon")
        print("-----------------------------------------------------------")
        header = (
            "idx | y_true | alpha_amazon(DC) | alpha_webcam(DC) | "
            "p_true_amazon | p_true_webcam | p_true_DC | p_true_UNIFORM | p_true_ORACLE"
        )
        print(header)

        for i in top_idx:
            p_true_amazon = H[i, y_true[i], amazon_idx]
            p_true_webcam = H[i, y_true[i], webcam_idx] if webcam_idx is not None else np.nan
            p_true_dc = dc_probs[i, y_true[i]]
            p_true_uniform = uniform_probs[i, y_true[i]]
            p_true_oracle = oracle_probs[i, y_true[i]]

            alpha_webcam_val = dc_alpha[i, webcam_idx] if webcam_idx is not None else np.nan

            print(
                f"{i:>3} | {y_true[i]:>6} | "
                f"{dc_alpha[i, amazon_idx]:>16.6f} | "
                f"{alpha_webcam_val:>15.6f} | "
                f"{p_true_amazon:>13.6f} | "
                f"{p_true_webcam:>13.6f} | "
                f"{p_true_dc:>9.6f} | "
                f"{p_true_uniform:>14.6f} | "
                f"{p_true_oracle:>13.6f}"
            )

    print("\nCE summary")
    print("----------")
    print(f"BEST_SINGLE_RAW        : {raw_single_ce:.6f}")
    print(f"BEST_SINGLE_WD(onehot) : {onehot_ce:.6f}")
    print(f"UNIFORM                : {uniform_ce:.6f}")
    print(f"ORACLE                 : {oracle_ce:.6f}")
    print(f"DC                     : {dc_ce:.6f}")
    if solver_ce is not None:
        print(f"SOLVER_W               : {solver_ce:.6f}")

    print("\nPairwise differences")
    print("--------------------")
    print(f"DC - BEST_SINGLE_RAW   : {dc_ce - raw_single_ce:+.6f}")
    print(f"DC - ORACLE            : {dc_ce - oracle_ce:+.6f}")
    print(
        f"SOLVER_W - ORACLE      : {(solver_ce - oracle_ce):+.6f}"
        if solver_ce is not None else "N/A"
    )

    return {
        "raw_single_ce": raw_single_ce,
        "onehot_ce": onehot_ce,
        "oracle_ce": oracle_ce,
        "uniform_ce": uniform_ce,
        "dc_ce": dc_ce,
        "solver_ce": solver_ce,
        "dc_alpha": dc_alpha,
        "oracle_alpha": oracle_alpha,
        "uniform_alpha": uniform_alpha,
        "solver_alpha": solver_alpha,
    }


if __name__ == "__main__":
    main()