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
import pickle
from datetime import datetime


def load_global_D_matrix(path: str):
    try:
        Dg = np.load(path)
        print(f"✅ Loaded Global D from: {path}")
        print(f"   Global D shape: {Dg.shape}")
        return Dg
    except Exception as e:
        print(f"❌ Failed loading Global D: {e}")
        return None

def fix_batch_resnet(data, mode):
    if mode in ["OFFICE", "OFFICE31", "OFFICE224"]:
        if data.shape[-1] < 224:
            data = F.interpolate(data, size=(224, 224), mode='bilinear', align_corners=False)
    return data


def slice_global_D_for_domain(Global_D, domains_order, domain_lengths, target_domain):
    """
    Assume Global_D is concatenated in the order of domains_order,
    and within each domain, matches ImageFolder ordering.
    Returns block: (N_domain, K_total)
    """
    start = 0
    for d in domains_order:
        if d == target_domain:
            break
        start += domain_lengths[d]
    end = start + domain_lengths[target_domain]
    return Global_D[start:end]

def map_weights_to_full_source_list(subset_weights, subset_sources, full_source_list):
    full_weights = np.zeros(len(full_source_list))
    if subset_weights is not None:
        for i, source in enumerate(full_source_list):
            if source in subset_sources:
                full_weights[i] = subset_weights[subset_sources.index(source)]
    return full_weights

def save_qstar_artifact(
    base_dir,
    Q,
    w,
    Y,
    source_domains,
    all_source_domains,
    solver,
    backend,
    eps_mult,
    delta_mult,
    dataset_mode,
    strategy_name=None,
    target_domains=None,
    true_r_weights=None,
    extra_info=None,
):
    """
    Save Q* and useful metadata for later analysis.
    """
    os.makedirs(base_dir, exist_ok=True)

    source_domains = list(source_domains)
    all_source_domains = list(all_source_domains)
    target_domains = list(target_domains) if target_domains is not None else list(source_domains)

    safe_target = "_".join(target_domains)
    safe_solver = str(solver).replace("/", "_")
    safe_backend = str(backend).replace("/", "_")
    safe_strategy = "NA" if strategy_name is None else str(strategy_name).replace(" ", "_")

    filename = (
        f"qstar_{dataset_mode}"
        f"__target_{safe_target}"
        f"__strategy_{safe_strategy}"
        f"__solver_{safe_solver}"
        f"__backend_{safe_backend}"
        f"__epsmult_{eps_mult}"
        f"__deltamult_{delta_mult}.pkl"
    )
    save_path = os.path.join(base_dir, filename)

    artifact = {
        "timestamp": datetime.now().isoformat(),
        "dataset_mode": dataset_mode,
        "solver": solver,
        "backend": backend,
        "eps_mult": eps_mult,
        "delta_mult": delta_mult,
        "strategy_name": strategy_name,
        "source_domains": source_domains,
        "target_domains": target_domains,
        "all_source_domains": all_source_domains,
        "Q": np.asarray(Q),
        "Q_shape": np.asarray(Q).shape,
        "w": np.asarray(w),
        "y_true": np.asarray(Y).argmax(axis=1),
        "true_r_weights": None if true_r_weights is None else np.asarray(true_r_weights),
        "extra_info": extra_info,
    }

    with open(save_path, "wb") as f:
        pickle.dump(artifact, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"💾 Saved q* artifact to: {save_path}")
    return save_path