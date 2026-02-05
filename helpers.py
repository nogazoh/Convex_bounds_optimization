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
