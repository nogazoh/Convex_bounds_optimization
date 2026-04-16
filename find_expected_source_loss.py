import torch
import numpy as np
import pandas as pd
import os
import torch.nn.functional as F
from datetime import datetime
import torch.nn as nn
from torchvision import models

import msa_all_summer as msa
import classifier as ClSFR


def calculate_source_self_performance():

    all_results = []
    dataset_modes = ["OFFICE224", "OFFICE31", "DOMAINNET"]

    for mode in dataset_modes:
        print(f"\n>>> Analyzing Dataset: {mode}")

        msa.DATASET_MODE = mode
        msa.CURRENT_CFG = msa.CONFIGS[mode]
        msa.SOURCE_ERRORS = msa.CURRENT_CFG["SOURCE_ERRORS"]
        msa.ALL_DOMAINS_LIST = msa.CURRENT_CFG["DOMAINS"]
        msa.NUM_CLASSES = msa.CURRENT_CFG["CLASSES"]
        msa.D_PRECOMP_PATH = msa.CURRENT_CFG["D_PRECOMP_PATH"]

        domains = msa.ALL_DOMAINS_LIST
        num_classes = msa.NUM_CLASSES

        if not os.path.exists(msa.D_PRECOMP_PATH):
            print(f"Skipping {mode}: D matrix not found at {msa.D_PRECOMP_PATH}")
            continue

        global_d = msa.load_global_D_matrix(msa.D_PRECOMP_PATH)
        domain_lengths = msa.compute_domain_lengths(domains)

        for dom_name in domains:
            print(f"  Processing domain: {dom_name}...")

            p1 = f"./classifiers/{dom_name}_224.pt"
            p2 = f"./classifiers/{dom_name}_classifier.pt"
            model_path = p1 if os.path.exists(p1) else p2

            if not os.path.exists(model_path):
                print(f"  [Skip] Classifier for {dom_name} not found.")
                continue

            if mode == 'DIGITS':
                net = ClSFR.Grey_32_64_128_gp()
                net.load_state_dict(torch.load(model_path, map_location=msa.device))
            else:
                base_resnet = models.resnet50(weights=None)
                base_resnet.fc = nn.Linear(base_resnet.fc.in_features, num_classes)
                base_resnet.load_state_dict(torch.load(model_path, map_location=msa.device))
                net = msa.FeatureExtractor(base_resnet)

            net.to(msa.device).eval()

            _, test_loader, _, te_idx, _ = msa.get_train_test_loaders_and_indices(dom_name, seed=1)

            d_dom_full = msa.slice_global_D_for_domain(global_d, domains, domain_lengths, dom_name)
            dom_col_idx = domains.index(dom_name)
            d_weights = d_dom_full[te_idx, dom_col_idx]  # (N_test_samples,)

            all_losses = []
            all_correct = []

            with torch.no_grad():
                for imgs, labels in test_loader:
                    imgs, labels = imgs.to(msa.device), labels.to(msa.device)

                    imgs = msa.fix_batch_resnet(imgs, mode)

                    if mode == 'DIGITS':
                        logits = net(imgs)
                    else:
                        _, logits = net(imgs)

                    loss_batch = F.cross_entropy(logits, labels, reduction='none').cpu().numpy()
                    preds = logits.argmax(dim=1)
                    correct_batch = (preds == labels).float().cpu().numpy()

                    all_losses.extend(loss_batch)
                    all_correct.extend(correct_batch)

            all_losses = np.array(all_losses)
            all_correct = np.array(all_correct)

            classic_error = 1.0 - np.mean(all_correct)
            classic_loss = np.mean(all_losses)

            w_sum = np.sum(d_weights) + 1e-9
            weighted_error = 1.0 - (np.sum(all_correct * d_weights) / w_sum)
            weighted_loss = np.sum(all_losses * d_weights) / w_sum

            all_results.append({
                "Dataset": mode,
                "Domain": dom_name,
                "Classic_Error": round(classic_error, 5),
                "Weighted_Error": round(weighted_error, 5),
                "Classic_Loss": round(classic_loss, 5),
                "Weighted_Loss": round(weighted_loss, 5),
                "N_Samples": len(all_correct)
            })

    if all_results:
        df = pd.DataFrame(all_results)
        filename = f"Source_Self_Performance.csv"
        df.to_csv(filename, index=False)

        print("\n" + "=" * 50)
        print(f"Results saved to: {filename}")
        print("=" * 50)
        print(df[["Dataset", "Domain", "Classic_Error", "Weighted_Error"]].to_string())
    else:
        print("No results were generated. Check your paths.")


if __name__ == "__main__":
    calculate_source_self_performance()