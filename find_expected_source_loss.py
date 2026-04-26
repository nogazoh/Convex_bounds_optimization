import torch
import numpy as np
import pandas as pd
import os
import torch.nn.functional as F
import torch.nn as nn
from torchvision import models

import msa_all_summer as msa
import classifier as ClSFR


def calculate_cross_domain_performance():
    all_results = []
    dataset_modes = ["OFFICE224", "OFFICE31", "DOMAINNET"]

    for mode in dataset_modes:
        print(f"\n>>> Analyzing Dataset: {mode}")

        msa.DATASET_MODE = mode
        msa.CURRENT_CFG = msa.CONFIGS[mode]
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

        for source_dom in domains:
            print(f"\n  Loading source model: {source_dom}")

            p1 = f"./classifiers/{source_dom}_224.pt"
            p2 = f"./classifiers/{source_dom}_classifier.pt"
            model_path = p1 if os.path.exists(p1) else p2

            if not os.path.exists(model_path):
                print(f"  [Skip] Classifier for {source_dom} not found.")
                continue

            if mode == "DIGITS":
                net = ClSFR.Grey_32_64_128_gp()
                net.load_state_dict(torch.load(model_path, map_location=msa.device))
            else:
                base_resnet = models.resnet50(weights=None)
                base_resnet.fc = nn.Linear(base_resnet.fc.in_features, num_classes)
                base_resnet.load_state_dict(torch.load(model_path, map_location=msa.device))
                net = msa.FeatureExtractor(base_resnet)

            net.to(msa.device).eval()

            source_col_idx = domains.index(source_dom)

            for target_dom in domains:
                _, test_loader, _, te_idx, _ = msa.get_train_test_loaders_and_indices(target_dom, seed=1)

                # D block של target domain
                d_target_full = msa.slice_global_D_for_domain(global_d, domains, domain_lengths, target_dom)

                # העמודה של source_dom, רק עבור דגימות הטסט
                d_weights = d_target_full[te_idx, source_col_idx]

                all_correct = []
                all_losses = []

                with torch.no_grad():
                    for imgs, labels in test_loader:
                        imgs, labels = imgs.to(msa.device), labels.to(msa.device)

                        imgs = msa.fix_batch_resnet(imgs, mode)

                        if mode == "DIGITS":
                            logits = net(imgs)
                        else:
                            _, logits = net(imgs)

                        preds = logits.argmax(dim=1)
                        correct_batch = (preds == labels).float().cpu().numpy()
                        loss_batch = F.cross_entropy(logits, labels, reduction="none").cpu().numpy()

                        all_correct.extend(correct_batch)
                        all_losses.extend(loss_batch)

                all_correct = np.array(all_correct)
                all_losses = np.array(all_losses)
                d_weights = np.array(d_weights)

                # לא ממושקל
                accuracy_percent = 100.0 * np.mean(all_correct)
                error_percent = 100.0 * (1.0 - np.mean(all_correct))
                mean_ce_loss = np.mean(all_losses)

                # ממושקל לפי P_ij
                w_sum = np.sum(d_weights) + 1e-12
                weighted_accuracy_percent = 100.0 * (np.sum(all_correct * d_weights) / w_sum)
                weighted_error_percent = 100.0 - weighted_accuracy_percent
                weighted_ce_loss = np.sum(all_losses * d_weights) / w_sum

                tag = "SELF" if source_dom == target_dom else "CROSS"
                print(
                    f"    [{tag}] {source_dom} -> {target_dom} | "
                    f"Acc: {accuracy_percent:.2f}% | "
                    f"W-Acc(Pij): {weighted_accuracy_percent:.2f}% | "
                    f"CE: {mean_ce_loss:.5f} | "
                    f"W-CE(Pij): {weighted_ce_loss:.5f}"
                )

                all_results.append({
                    "Dataset": mode,
                    "Source_Domain": source_dom,
                    "Target_Domain": target_dom,
                    "Accuracy": round(accuracy_percent, 2),
                    "Weighted_Accuracy_Pij": round(weighted_accuracy_percent, 2),
                    "Error": round(error_percent, 2),
                    "Weighted_Error_Pij": round(weighted_error_percent, 2),
                    "CrossEntropy_Loss": round(mean_ce_loss, 5),
                    "Weighted_CrossEntropy_Loss_Pij": round(weighted_ce_loss, 5),
                    "N_Samples": len(all_correct),
                    "Weight_Sum": round(float(np.sum(d_weights)), 6),
                    "Is_Self": source_dom == target_dom
                })

    if all_results:
        df = pd.DataFrame(all_results)
        filename = "Cross_Domain_Performance_with_Pij.csv"
        df.to_csv(filename, index=False)

        print("\n" + "=" * 90)
        print(f"Results saved to: {os.path.abspath(filename)}")
        print("=" * 90)

        print(df[[
            "Dataset", "Source_Domain", "Target_Domain",
            "Accuracy", "Weighted_Accuracy_Pij", "Is_Self"
        ]].to_string(index=False))
    else:
        print("No results were generated. Check your paths.")


if __name__ == "__main__":
    calculate_cross_domain_performance()