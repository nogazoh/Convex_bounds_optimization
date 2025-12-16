import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import glob
import os, glob
import pandas as pd


def load_losses(model_type, alpha_pos, alpha_neg, data_name, metric):
    """Load loss metric (train/test loss or reconstruction) for given config."""
    path = f"./models_{data_name}"
    losses = []
    for seed in [1, 3, 5]:
        file_path = os.path.join(
            path, f"{model_type}_{alpha_pos}_{alpha_neg}_{data_name}_{metric}.pt")
        if os.path.exists(file_path):
            losses.append(torch.load(file_path))
        else:
            print("File not found:", file_path)
    return losses

def plot_test_loss_mnist():
    """Recreate Figure 7: Test loss over epochs for MNIST"""
    configs = [
        ('vr', 0.5, -1),
        ('vr', 2, -1),
        ('vr', 5, -1),
        ('vrs', 0.5, -0.5),
        ('vrs', 2, -2),
    ]
    labels = [
        "VR$_{0.5}$", "VR$_2$", "VR$_5$", "VRS$_{0.5,-0.5}$", "VRS$_{2,-2}$"
    ]
    linestyles = ['-', '-', '-', '--', '--']

    plt.figure(figsize=(7, 4))
    for (model_type, alpha_pos, alpha_neg), label, ls in zip(configs, labels, linestyles):
        test_losses = load_losses(model_type, alpha_pos, alpha_neg, "MNIST", "test_losses")
        if not test_losses:
            continue
        avg_len = min(len(x) for x in test_losses)
        avg_curve = np.mean([x[:avg_len] for x in test_losses], axis=0)
        plt.plot(np.linspace(0, 100, avg_len), avg_curve, label=label, linestyle=ls)

    plt.xlabel("Epoch percentage")
    plt.ylabel("Test Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("figure7_test_loss_mnist.png")
    plt.close()

def plot_reconstruction_mse():
    """Recreate Figure 8: MSE comparison bar plot"""
    domains = ['USPS', 'MNIST', 'SVHN']
    models = [
        ('vae', 1, -1),
        ('vr', 0.5, -1),
        ('vr', 0.5, -0.5),
        ('vrs', 0.5, -0.5),
    ]
    labels = ['VAE', 'VR$_{0.5}$', 'VRLU$_{-0.5}$', 'VRS$_{0.5,-0.5}$']
    colors = ['#ffb07c', '#8e063b', '#e60049', '#f9b4ab']

    bar_width = 0.2
    x = np.arange(len(domains))
    plt.figure(figsize=(7, 4))

    for i, (model_type, alpha_pos, alpha_neg) in enumerate(models):
        means = []
        for domain in domains:
            recon = load_losses(model_type, alpha_pos, alpha_neg, domain, "test_recon_losses")
            if not recon:
                means.append(np.nan)
                continue
            mse = [x[0] for x in recon]  # extract MSE from tuple
            means.append(np.mean(mse))
        plt.bar(x + i * bar_width, means, width=bar_width, label=labels[i], color=colors[i])

    plt.xticks(x + bar_width * 1.5, domains)
    plt.ylabel("Reconstruction MSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figure8_mse_comparison.png")
    plt.close()

def plot_log_likelihood_pie():
    """Recreate Figure 10: Log likelihood on PIE"""
    domains = ['PIE05', 'PIE07', 'PIE09']
    models = [
        ('vae', 1, -1),
        ('vr', 0.5, -1),
        ('vr', 2, -1),
        ('vrs', 0.5, -0.5),
        ('vrs', 0.5, -2),
        ('vrs', 2, -0.5),
    ]
    labels = ['VAE', 'VR$_{0.5}$', 'VR$_2$', 'VRS$_{0.5,-0.5}$', 'VRS$_{0.5,-2}$', 'VRS$_{2,-0.5}$']
    colors = ['skyblue', 'lightgreen', 'salmon', 'mediumpurple', 'gold', 'plum']

    bar_width = 0.12
    x = np.arange(len(domains))
    plt.figure(figsize=(7, 5))

    for i, (model_type, alpha_pos, alpha_neg) in enumerate(models):
        means = []
        for domain in domains:
            log_p = load_losses(model_type, alpha_pos, alpha_neg, domain, "test_log_p_vals")
            if not log_p:
                means.append(np.nan)
                continue
            means.append(np.mean(log_p))
        plt.bar(x + i * bar_width, means, width=bar_width, label=labels[i], color=colors[i])

    plt.xticks(x + bar_width * len(models) / 2, domains)
    plt.ylabel("Marginal Log Likelihood Estimations")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figure10_log_likelihood_pie.png")
    plt.close()


# def build_results_table(root_dir="."):
#     """
#     Scan recursively for accuracy result files (*_accuracy_score_*.txt),
#     parse their domain accuracies, and build a formatted table
#     [s, m, u, mu, su, sm, smu, Mean].
#     """

#     # Map domain combinations -> table column names
#     COL_MAP = {
#         frozenset(["SVHN"]): "s",
#         frozenset(["MNIST"]): "m",
#         frozenset(["USPS"]): "u",
#         frozenset(["MNIST", "USPS"]): "mu",
#         frozenset(["SVHN", "USPS"]): "su",
#         frozenset(["SVHN", "MNIST"]): "sm",
#         frozenset(["MNIST", "USPS", "SVHN"]): "smu",
#     }

#     # Custom rename rules for pretty row names
#     def pretty_name(path, algo):
#         name = os.path.basename(os.path.dirname(path))
#         if "vae" in name and algo == "DC":
#             return "VAE-MSA"
#         if "vae" in name and algo == "SGD":
#             return "VAE-SGD"
#         if "vr_" in name and algo == "DC":
#             return "VR-MSA"
#         if "vr_" in name and algo == "SGD":
#             return "VR-SGD"
#         if "vrs" in name and algo == "DC":
#             return "VRS-MSA"
#         if "vrs" in name and algo == "SGD":
#             return "VRS-SGD"
#         return f"{algo} | {name}"

#     # Find all result files recursively
#     files = glob.glob(os.path.join(root_dir, "**", "*_accuracy_score_*.txt"), recursive=True)

#     rows = {}
#     for path in files:
#         algo = "SGD" if "SGD" in os.path.basename(path) else "DC"
#         row_name = pretty_name(path, algo)
#         if row_name not in rows:
#             rows[row_name] = {k: np.nan for k in ["s","m","u","mu","su","sm","smu"]}

#         with open(path, "r", encoding="utf-8") as f:
#             for line in f:
#                 parts = [p.strip() for p in line.strip().split("\t") if p.strip()]
#                 if not parts:
#                     continue
#                 try:
#                     val = float(parts[-1])  # last token = score
#                     domains = parts[:-1]    # preceding tokens = domains
#                 except ValueError:
#                     continue
#                 key = COL_MAP.get(frozenset(domains))
#                 if key:
#                     rows[row_name][key] = val

#     # Build DataFrame
#     cols = ["s","m","u","mu","su","sm","smu"]
#     df = pd.DataFrame.from_dict(rows, orient="index")[cols]
#     df["Mean"] = df.mean(axis=1, skipna=True)

#     return df.round(2)

import os, glob, re
import pandas as pd
import numpy as np

def build_results_table_with_alphas(root_dirs):
    """
    Scan the given result root directories (list or str), parse accuracy files,
    and return a pretty DataFrame with rows split by (model, pos_alpha, neg_alpha).
    Columns are ordered as in the paper: s, m, u, mu, su, sm, smu, Mean.

    It understands both DC (MSA) outputs like:
      .../{ESTIMATE}_results_{date}/.../model_type_vrs__pos_alpha_2___neg_alpha_-2/DC_accuracy_score_10.txt
    and SGD outputs like:
      .../Results_____{date}/.../model_type_vrs__pos_alpha_2___neg_alpha_-2/SGD_accuracy_score_48.txt
    """

    # Normalize root_dirs to a list
    if isinstance(root_dirs, str):
        root_dirs = [root_dirs]

    # Domain-combo -> column name
    COL_MAP = {
        frozenset(["SVHN"]): "s",
        frozenset(["MNIST"]): "m",
        frozenset(["USPS"]): "u",
        frozenset(["MNIST", "USPS"]): "mu",
        frozenset(["SVHN", "USPS"]): "su",
        frozenset(["SVHN", "MNIST"]): "sm",
        frozenset(["MNIST", "USPS", "SVHN"]): "smu",
    }
    COL_ORDER = ["s", "m", "u", "mu", "su", "sm", "smu"]

    # Regex to extract model + alphas from a directory name like:
    # "model_type_vrs__pos_alpha_0.5___neg_alpha_-2"
    MODEL_RE = re.compile(
        r"model_type_(?P<model>[a-zA-Z0-9]+)__+pos_alpha_(?P<pos>-?\d+(?:\.\d+)?)__+neg_alpha_(?P<neg>-?\d+(?:\.\d+)?)"
    )

    def guess_algo_from_filename(path):
        """Return 'MSA' (for DC) or 'SGD' based on the file name."""
        base = os.path.basename(path).lower()
        if "dc_accuracy_score" in base:
            return "MSA"
        if "sgd_accuracy_score" in base:
            return "SGD"
        # Fallback: infer by directory name
        return "MSA" if "results_" in path.lower() and "std-score" in path.lower() else "SGD"

    def extract_model_and_alphas(path):
        """
        Walk up directories from the file path until we find a model_type_* folder.
        Return (MODEL_STR, POS_ALPHA, NEG_ALPHA) or (None, None, None) if not found.
        """
        d = os.path.dirname(path)
        for _ in range(5):  # climb a few levels just in case
            m = MODEL_RE.search(os.path.basename(d))
            if m:
                model = m.group("model").upper()  # VRS / VR / VAE
                pos = float(m.group("pos"))
                neg = float(m.group("neg"))
                return model, pos, neg
            d = os.path.dirname(d)
        return None, None, None

    def pretty_row_name(model, pos, neg, algo):
        """Format like VRS_{2,-2}-MSA or VR_{0.5,-1}-SGD or VAE_{1,-1}-MSA."""
        # Compact formatting for integers vs floats
        def fmt(x):
            return str(int(x)) if float(x).is_integer() else str(x)
        return f"{model}_{{{fmt(pos)},{fmt(neg)}}}-{algo}"

    # Collect all *_accuracy_score_*.txt files
    files = []
    for root in root_dirs:
        files.extend(glob.glob(os.path.join(root, "**", "*_accuracy_score_*.txt"), recursive=True))

    # Aggregate results
    table = {}
    for path in files:
        algo = guess_algo_from_filename(path)  # 'MSA' or 'SGD'
        model, pos, neg = extract_model_and_alphas(path)
        if model is None:
            # Skip files we cannot parse
            continue
        row_key = pretty_row_name(model, pos, neg, algo)
        if row_key not in table:
            table[row_key] = {k: np.nan for k in COL_ORDER}

        # Parse lines like: "MNIST\tUSPS\t97.5347" or "MNIST\t97.93"
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = [p.strip() for p in line.strip().split("\t") if p.strip()]
                if not parts:
                    continue
                try:
                    score = float(parts[-1])
                except ValueError:
                    continue
                domains = parts[:-1]
                col = COL_MAP.get(frozenset(domains))
                if col is not None:
                    table[row_key][col] = score

    # Build DataFrame in the desired column order and add Mean
    df = pd.DataFrame.from_dict(table, orient="index")[COL_ORDER]
    df["Mean"] = df.mean(axis=1, skipna=True)

    # Optional: enforce paper-like row ordering
    ordering_blocks = [
        # MSA rows first (DC), then SGD rows — and within each block, the specific alpha combos
        ("MSA", [("VAE", 1, -1),
                 ("VR",  2, -1), ("VR",  0.5, -1),
                 ("VRS", 2, -2), ("VRS", 2, -0.5), ("VRS", 0.5, -2), ("VRS", 0.5, -0.5)]),
        ("SGD", [("VAE", 1, -1),
                 ("VR",  2, -1), ("VR",  0.5, -1),
                 ("VRS", 2, -2), ("VRS", 2, -0.5), ("VRS", 0.5, -2), ("VRS", 0.5, -0.5)]),
    ]
    ordered_index = []
    existing = set(df.index)
    for algo, combos in ordering_blocks:
        for (m, pa, na) in combos:
            name = pretty_row_name(m, pa, na, algo)
            if name in existing:
                ordered_index.append(name)
    # Append any remaining rows (if present) to the end
    ordered_index += [idx for idx in df.index if idx not in ordered_index]
    df = df.loc[ordered_index]

    return df.round(2)
import os, glob, re
import pandas as pd
import numpy as np

import os
import re
import glob
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

def build_readable_msa_table(
    msa_roots: List[str],
    sgd_roots: List[str] = None,
    round_to: int = 2
) -> pd.DataFrame:
    """
    Build a paper-like results table from accuracy files.

    What it does:
    1) Recursively scans `msa_roots` for *DC_accuracy_score_*.txt (MSA rows)
       and `sgd_roots` for *SGD_accuracy_score_*.txt (SGD rows, optional).
    2) Parses domain tuples and their accuracies (single/pairs/triple).
    3) Extracts model type and alphas from the folder name:
         model_type_<MODEL>___pos_alpha_<pos>___neg_alpha_<neg>
    4) Produces a table with columns: s, m, u, mu, su, sm, smu, Mean.
    5) Row names look like: "<MODEL>_{pos,neg}-MSA" or "-SGD".
    
    Notes:
    - Robust to mixed tabs/spaces.
    - Skips lines starting with "z_" (the learned mixture weights).
    - Will only fill columns present in the file; Mean is computed over available columns.
    """

    if sgd_roots is None:
        sgd_roots = []

    # --- columns and mapping from domain set -> column name ---
    COLS = ["s", "m", "u", "mu", "su", "sm", "smu"]
    DOM2COL = {
        frozenset(["SVHN"]): "s",
        frozenset(["MNIST"]): "m",
        frozenset(["USPS"]): "u",
        frozenset(["MNIST", "USPS"]): "mu",
        frozenset(["SVHN", "USPS"]): "su",
        frozenset(["SVHN", "MNIST"]): "sm",
        frozenset(["MNIST", "USPS", "SVHN"]): "smu",
    }

    # --- regex for model/alphas from folder path ---
    MODEL_DIR_RE = re.compile(
        r"model_type_(?P<model>[A-Za-z0-9]+)___pos_alpha_(?P<pos>-?\d+(?:\.\d+)?)___neg_alpha_(?P<neg>-?\d+(?:\.\d+)?)"
    )

    def _fmt_num(x: float) -> str:
        return str(int(x)) if float(x).is_integer() else str(x)

    def _row_name(model: str, pos: float, neg: float, algo: str) -> str:
        return f"{model.upper()}_{{{_fmt_num(pos)},{_fmt_num(neg)}}}-{algo}"

    def _extract_model_and_alphas(path: str) -> Tuple[str, float, float]:
        """
        Walks up the directory tree from `path` until it finds the model folder.
        """
        d = os.path.dirname(path)
        for _ in range(10):
            m = MODEL_DIR_RE.search(os.path.basename(d))
            if m:
                return (
                    m.group("model"),
                    float(m.group("pos")),
                    float(m.group("neg")),
                )
            parent = os.path.dirname(d)
            if parent == d:
                break
            d = parent
        return None, None, None  # not found

    def _parse_accuracy_file(fp: str) -> Dict[str, float]:
        """
        Parses one accuracy file into {column_name: accuracy}.
        Handles any whitespace; skips 'z_*' line.
        """
        out: Dict[str, float] = {}
        with open(fp, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("z_"):
                    continue
                # split on any whitespace (tabs/spaces)
                parts = re.split(r"\s+", line)
                # last token should be the accuracy number
                try:
                    acc = float(parts[-1])
                except ValueError:
                    continue
                domains = parts[:-1]
                col = DOM2COL.get(frozenset(domains))
                if col:
                    out[col] = acc
        return out

    def _gather(files: List[str], algo_label: str, rows_store: Dict[str, Dict[str, float]]):
        for fp in files:
            model, pos, neg = _extract_model_and_alphas(fp)
            if model is None:
                continue
            rname = _row_name(model, pos, neg, algo_label)
            parsed = _parse_accuracy_file(fp)
            if not parsed:
                continue
            if rname not in rows_store:
                rows_store[rname] = {c: np.nan for c in COLS}
            rows_store[rname].update(parsed)

    def _collect_files(roots: List[str], needle: str) -> List[str]:
        acc = []
        for root in roots:
            acc.extend(glob.glob(os.path.join(root, "**", f"*{needle}*.txt"), recursive=True))
        return acc

    # --- collect and parse ---
    table_rows: Dict[str, Dict[str, float]] = {}

    msa_files = _collect_files(msa_roots, "DC_accuracy_score_")
    _gather(msa_files, "MSA", table_rows)

    if sgd_roots:
        sgd_files = _collect_files(sgd_roots, "SGD_accuracy_score_")
        _gather(sgd_files, "SGD", table_rows)

    # --- build DataFrame ---
    if not table_rows:
        return pd.DataFrame(columns=COLS + ["Mean"])

    df = pd.DataFrame.from_dict(table_rows, orient="index")
    df = df[COLS]  # ensure correct column order
    df["Mean"] = df.mean(axis=1, skipna=True)

    # small cosmetic rounding
    df = df.round(round_to)

    # optional: order rows in a paper-like way if present
    desired_order = [
        "VAE_{1,-1}-MSA",
        "VR_{2,-1}-MSA",
        "VR_{0.5,-1}-MSA",
        "VRS_{2,-2}-MSA",
        "VRS_{2,-0.5}-MSA",
        "VRS_{0.5,-2}-MSA",
        "VRS_{0.5,-0.5}-MSA",
        "VAE_{1,-1}-SGD",
        "VR_{2,-1}-SGD",
        "VR_{0.5,-1}-SGD",
        "VRS_{2,-2}-SGD",
        "VRS_{2,-0.5}-SGD",
        "VRS_{0.5,-2}-SGD",
        "VRS_{0.5,-0.5}-SGD",
    ]
    in_order = [r for r in desired_order if r in df.index]
    remaining = [r for r in df.index if r not in in_order]
    df = df.loc[in_order + remaining]

    # pretty index -> human-readable labels
    pretty_index = []
    for r in df.index:
        # "VRS_{2,-0.5}-MSA" -> "VRS2,-0.5-MSA" (matches the screenshot style closely)
        pretty = r.replace("_{", "").replace("}", "")
        pretty_index.append(pretty)
    df.index = pretty_index

    return df

# If you want to run it after training automatically:
if __name__ == "__main__":
    # plot_test_loss_mnist()
    # plot_reconstruction_mse()
    # plot_log_likelihood_pie()
    # df = build_results_table_with_alphas(".")
    msa_root = r"./OURS-STD-SCORE-WITH-KDE_results_26_9"
    sgd_root = r"./Results_____26_9"  # optional; pass [] if you don't want SGD rows

    table = build_readable_msa_table([msa_root], [sgd_root])
    print(table.to_string())          # console view
    table.to_csv("msa_results_table.csv")  # file

