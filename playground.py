import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import ast

# ==========================================
# --- CONFIGURATION & PATHS ---
# ==========================================
BASE_ROOT_DIR = "/data/nogaz/Convex_bounds_optimization"
BASE_EXP_DIR = os.path.join(BASE_ROOT_DIR, "LatentFlow_Pixel_Experiments")

DATASET_CONFIGS = {
    "OFFICE31": {
        "DOMAINS": ['amazon', 'dslr', 'webcam'],
        "MODELS_DIR": os.path.join(BASE_EXP_DIR, "models", "gmm_models_soft"),
        "RESULTS_FILE": os.path.join(BASE_ROOT_DIR,
                                     "results_OFFICE31_mosek/seed_1/Sweep_Results_1_3_sets_with_config.txt"),
    },
    "OFFICE224": {
        "DOMAINS": ['Art', 'Clipart', 'Product', 'Real World'],
        "MODELS_DIR": os.path.join(BASE_EXP_DIR, "models", "gmm_models_soft_officehome"),
        "RESULTS_FILE": os.path.join(BASE_ROOT_DIR,
                                     "results_OFFICE224_mosek/seed_1/Sweep_Results_1_3_sets_with_config.txt"),
    }
}

OUTPUT_DIR = os.path.join(BASE_EXP_DIR, "analysis", "pairs_analysis_fix")
PAIRS_DIR = os.path.join(OUTPUT_DIR, "per_pair_detailed")
os.makedirs(PAIRS_DIR, exist_ok=True)


# ==========================================
# --- JSD CALCULATION ---
# ==========================================
def calculate_jsd(gmm_p, gmm_q, alpha=0.5):
    n_samples = 2000
    Xp, _ = gmm_p.sample(n_samples)
    Xq, _ = gmm_q.sample(n_samples)
    lp_p, lq_p = gmm_p.score_samples(Xp), gmm_q.score_samples(Xp)
    lp_q, lq_q = gmm_p.score_samples(Xq), gmm_q.score_samples(Xq)
    a, b = alpha, 1.0 - alpha
    log_m_p = np.logaddexp(np.log(a) + lp_p, np.log(b) + lq_p)
    log_m_q = np.logaddexp(np.log(a) + lp_q, np.log(b) + lq_q)
    return max(0.0, float(a * np.mean(lp_p - log_m_p) + b * np.mean(lq_q - log_m_q)))


# ==========================================
# --- ENHANCED PARSER ---
# ==========================================
def parse_detailed_results(filepath, dataset_name):
    if not os.path.exists(filepath): return []
    with open(filepath, 'r') as f:
        content = f.read()
    data, blocks = [], content.split("TARGET:")[1:]

    for block in blocks:
        lines = [l.strip() for l in block.strip().split('\n') if l]
        try:
            target_list = ast.literal_eval(lines[0].split('|')[0].strip())
            if len(target_list) != 2: continue

            baselines = {}
            ratios = None

            for l in lines:
                if "UNIFORM" in l:
                    baselines['Uniform'] = float(l.split('|')[3])

                if "ORACLE " in l:
                    baselines['Oracle'] = float(l.split('|')[3])

                if "RATIOS:" in l:
                    ratios = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+", l)]

                if "DC (" in l:
                    m_match = re.search(r"(\d+\.\d+)\s*±\s*(\d+\.\d+)", l)
                    if m_match:
                        baselines['Mean_DC'] = float(m_match.group(1))
                        baselines['DC_Std'] = float(m_match.group(2))

            mean_dc = baselines.get('Mean_DC', baselines.get('Uniform', 0.0))
            dc_std = baselines.get('DC_Std', 0.0)
            for l in lines:
                if "acc_Q:" in l and "|" in l:
                    if "nan" in l.lower() or "inf" in l.lower(): continue

                    parts = [p.strip() for p in l.split('|')]

                    # --- התיקון המרכזי כאן ---
                    # אנחנו לוקחים את החלק הראשון ומנקים אותו
                    raw_algo = parts[0].split()[0] if parts[0] else ""

                    # סינון: רק אם זה מתחיל ב-3. (גרסאות האלגוריתם) או מכיל CVXPY
                    # זה מונע מפרמטרים כמו m=1.0 להיכנס בטעות כאלגוריתם
                    if not (raw_algo.startswith("3.") or "CVXPY" in raw_algo):
                        continue

                    algo = raw_algo
                    backend = parts[1]
                    m_val = float(re.search(r"m:([\d\.]+)", parts[2]).group(1))
                    acc_q = float(re.search(r"acc_Q:\s*([\d\.]+)", l).group(1))

                    data.append({
                        "Dataset": dataset_name,
                        "Pair": f"{target_list[0][:2]}-{target_list[1][:2]}",
                        "Algorithm": algo,
                        "Backend": backend,
                        "m": m_val,
                        "Accuracy": acc_q,
                        "Gain": acc_q - mean_dc,
                        "Oracle": baselines.get('Oracle'),
                        "Mean_DC": mean_dc,
                        "DC_Std": dc_std,
                        "Target_List": target_list,
                        "Ratios": ratios
                    })

        except Exception as e:
            print(f"Failed parsing block in {dataset_name}: {e}")
            continue
    return data

# ==========================================
# --- VISUALIZATION MASTER ---
# ==========================================
def generate_extensive_visuals(df):
    print("\n🔍 --- Diagnostic Start ---")

    df['Algorithm'] = df['Algorithm'].astype(str).str.strip()
    df['Backend'] = df['Backend'].astype(str).str.strip()

    allowed_algos = sorted([a for a in df['Algorithm'].unique() if a.startswith('3.') or 'CVXPY' in a])
    allowed_backends = sorted([b for b in df['Backend'].unique() if not re.match(r'^-?\d+\.?\d*$', b)])

    df = df[df['Algorithm'].isin(allowed_algos) & df['Backend'].isin(allowed_backends)].copy()

    print(f"Algorithms for plot: {allowed_algos}")
    print(f"Backends for plot: {allowed_backends}")
    print(f"Total rows to plot: {len(df)}")
    print("--- Diagnostic End ---\n")

    sns.set_theme(style="whitegrid", context="talk")
    df['sqrt_JSD'] = np.sqrt(df['JSD'])



    group_cols = ['Dataset', 'Pair', 'Algorithm', 'm', 'sqrt_JSD']

    idx_best_backend = df.groupby(group_cols)['Gain'].idxmax()
    df_best_backend = df.loc[idx_best_backend].copy()
    plt.figure(figsize=(14, 10))

    ax = sns.scatterplot(
        data=df_best_backend,
        x='sqrt_JSD',
        y='Gain',
        hue='Algorithm',
        hue_order=allowed_algos,
        alpha=0.7,
        s=100,
        palette='viridis'
    )

    plt.axhline(0, color='red', linestyle='--', linewidth=2, label='DC Baseline')
    plt.title("All Permutations: Gain vs. $\sqrt{JSD}$", fontsize=20)
    plt.xlabel("$\sqrt{JSD}$ (Distance)", fontsize=16)
    plt.ylabel("Gain over Mean DC (%)", fontsize=16)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Algorithm")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "1_All_Permutations_Scatter.png"), dpi=300)
    plt.close()

    # --- 1A. Scatter with error bars: best backend per point ---
    plt.figure(figsize=(14, 10))

    colors = dict(zip(allowed_algos, sns.color_palette('viridis', n_colors=len(allowed_algos))))

    for algo in allowed_algos:
        sub = df_best_backend[df_best_backend['Algorithm'] == algo]
        if sub.empty:
            continue

        plt.errorbar(
            sub['sqrt_JSD'],
            sub['Gain'],
            yerr=sub['DC_Std'],
            fmt='o',
            linestyle='none',
            color=colors[algo],
            alpha=0.7,
            markersize=7,
            capsize=3,
            label=algo
        )

    plt.axhline(0, color='red', linestyle='--', linewidth=2, label='DC Baseline')
    plt.title("All Permutations: Gain vs. $\sqrt{JSD}$ with $\pm 1$ Std. DC Error Bars", fontsize=20)
    plt.xlabel("$\sqrt{JSD}$ (Distance)", fontsize=16)
    plt.ylabel("Gain over Mean DC (%)", fontsize=16)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Algorithm")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "1A_All_Permutations_Scatter_With_ErrorBars.png"), dpi=300)
    plt.close()


    win_rows = []

    for pair, pdf in df.groupby('Pair'):
        max_gain = pdf['Gain'].max()
        winners = pdf[np.isclose(pdf['Gain'], max_gain)]

        is_tie = len(winners) > 1
        for _, row in winners.iterrows():
            win_rows.append({
                'Solver': f"{row['Algorithm']} | {row['Backend']}",
                'WinType': 'Tied First' if is_tie else 'Solo First'
            })

    wins_df = pd.DataFrame(win_rows)

    if not wins_df.empty:
        win_summary = (
            wins_df.groupby(['Solver', 'WinType'])
            .size()
            .unstack(fill_value=0)
            .reset_index()
        )

        solver_order = [
            f"{algo} | {backend}"
            for algo in allowed_algos
            for backend in allowed_backends
            if f"{algo} | {backend}" in win_summary['Solver'].values
        ]

        win_summary = win_summary.set_index('Solver').reindex(solver_order, fill_value=0)
        for col in ['Solo First', 'Tied First']:
            if col not in win_summary.columns:
                win_summary[col] = 0

        plt.figure(figsize=(14, 7))
        win_summary[['Solo First', 'Tied First']].plot(
            kind='bar',
            stacked=True,
            ax=plt.gca()
        )
        plt.title("Number of Pair Wins per Solver (Solo vs Tie)", fontsize=18)
        plt.xlabel("Solver")
        plt.ylabel("Count of First-Place Finishes")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "2_Algorithm_vs_Backend_Comparison.png"))
        plt.close()

    plt.figure(figsize=(10, 6))
    summary = df.groupby(['Algorithm', 'Backend'])['Gain'].mean().reset_index()
    sns.barplot(data=summary, x='Algorithm', y='Gain', hue='Backend',
                order=allowed_algos, hue_order=allowed_backends)
    plt.title("Average Improvement per Solver Configuration", fontsize=16)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "3_Average_Improvement_Bar.png"))

    plt.figure(figsize=(12, 8))

    win_rows_4 = []

    for pair, pdf in df.groupby('Pair'):
        max_gain = pdf['Gain'].max()
        winners = pdf[np.isclose(pdf['Gain'], max_gain)].copy()

        # חשוב: לא לספור אותו אלגוריתם פעמיים בגלל backend שונה
        winning_algorithms = sorted(winners['Algorithm'].unique())

        is_tie = len(winning_algorithms) > 1

        for algo in winning_algorithms:
            win_rows_4.append({
                'Algorithm': algo,
                'WinType': 'Tied First' if is_tie else 'Solo First'
            })

    wins4_df = pd.DataFrame(win_rows_4)

    if not wins4_df.empty:
        win4_summary = (
            wins4_df.groupby(['Algorithm', 'WinType'])
            .size()
            .unstack(fill_value=0)
            .reindex(allowed_algos, fill_value=0)
        )

        for col in ['Solo First', 'Tied First']:
            if col not in win4_summary.columns:
                win4_summary[col] = 0

        win4_summary = win4_summary[['Solo First', 'Tied First']]

        win4_summary.plot(
            kind='bar',
            stacked=True,
            figsize=(12, 8)
        )

        plt.title("Number of Pairs Where Algorithm Reached First Place", fontsize=18)
        plt.xlabel("Algorithm")
        plt.ylabel("Count")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "4_Win_Count_Per_Algo.png"))
        plt.close()

    for pair in df['Pair'].unique():
        pdf = df[df['Pair'] == pair]
        if pdf.empty: continue
        plt.figure(figsize=(12, 7))
        sns.lineplot(data=pdf, x='m', y='Accuracy', hue='Algorithm', style='Backend',
                     marker='o', hue_order=allowed_algos, style_order=allowed_backends)

        plt.axhline(pdf['Oracle'].iloc[0], color='green', label='Oracle', linestyle=':')
        plt.axhline(pdf['Mean_DC'].iloc[0], color='red', label='Mean DC', linestyle='-.')
        plt.title(f"Detailed Mix Analysis: {pair} (Dist: {pdf['sqrt_JSD'].iloc[0]:.3f})")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(PAIRS_DIR, f"Detail_{pair}.png"))
        plt.close()


# ==========================================
# --- MAIN ---
# ==========================================
def main():
    print("🚀 Running Extensive Analysis...")
    all_rows = []
    for ds_name, cfg in DATASET_CONFIGS.items():
        print(f"Processing {ds_name}...")
        try:
            gmms = {d: joblib.load(os.path.join(cfg['MODELS_DIR'], f"gmm_{d}.pkl")) for d in cfg['DOMAINS']}
        except Exception as e:
            print(f"Failed parsing block in {dataset_name}: {e}")
            continue

        results = parse_detailed_results(cfg['RESULTS_FILE'], ds_name)
        for row in results:
            d1, d2 = row['Target_List']
            doms = cfg['DOMAINS']
            r1 = row['Ratios'][doms.index(d1)]
            r2 = row['Ratios'][doms.index(d2)]

            alpha = r1 / (r1 + r2) if (r1 + r2) > 0 else 0.5
            row['JSD'] = calculate_jsd(gmms[d1], gmms[d2], alpha=alpha)
            all_rows.append(row)

    if all_rows:
        df = pd.DataFrame(all_rows)
        df = df[df['Algorithm'].apply(lambda x: str(x).startswith('3.') or 'CVXPY' in str(x))].copy()
        generate_extensive_visuals(df)
        df.to_csv(os.path.join(OUTPUT_DIR, "complete_permutations_data.csv"), index=False)
        print(f"✅ Success! Data saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()