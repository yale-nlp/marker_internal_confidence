import os
import json
import pickle
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from termcolor import colored

from scipy.stats import spearmanr, pearsonr

def parse_args():
    """
    Define and parse script arguments.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("--df_dir", type=str, required=True, help="Directory to results for specific model")
    parser.add_argument("--ignore_blank_hedge", default=False, action="store_true")

    args = parser.parse_args()
    return args


def run(args):

    model_name = args.df_dir.split("/__marker")[0].split("_results/")[-1]
    print(colored(f"Running metric computation for model {model_name}...", "cyan"))

    # create save paths
    dfs_dir = Path(args.df_dir)

    if args.ignore_blank_hedge:
        scores_dirname = "_scores_no_blank_hedge"
    else:
        scores_dirname = "_scores"
    result_dir = Path(args.df_dir.replace("_dfs", scores_dirname).replace("train/","").replace("test/", ""))
    result_dir.mkdir(exist_ok=True)

    # load MIC dfs
    mic_df = pd.read_csv(dfs_dir / "mic.csv", index_col="hedges_per_sentence_mapped").apply(pd.to_numeric)
    mic_df_shared = pd.read_csv(dfs_dir / "shared_mic.csv", index_col=0).apply(pd.to_numeric)

    # load MIC dfs
    mic_df = pd.read_csv(dfs_dir / "mic.csv", index_col="hedges_per_sentence_mapped").apply(pd.to_numeric)
    mic_df_shared = pd.read_csv(dfs_dir / "shared_mic.csv", index_col=0).apply(pd.to_numeric)

    if args.ignore_blank_hedge:
        mic_df        = mic_df.drop(index="<no_hedge>", errors="ignore")
        mic_df_shared = mic_df_shared.drop(index="<no_hedge>", errors="ignore")

    # load acc stats
    with open(dfs_dir / "hedge_counts.pkl", "rb") as f:
        num_unique_hedges, unique_hedges, hedge_counts, datasets, dataset_acc_map, dataset_cmfg_map = pickle.load(f)
    acc_series  = pd.Series(dataset_acc_map)
    cmfg_series = pd.Series(dataset_cmfg_map)

    train_datasets = ['arc_challenge', 'sciq', 'mmlu', 'superglue', "wnli", "ambignq", "hallueval", "popqa", "simpleqa", "selfaware", "truthfulqa"]
    test_datasets = ['arc_challenge', 'sciq', 'mmlu', 'superglue', "wnli", "ambignq"]

    # compute MAE variants & MRC correlations
    print(colored(f"   Computing MAE variants & MRC correlations...", "yellow"))
    mrc_corr_df = pd.DataFrame(index=test_datasets, columns=test_datasets, dtype=float)
    mrc_p_df = pd.DataFrame(index=test_datasets, columns=test_datasets, dtype=float)
    mae_df1 = pd.DataFrame(index=test_datasets, columns=test_datasets, dtype=float)
    mae_df2 = pd.DataFrame(index=test_datasets, columns=test_datasets, dtype=float)
    mae_df3 = pd.DataFrame(index=test_datasets, columns=test_datasets, dtype=float)
    mae_df4 = pd.DataFrame(index=test_datasets, columns=test_datasets, dtype=float)

    std_df1 = pd.DataFrame(index=test_datasets, columns=test_datasets, dtype=float)
    std_df2 = pd.DataFrame(index=test_datasets, columns=test_datasets, dtype=float)
    std_df3 = pd.DataFrame(index=test_datasets, columns=test_datasets, dtype=float)
    std_df4 = pd.DataFrame(index=test_datasets, columns=test_datasets, dtype=float)

    n_df1 = pd.DataFrame(index=test_datasets, columns=test_datasets, dtype=float)
    n_df2 = pd.DataFrame(index=test_datasets, columns=test_datasets, dtype=float)
    n_df3 = pd.DataFrame(index=test_datasets, columns=test_datasets, dtype=float)
    n_df4 = pd.DataFrame(index=test_datasets, columns=test_datasets, dtype=float)
    
    for train_ds_name in test_datasets:

        for test_ds_name in test_datasets:
            exploded_df = pd.read_csv(dfs_dir / f"_exploded_df_{test_ds_name}.csv", index_col=0)

            # compute MAE values for I/C-AvgMAE later

            mae_df1.loc[train_ds_name, test_ds_name], std_df1.loc[train_ds_name, test_ds_name], n_df1.loc[train_ds_name, test_ds_name] = get_mae(exploded_df, mic_df, 1, train_ds_name)
            mae_df2.loc[train_ds_name, test_ds_name], std_df2.loc[train_ds_name, test_ds_name], n_df2.loc[train_ds_name, test_ds_name] = get_mae(exploded_df, mic_df, 2, train_ds_name)
            mae_df3.loc[train_ds_name, test_ds_name], std_df3.loc[train_ds_name, test_ds_name], n_df3.loc[train_ds_name, test_ds_name] = get_mae(exploded_df, mic_df, 3, train_ds_name)
            mae_df4.loc[train_ds_name, test_ds_name], std_df4.loc[train_ds_name, test_ds_name], n_df4.loc[train_ds_name, test_ds_name] = get_mae(exploded_df, mic_df, 4, train_ds_name)

    # compute individual marker rank correlation values per ds pair -- use shared markers
    for train_ds_name in train_datasets:
        for test_ds_name in train_datasets:
            corr, p = spearmanr(mic_df_shared[train_ds_name], mic_df_shared[test_ds_name])
            mrc_corr_df.loc[train_ds_name, test_ds_name] = corr
            mrc_p_df.loc[train_ds_name, test_ds_name] = p
            
    # compute MIC rank correlation (mrc)
    c_mask = ~np.eye(len(test_datasets), dtype=bool)
    mrc_vals = mrc_corr_df.values.astype(float)[c_mask]
    mrc_p_vals = mrc_p_df.values.astype(float)[c_mask]
    mrc_sig_mask = mrc_p_vals < 0.05
    n_mrc_sig = mrc_sig_mask.sum()

    mrc_corr_all = fisher_mean(mrc_vals)
    mrc_corr_sig = fisher_mean(mrc_vals[mrc_sig_mask]) if mrc_sig_mask.any() else -1

    # compute accuracy & cmfg correlations -- use shared markers
    print(colored(f"   Computing accuracy & cMFG correlations with shared MICs...", "yellow"))
    mac_spear_corrs, mac_spear_ps, mac_pear_corrs, mac_pear_ps = [], [], [], []
    mcc_spear_corrs, mcc_spear_ps, mcc_pear_corrs, mcc_pear_ps = [], [], [], []

    for marker in mic_df_shared.index:
        mic_row = mic_df_shared.loc[marker, acc_series.index]   # ensure same dataset order as acc/cmfg

        c, p = spearmanr(mic_row, acc_series);   mac_spear_corrs.append(c); mac_spear_ps.append(p)
        c, p = pearsonr(mic_row, acc_series);    mac_pear_corrs.append(c);  mac_pear_ps.append(p)
        c, p = spearmanr(mic_row, cmfg_series);  mcc_spear_corrs.append(c); mcc_spear_ps.append(p)
        c, p = pearsonr(mic_row, cmfg_series);   mcc_pear_corrs.append(c);  mcc_pear_ps.append(p)

    # # get mac/mcc over all correlations (including potentially insignif.)
    mac_corr_spear_all = fisher_mean(mac_spear_corrs)
    mac_corr_pear_all  = fisher_mean(mac_pear_corrs)
    mcc_corr_spear_all = fisher_mean(mcc_spear_corrs)
    mcc_corr_pear_all  = fisher_mean(mcc_pear_corrs)

    # determine significant correlations for mac/mcc
    mac_spear_sig  = [(c, p) for c, p in zip(mac_spear_corrs, mac_spear_ps) if p < 0.05]
    mac_pear_sig   = [(c, p) for c, p in zip(mac_pear_corrs,  mac_pear_ps)  if p < 0.05]
    mcc_spear_sig  = [(c, p) for c, p in zip(mcc_spear_corrs, mcc_spear_ps) if p < 0.05]
    mcc_pear_sig   = [(c, p) for c, p in zip(mcc_pear_corrs,  mcc_pear_ps)  if p < 0.05]

    # get avg over markers of spearman coefficient of (mic of marker i per dataset, accuracy/cmfg per dataset) -- use shared markers
    mac_corr_spear, n_mac_spear = (fisher_mean([c for c,_ in mac_spear_sig]), len(mac_spear_sig)) if mac_spear_sig else (-1, 0)
    mac_corr_pear,  n_mac_pear  = (fisher_mean([c for c,_ in mac_pear_sig]),  len(mac_pear_sig))  if mac_pear_sig  else (-1, 0)
    mcc_corr_spear, n_mcc_spear = (fisher_mean([c for c,_ in mcc_spear_sig]), len(mcc_spear_sig)) if mcc_spear_sig else (-1, 0)
    mcc_corr_pear,  n_mcc_pear  = (fisher_mean([c for c,_ in mcc_pear_sig]),  len(mcc_pear_sig))  if mcc_pear_sig  else (-1, 0)

    # CV = std/mean per column (I) or per row (C)
    print(colored(f"   Computing CV scores...", "yellow"))
    # get coefficient of variation of mic's for each ds (one col of mic_df) & avg over datasets
    i_avg_cv = (mic_df.std() / mic_df.mean()).mean()
    i_avg_cv_shared = (mic_df_shared.std() / mic_df_shared.mean()).mean()

    # get coefifcient of variation of mics across datasets (one row of mic_df_shared) & avg over number of shared markers (len of mic_df_shared)
    c_avg_cv = (mic_df_shared.std(axis=1) / mic_df_shared.mean(axis=1)).mean()
   
    # other stats
    mac_spear_stats  = [(c, p) for c, p in zip(mac_spear_corrs, mac_spear_ps)]
    mac_pear_stats   = [(c, p) for c, p in zip(mac_pear_corrs,  mac_pear_ps)]
    mcc_spear_stats  = [(c, p) for c, p in zip(mcc_spear_corrs, mcc_spear_ps)]
    mcc_pear_stats   = [(c, p) for c, p in zip(mcc_pear_corrs,  mcc_pear_ps)]

    # diagonal = same train/test
    # off-diagonal = different train/test
    # mode 1 = sentence level aggregation; mode 2 = response level; mode 3 = marker level; mode 4 = compare train/test MICs
    results = {
        "I-AvgMAE (Mode 1)": np.diag(mae_df1.values.astype(float)).mean(),
        "I-AvgMAE (Mode 1) std": get_pooled_std(np.diag(std_df1.values.astype(float)), np.diag(n_df1.values.astype(float))),
        "I-AvgMAE (Mode 2)": np.diag(mae_df2.values.astype(float)).mean(),
        "I-AvgMAE (Mode 2) std": get_pooled_std(np.diag(std_df2.values.astype(float)), np.diag(n_df2.values.astype(float))),
        "I-AvgMAE (Mode 3)": np.diag(mae_df3.values.astype(float)).mean(),
        "I-AvgMAE (Mode 3) std": get_pooled_std(np.diag(std_df3.values.astype(float)), np.diag(n_df3.values.astype(float))),
        "I-AvgMAE (Mode 4)": np.diag(mae_df4.values.astype(float)).mean(),
        "I-AvgMAE (Mode 4) std": get_pooled_std(np.diag(std_df4.values.astype(float)), np.diag(n_df4.values.astype(float))),
        "C-AvgMAE (Mode 1)": mae_df1.values.astype(float)[c_mask].mean(),
        "C-AvgMAE (Mode 1) std": get_pooled_std(std_df1.values.astype(float)[c_mask], n_df1.values.astype(float)[c_mask]),
        "C-AvgMAE (Mode 2)": mae_df2.values.astype(float)[c_mask].mean(),
        "C-AvgMAE (Mode 2) std": get_pooled_std(std_df2.values.astype(float)[c_mask], n_df2.values.astype(float)[c_mask]),
        "C-AvgMAE (Mode 3)": mae_df3.values.astype(float)[c_mask].mean(),
        "C-AvgMAE (Mode 3) std": get_pooled_std(std_df3.values.astype(float)[c_mask], n_df3.values.astype(float)[c_mask]),
        "C-AvgMAE (Mode 4)": mae_df4.values.astype(float)[c_mask].mean(),
        "C-AvgMAE (Mode 4) std": get_pooled_std(std_df4.values.astype(float)[c_mask], n_df4.values.astype(float)[c_mask]),
        "MRC-S-all":         mrc_corr_all,
        "MRC-S-sig":         mrc_corr_sig,
        "MRC-S-n_signif":    int(n_mrc_sig),
        "MAC-S-all":            mac_corr_spear_all,
        "MAC-S-sig":            mac_corr_spear,
        "MAC-S-n_signif":       int(n_mac_spear),
        "MAC-P-all":            mac_corr_pear_all,
        "MAC-P-sig":            mac_corr_pear,
        "MAC-P-n_signif":       int(n_mac_pear),
        "MCC-S-all":            mcc_corr_spear_all,
        "MCC-S-sig":            mcc_corr_spear,
        "MCC-S-n_signif":       int(n_mcc_spear),
        "MCC-P-all":            mcc_corr_pear_all,
        "MCC-P-sig":            mcc_corr_pear,
        "MCC-P-n_signif":       int(n_mcc_pear),
        "I-AvgCV":              i_avg_cv,
        "I-AvgCV (Shared)":     i_avg_cv_shared,
        "C-AvgCV":              c_avg_cv,
        "mac_spear_stats":      mac_spear_stats,
        "mac_pear_stats":       mac_pear_stats,
        "mcc_spear_stats":      mcc_spear_stats,
        "mcc_pear_stats":       mcc_pear_stats,
    }

    # Save results
    print(colored(f"Saving results...", "cyan"))
    result_file = result_dir / "_scores.json"
    with open(result_file, "w") as f:
        json.dump(results, f, indent=4)  

    # Save correlation and p-value DFs
    mrc_df = pd.DataFrame(
        {col: {row: (mrc_corr_df.loc[row, col], mrc_p_df.loc[row, col]) for row in test_datasets} for col in test_datasets}
    )
    mrc_df.to_csv(result_dir / "mrc.csv")
    mae_df1.to_csv(result_dir / "mae1.csv")
    mae_df2.to_csv(result_dir / "mae2.csv")
    mae_df3.to_csv(result_dir / "mae3.csv")
    mae_df4.to_csv(result_dir / "mae4.csv")

    std_df1.to_csv(result_dir / "std1.csv")
    std_df2.to_csv(result_dir / "std2.csv")
    std_df3.to_csv(result_dir / "std3.csv")
    std_df4.to_csv(result_dir / "std4.csv")

    n_df1.to_csv(result_dir / "n1.csv")
    n_df2.to_csv(result_dir / "n2.csv")
    n_df3.to_csv(result_dir / "n3.csv")
    n_df4.to_csv(result_dir / "n4.csv")

    print(colored(f"Finished saving everything to {result_dir}!", "green"))


def get_mae(exploded_df, mic_df, mode, train_ds_name):

    df = exploded_df.copy()
    df = df[df['gold_conf_per_sentence'] != -1]
    df = df[df['hedges_per_sentence_mapped'].isin(mic_df.index)]
    df['mic'] = df['hedges_per_sentence_mapped'].map(lambda x: mic_df.loc[x, train_ds_name])
    df['sq_err'] = (df['mic'] - df['gold_conf_per_sentence']) ** 2.

    # Mode 1: per sentence
    if mode==1:
        errors = abs(df['mic'] - df['gold_conf_per_sentence'])
        mae = errors.mean()
        std = errors.std()
        return mae, std, len(errors)  # mode 1
        
    # Mode 2: per sample 
    elif mode==2:
        sample_mae = df.groupby('sample_idx').apply(lambda x: (x['mic'] - x['gold_conf_per_sentence']).abs().mean())
        mae = sample_mae.mean()
        std = sample_mae.std()

        return mae, std, len(sample_mae)  # mode 2

    # Mode 3: per marker
    elif mode==3:
        marker_mae = df.groupby('hedges_per_sentence_mapped').apply(lambda x: (x['mic'] - x['gold_conf_per_sentence']).abs().mean())
        mae = marker_mae.mean()
        std = marker_mae.std()

        return mae, std, len(marker_mae)  # mode 3

    # Mode 4: per marker aggr. over sentences
    elif mode == 4:
        test_marker_mean_conf = df.groupby('hedges_per_sentence_mapped')['gold_conf_per_sentence'].mean()   # test mic
        marker_mic = mic_df.loc[test_marker_mean_conf.index, train_ds_name]

        abs_err = (marker_mic - test_marker_mean_conf).abs()
        mae = abs_err.mean()
        std = abs_err.std()

        return mae, std, len(abs_err)  # mode 4
        
    return mae, std

def get_pooled_std(stds, ns):
    ns = np.array(ns)
    stds = np.array(stds)
    return np.sqrt(np.sum((ns - 1) * stds**2) / np.sum(ns - 1))

def fisher_mean(corrs):
    z = np.arctanh(np.clip(corrs, -0.9999, 0.9999))  # clip to avoid inf at ±1
    return np.tanh(np.mean(z))

if __name__ == "__main__":
    args = parse_args()
    run(args)
