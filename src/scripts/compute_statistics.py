import os
import re
import argparse
import pandas as pd
import json
import math
import pickle
import json_repair
from pathlib import Path
from termcolor import colored
from collections import Counter
from scipy.stats import linregress

import google.generativeai as genai
from google.generativeai import GenerationConfig

import seaborn as sns
import matplotlib.pyplot as plt

from src.scripts.prompts import STANDARDIZATION_PROMPT

def parse_args():
    """
    Define and parse script arguments.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_dir", type=str, required=True, help="Directory to results for specific model")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--marker_count_threshold", type=int, default=10)
    parser.add_argument("--marker_count_for_plots", type=int, default=30)
    parser.add_argument("--num_samps", type=int, default=None)
    parser.add_argument("--sys_prompt", type=str, default=None)
    parser.add_argument("--read_dfs_from_file", default=False, action="store_true")

    args = parser.parse_args()
    return args

def standardize_hedges(model, generation_config, prompt):

    response = model.generate_content(
        prompt,
        generation_config=generation_config,
    ).text
    canonical_map = json_repair.loads(response.strip().removeprefix("```json").removesuffix("```").strip())

    CUTOFF_WORDS = {'appears', 'although', 'be', 'have', 'find', 'of', 'often', 'seems', 'that', 'appear', 'to'}

    def trim_canonical(phrase):
        words = phrase.lower().split()
        for i, word in enumerate(words):
            if word in CUTOFF_WORDS:
                if word=="that":
                    return ' '.join(words[:i])
                elif word=="to" and i==0:
                    continue
                else:
                    return ' '.join(words[:i+1])
        return phrase

    canonical_map = {k: trim_canonical(v) for k, v in canonical_map.items()}
    canonical_map = {k: re.sub(r'^.*?(suggest)', r'\1', v) for k, v in canonical_map.items()}
    canonical_map = {k: v.replace(" also", " ").strip() for k, v in canonical_map.items()}
    canonical_map = {k: v.replace(" also", "").strip() for k, v in canonical_map.items()}
    canonical_map = {k: v.replace("also ", "").strip() for k, v in canonical_map.items()}
    canonical_map = {k: v.replace("  ", " ").strip() for k, v in canonical_map.items()}
    canonical_map = {k: re.sub(r"^it(?:'s| is)?\s*", "", v).strip() for k, v in canonical_map.items()}

    values = set(canonical_map.values())
    plural_map = {v: v[:-1] for v in values if v.endswith('s') and v[:-1] in values}
    canonical_map = {k: plural_map.get(v, v) for k, v in canonical_map.items()}

    values = set(canonical_map.values())
    ly_map = {v: v[:-2] + 'e' for v in values if v.endswith('ly') and v[:-2] + 'e' in values}
    canonical_map = {k: ly_map.get(v, v) for k, v in canonical_map.items()}
    return canonical_map

def run(args):

    google_key = os.getenv("GOOGLE_API_KEY")
    genai.configure(api_key=google_key)
    model = genai.GenerativeModel("gemini-2.5-flash-lite")
    generation_config = GenerationConfig(
        max_output_tokens=10000, 
        candidate_count=1,
        stop_sequences=['}'],
        temperature=0,
    )

    if args.num_samps is None:
        if args.sys_prompt is None:
            os.makedirs(f"{args.model_dir}/__marker_thresh_{args.marker_count_threshold}", exist_ok=True)
            os.makedirs(f"{args.model_dir}/__marker_thresh_{args.marker_count_threshold}/{args.split}", exist_ok=True)

            plots_dir = Path(args.model_dir) / f"__marker_thresh_{args.marker_count_threshold}" / f"{args.split}" / "_plots"
            plots_dir.mkdir(exist_ok=True)
            dfs_dir = Path(args.model_dir) / f"__marker_thresh_{args.marker_count_threshold}" / f"{args.split}" / "_dfs"
            dfs_dir.mkdir(exist_ok=True)
        else: 
            os.makedirs(f"{args.model_dir}/__marker_thresh_{args.marker_count_threshold}_{args.sys_prompt}", exist_ok=True)
            os.makedirs(f"{args.model_dir}/__marker_thresh_{args.marker_count_threshold}_{args.sys_prompt}/{args.split}", exist_ok=True)

            plots_dir = Path(args.model_dir) / f"__marker_thresh_{args.marker_count_threshold}_{args.sys_prompt}" / f"{args.split}" / "_plots"
            plots_dir.mkdir(exist_ok=True)
            dfs_dir = Path(args.model_dir) / f"__marker_thresh_{args.marker_count_threshold}_{args.sys_prompt}" / f"{args.split}" / "_dfs"
            dfs_dir.mkdir(exist_ok=True)
    else: 
        os.makedirs(f"{args.model_dir}/__marker_thresh_{args.marker_count_threshold}_{args.num_samps}samps", exist_ok=True)
        os.makedirs(f"{args.model_dir}/__marker_thresh_{args.marker_count_threshold}_{args.num_samps}samps/{args.split}", exist_ok=True)

        plots_dir = Path(args.model_dir) / f"__marker_thresh_{args.marker_count_threshold}_{args.num_samps}samps" / f"{args.split}" / "_plots"
        plots_dir.mkdir(exist_ok=True)
        dfs_dir = Path(args.model_dir) / f"__marker_thresh_{args.marker_count_threshold}_{args.num_samps}samps" / f"{args.split}" / "_dfs"
        dfs_dir.mkdir(exist_ok=True)

    model_modifier = f" for {args.model_name}"

    if args.read_dfs_from_file==False:

        # read in model results
        dir_path = Path(args.model_dir)
        datasets = []

        unique_hedges = set()
        hedge_counts = Counter()
        dataset_acc_map = {}
        dataset_cmfg_map = {}

        all_mic, all_mic_i, all_mic_c, all_mic_uf, all_mic_f = {}, {}, {}, {}, {}
        all_mf, all_mf_i, all_mf_c, all_mf_uf, all_mf_f = {}, {}, {}, {}, {}
        all_hedge_counts = {}

        if args.num_samps is None: 
            num_samps = 5000
        else: 
            num_samps = args.num_samps
        if args.sys_prompt is None:
            sys_prompt = "sys1"
        else: 
            sys_prompt = args.sys_prompt
        for file_path in dir_path.glob(f"{args.split}_scores_{sys_prompt}*{num_samps}samps.json"):
            with open(file_path, "r") as f:
                metrics = json.load(f)
            
            print(colored(f"Processing {file_path.name} for {args.model_name}...", "cyan"))
            dataset_name = file_path.name.split("_")[3]
            if dataset_name=="arc": dataset_name = "arc_challenge"
            datasets.append(dataset_name)

            sentences_per_response = metrics['sentences_per_response']
            responses_without_please = metrics['responses_without_please']
            hedges_per_sentence_per_response = metrics['hedges_per_sentence_per_response']
            dec_per_sentence_per_response = metrics['dec_per_sentence_per_response']
            gold_conf_per_sentence_per_response = metrics['gold_conf_per_sentence_per_response']
            dc_gap_per_sentence_per_response = metrics['dc_gap_per_sentence_per_response']
            acc_per_response = metrics['acc_per_response']
            f_score_per_response = metrics['f_score_per_response']
            d_per_response = metrics['d_per_response']
            c_per_response = metrics['c_per_response']
            cmfg = metrics['cmfg']
            mfg = metrics['mfg']
            cmfg_stats = metrics['stats']

            dataset_acc_map[dataset_name] = sum(acc_per_response) / len(acc_per_response)
            dataset_cmfg_map[dataset_name] = cmfg

            response_df = pd.DataFrame({
                'sample_idx': range(len(sentences_per_response)),
                'responses_without_please': responses_without_please,
                'sentences': sentences_per_response,
                'hedges_per_sentence': hedges_per_sentence_per_response,
                'dec_per_sentence': dec_per_sentence_per_response,
                'gold_conf_per_sentence': gold_conf_per_sentence_per_response,
                "dc_gap_per_sentence": dc_gap_per_sentence_per_response,
                'acc': acc_per_response,
                'f_score': f_score_per_response,
                'd': d_per_response,
                'c': c_per_response,
                'cmfg': [cmfg]*len(sentences_per_response),
                'mfg': [mfg]*len(sentences_per_response),
            })

            # Explode all per-sentence columns together (they're all the same length per row)
            try:
                cols = ['sentences', 'hedges_per_sentence', 'dec_per_sentence', 'gold_conf_per_sentence', 'dc_gap_per_sentence']
                mask = response_df[cols].applymap(len).nunique(axis=1) == 1
                print(colored(f"    Skipping {(~mask).sum()} / {len(mask)} rows...", "yellow"))
                exploded_df = response_df[mask].explode(cols).reset_index(drop=True)
            except: 
                x = response_df[cols].applymap(len).nunique(axis=1).pipe(lambda s: response_df[s > 1])
                print(x)
                import ipdb; ipdb.set_trace()
            exploded_df.rename(columns={
                "sentences": "sentence",
                "responses_without_please": "response_without_please",
            }, inplace=True)

            # map hedges to standardized form
            exploded_df2 = exploded_df.explode(['hedges_per_sentence']).reset_index(drop=True)  # expand each per-sentence list of hedges into one row/str
            
            hedge_list = sorted(set(exploded_df2['hedges_per_sentence'].dropna().unique()))
            prompt = STANDARDIZATION_PROMPT.format(expressions=json.dumps(hedge_list, indent=2))
            canonical_map = standardize_hedges(model, generation_config, prompt)

            exploded_df2['hedges_per_sentence_mapped'] = exploded_df2['hedges_per_sentence'].map(lambda x: canonical_map.get(x, x))
            valid_markers = exploded_df2.groupby('hedges_per_sentence_mapped').filter(lambda x: len(x) >= args.marker_count_threshold)
            print(colored(f"    Mapped hedges to canonical form...", "yellow"))

            # save exploded df for metric computation later
            exploded_df2.to_csv(f"{dfs_dir}/_exploded_df_{dataset_name}.csv")

            # get unique hedges 
            unique_hedges.update(exploded_df2['hedges_per_sentence_mapped'].dropna().unique())
            hedge_counts.update(exploded_df2['hedges_per_sentence_mapped'].dropna())

            # compute MIC per model per dataset per marker
            mic_base = valid_markers[valid_markers['gold_conf_per_sentence'] != -1]

            mic  = mic_base.groupby('hedges_per_sentence_mapped')['gold_conf_per_sentence'].mean().sort_values(ascending=False)
            mic_i = mic_base[mic_base['acc'] == 0].groupby('hedges_per_sentence_mapped')['gold_conf_per_sentence'].mean().sort_values(ascending=False)
            mic_c = mic_base[mic_base['acc'] == 1].groupby('hedges_per_sentence_mapped')['gold_conf_per_sentence'].mean().sort_values(ascending=False)
            mic_uf  = mic_base[mic_base['f_score'] < 0.75].groupby('hedges_per_sentence_mapped')['gold_conf_per_sentence'].mean().sort_values(ascending=False)
            mic_f  = mic_base[mic_base['f_score'] >= 0.75].groupby('hedges_per_sentence_mapped')['gold_conf_per_sentence'].mean().sort_values(ascending=False)

            # compute MF per model per dataset per marker
            mf = valid_markers.groupby('hedges_per_sentence_mapped')['dc_gap_per_sentence'].mean().sort_values(ascending=False)
            mf_i = valid_markers[valid_markers['acc'] == 0].groupby('hedges_per_sentence_mapped')['dc_gap_per_sentence'].mean().sort_values(ascending=False)
            mf_c = valid_markers[valid_markers['acc'] == 1].groupby('hedges_per_sentence_mapped')['dc_gap_per_sentence'].mean().sort_values(ascending=False)
            mf_uf  = valid_markers[valid_markers['f_score'] < 0.75].groupby('hedges_per_sentence_mapped')['dc_gap_per_sentence'].mean().sort_values(ascending=False)
            mf_f  = valid_markers[valid_markers['f_score'] >= 0.75].groupby('hedges_per_sentence_mapped')['dc_gap_per_sentence'].mean().sort_values(ascending=False)

            # add as a column of ongoing DF with one row per each of the 10 metrics computed here, and one column per dataset titled dataset_name
            all_mic[dataset_name]    = mic
            all_mic_i[dataset_name]  = mic_i
            all_mic_c[dataset_name]  = mic_c
            all_mic_uf[dataset_name] = mic_uf
            all_mic_f[dataset_name]  = mic_f
            all_mf[dataset_name]     = mf
            all_mf_i[dataset_name]   = mf_i
            all_mf_c[dataset_name]   = mf_c
            all_mf_uf[dataset_name]  = mf_uf
            all_mf_f[dataset_name]   = mf_f
            all_hedge_counts[dataset_name] = dict(Counter(valid_markers['hedges_per_sentence_mapped'].dropna()))
            print(colored(f"    Computed DFs...", "yellow"))

        print(colored(f"Saving DFs for {args.model_name}...", "cyan"))
        
        DATASET_ORDER = ['popqa', 'selfaware', 'simpleqa', "ambignq", "truthfulqa", "hallueval", "wnli", "mmlu", "sciq", "arc_challenge", "superglue"]

        temp = datasets.copy()
        datasets = [d for d in DATASET_ORDER if d in temp]
        ds_order = datasets

        # get and save dataframes
        mic_df    = pd.DataFrame(all_mic).fillna(float('nan'))[ds_order]
        mic_i_df  = pd.DataFrame(all_mic_i).fillna(float('nan'))[ds_order]
        mic_c_df  = pd.DataFrame(all_mic_c).fillna(float('nan'))[ds_order]
        mic_uf_df = pd.DataFrame(all_mic_uf).fillna(float('nan'))[ds_order]
        mic_f_df  = pd.DataFrame(all_mic_f).fillna(float('nan'))[ds_order]
        mf_df     = pd.DataFrame(all_mf).fillna(float('nan'))[ds_order]
        mf_i_df   = pd.DataFrame(all_mf_i).fillna(float('nan'))[ds_order]
        mf_c_df   = pd.DataFrame(all_mf_c).fillna(float('nan'))[ds_order]
        mf_uf_df  = pd.DataFrame(all_mf_uf).fillna(float('nan'))[ds_order]
        mf_f_df   = pd.DataFrame(all_mf_f).fillna(float('nan'))[ds_order]
        freq_df   = pd.DataFrame(all_hedge_counts).fillna(0)[ds_order]  # markers x datasets
        mic_df = mic_df.rename(index={"": "<no_hedge>"})
        mic_i_df = mic_i_df.rename(index={"": "<no_hedge>"})
        mic_c_df = mic_c_df.rename(index={"": "<no_hedge>"})
        mic_f_df = mic_f_df.rename(index={"": "<no_hedge>"})
        mic_uf_df = mic_uf_df.rename(index={"": "<no_hedge>"})
        mf_df = mf_df.rename(index={"": "<no_hedge>"})
        mf_i_df = mf_i_df.rename(index={"": "<no_hedge>"})
        mf_c_df = mf_c_df.rename(index={"": "<no_hedge>"})
        mf_uf_df = mf_uf_df.rename(index={"": "<no_hedge>"})
        mf_f_df = mf_f_df.rename(index={"": "<no_hedge>"})
        freq_df = freq_df.rename(index={"": "<no_hedge>"})

        mic_df.to_csv(f"{dfs_dir}/mic.csv")
        mic_i_df.to_csv(f"{dfs_dir}/mic_i.csv")
        mic_c_df.to_csv(f"{dfs_dir}/mic_c.csv")
        mic_f_df.to_csv(f"{dfs_dir}/mic_f.csv")
        mic_uf_df.to_csv(f"{dfs_dir}/mic_uf.csv")
        mf_df.to_csv(f"{dfs_dir}/mf.csv")
        mf_i_df.to_csv(f"{dfs_dir}/mf_i.csv")
        mf_c_df.to_csv(f"{dfs_dir}/mf_c.csv")
        mf_f_df.to_csv(f"{dfs_dir}/mf_f.csv")
        mf_uf_df.to_csv(f"{dfs_dir}/mf_uf.csv")
        freq_df.to_csv(f"{dfs_dir}/hedge_freq_per_dataset.csv")

        shared_markers = freq_df.index[freq_df.gt(0).all(axis=1)]

        mic_df_shared    = mic_df.loc[shared_markers]
        mic_i_df_shared  = mic_i_df.loc[shared_markers]
        mic_c_df_shared  = mic_c_df.loc[shared_markers]
        mic_uf_df_shared = mic_uf_df.loc[shared_markers]
        mic_f_df_shared  = mic_f_df.loc[shared_markers]
        mf_df_shared     = mf_df.loc[shared_markers]
        mf_i_df_shared   = mf_i_df.loc[shared_markers]
        mf_c_df_shared   = mf_c_df.loc[shared_markers]
        mf_uf_df_shared  = mf_uf_df.loc[shared_markers]
        mf_f_df_shared   = mf_f_df.loc[shared_markers]
        freq_df_shared   = freq_df.loc[shared_markers]

        mic_df_shared = mic_df_shared.rename(index={"": "<no_hedge>"})
        mic_i_df_shared = mic_i_df_shared.rename(index={"": "<no_hedge>"})
        mic_c_df_shared = mic_c_df_shared.rename(index={"": "<no_hedge>"})
        mic_f_df_shared = mic_f_df_shared.rename(index={"": "<no_hedge>"})
        mic_uf_df_shared = mic_uf_df_shared.rename(index={"": "<no_hedge>"})
        mf_df_shared = mf_df_shared.rename(index={"": "<no_hedge>"})
        mf_i_df_shared = mf_i_df_shared.rename(index={"": "<no_hedge>"})
        mf_c_df_shared = mf_c_df_shared.rename(index={"": "<no_hedge>"})
        mf_uf_df_shared = mf_uf_df_shared.rename(index={"": "<no_hedge>"})
        mf_f_df_shared = mf_f_df_shared.rename(index={"": "<no_hedge>"})
        freq_df_shared = freq_df_shared.rename(index={"": "<no_hedge>"})

        all_shared_dfs = [freq_df_shared, mic_df_shared, mic_i_df_shared, mic_c_df_shared, mic_uf_df_shared, mic_f_df_shared, mf_df_shared, mf_i_df_shared, mf_c_df_shared, mf_uf_df_shared, mf_f_df_shared]

        # DO NOT interpolate
        # all_shared_dfs = [df.fillna(df.mean()) for df in all_shared_dfs]

        freq_df_shared, mic_df_shared, mic_i_df_shared, mic_c_df_shared, mic_uf_df_shared, mic_f_df_shared, mf_df_shared, mf_i_df_shared, mf_c_df_shared, mf_uf_df_shared, mf_f_df_shared = all_shared_dfs

        mic_df_shared.to_csv(f"{dfs_dir}/shared_mic.csv")
        mic_i_df_shared.to_csv(f"{dfs_dir}/shared_mic_i.csv")
        mic_c_df_shared.to_csv(f"{dfs_dir}/shared_mic_c.csv")
        mic_f_df_shared.to_csv(f"{dfs_dir}/shared_mic_f.csv")
        mic_uf_df_shared.to_csv(f"{dfs_dir}/shared_mic_uf.csv")
        mf_df_shared.to_csv(f"{dfs_dir}/shared_mf.csv")
        mf_i_df_shared.to_csv(f"{dfs_dir}/shared_mf_i.csv")
        mf_c_df_shared.to_csv(f"{dfs_dir}/shared_mf_c.csv")
        mf_f_df_shared.to_csv(f"{dfs_dir}/shared_mf_f.csv")
        mf_uf_df_shared.to_csv(f"{dfs_dir}/shared_mf_uf.csv")
        freq_df_shared.to_csv(f"{dfs_dir}/shared_hedge_freq_per_dataset.csv")

        # get unique # hedges & hedge frequency across datasets
        num_unique_hedges = len(unique_hedges)
        with open(dfs_dir / "hedge_counts.pkl", "wb") as f:
            pickle.dump((num_unique_hedges, unique_hedges, hedge_counts, datasets, dataset_acc_map, dataset_cmfg_map), f)
        with open(dfs_dir / "unique_hedges.txt", "a") as f:
            f.write(str(unique_hedges) + "\n")
        pd.Series(hedge_counts).sort_values(ascending=False).head(args.marker_count_for_plots).plot(kind='bar', width=0.95, figsize=(14, 5))
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.title(f"Marker Frequency Across Datasets{model_modifier}")
        plt.xlabel("Marker")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(plots_dir / "0_hedge_freq_across_ds_bar.png", dpi=150); plt.close()

    else:
        mic_df = pd.read_csv(f"{dfs_dir}/mic.csv", index_col="hedges_per_sentence_mapped").apply(pd.to_numeric)
        mic_i_df = pd.read_csv(f"{dfs_dir}/mic_i.csv", index_col="hedges_per_sentence_mapped").apply(pd.to_numeric)
        mic_c_df = pd.read_csv(f"{dfs_dir}/mic_c.csv", index_col="hedges_per_sentence_mapped").apply(pd.to_numeric)
        mic_f_df = pd.read_csv(f"{dfs_dir}/mic_f.csv", index_col="hedges_per_sentence_mapped").apply(pd.to_numeric)
        mic_uf_df = pd.read_csv(f"{dfs_dir}/mic_uf.csv", index_col="hedges_per_sentence_mapped").apply(pd.to_numeric)
        mf_df = pd.read_csv(f"{dfs_dir}/mf.csv", index_col="hedges_per_sentence_mapped").apply(pd.to_numeric)
        mf_i_df = pd.read_csv(f"{dfs_dir}/mf_i.csv", index_col="hedges_per_sentence_mapped").apply(pd.to_numeric)
        mf_c_df = pd.read_csv(f"{dfs_dir}/mf_c.csv", index_col="hedges_per_sentence_mapped").apply(pd.to_numeric)
        mf_f_df = pd.read_csv(f"{dfs_dir}/mf_f.csv", index_col="hedges_per_sentence_mapped").apply(pd.to_numeric)
        mf_uf_df = pd.read_csv(f"{dfs_dir}/mf_uf.csv", index_col="hedges_per_sentence_mapped").apply(pd.to_numeric)
        freq_df = pd.read_csv(f"{dfs_dir}/hedge_freq_per_dataset.csv", index_col=0)
        freq_df = freq_df.apply(pd.to_numeric)

        mic_df_shared = pd.read_csv(f"{dfs_dir}/shared_mic.csv", index_col=0).apply(pd.to_numeric)
        mic_i_df_shared = pd.read_csv(f"{dfs_dir}/shared_mic_i.csv", index_col=0).apply(pd.to_numeric)
        mic_c_df_shared = pd.read_csv(f"{dfs_dir}/shared_mic_c.csv", index_col=0).apply(pd.to_numeric)
        mic_f_df_shared = pd.read_csv(f"{dfs_dir}/shared_mic_f.csv", index_col=0).apply(pd.to_numeric)
        mic_uf_df_shared = pd.read_csv(f"{dfs_dir}/shared_mic_uf.csv", index_col=0).apply(pd.to_numeric)
        mf_df_shared = pd.read_csv(f"{dfs_dir}/shared_mf.csv", index_col=0).apply(pd.to_numeric)
        mf_i_df_shared = pd.read_csv(f"{dfs_dir}/shared_mf_i.csv", index_col=0).apply(pd.to_numeric)
        mf_c_df_shared = pd.read_csv(f"{dfs_dir}/shared_mf_c.csv", index_col=0).apply(pd.to_numeric)
        mf_f_df_shared = pd.read_csv(f"{dfs_dir}/shared_mf_f.csv", index_col=0).apply(pd.to_numeric)
        mf_uf_df_shared = pd.read_csv(f"{dfs_dir}/shared_mf_uf.csv", index_col=0).apply(pd.to_numeric)
        freq_df_shared = pd.read_csv(f"{dfs_dir}/shared_hedge_freq_per_dataset.csv", index_col=0).apply(pd.to_numeric)

        with open(dfs_dir / "hedge_counts.pkl", "rb") as f:
            num_unique_hedges, unique_hedges, hedge_counts, datasets, dataset_acc_map, dataset_cmfg_map = pickle.load(f)

    # plot statistics + MIC-I/C, MIC-F/UF, MIC vs. MF
    print(colored(f"Proceeding to plots for {args.model_name}...", "cyan"))

    pd.Series(hedge_counts).sort_values(ascending=False).head(args.marker_count_for_plots).plot(kind='bar', width=0.95, figsize=(14, 5))
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.title(f"Marker Frequency Across Datasets{model_modifier}", fontdict={'fontsize': 14})
    plt.xlabel("Marker", fontdict={'fontsize': 12})
    plt.ylabel("Count", fontdict={'fontsize': 12})
    plt.tight_layout()
    plt.savefig(plots_dir / "0_hedge_freq_across_ds_bar.png", dpi=150); plt.close()

    # plot frequency of hedges -- heatmap
    top_markers = freq_df.sum(axis=1).nlargest(args.marker_count_for_plots).index
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(freq_df.loc[top_markers], ax=ax, cmap='crest', linewidths=0.3, vmin=0) 
    ax.set_title(f"Marker Frequency by Dataset{model_modifier}")
    ax.set_xlabel("Dataset"); ax.set_ylabel("Marker")
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig(plots_dir / "1_hedge_freq_per_ds_heatmap.png", dpi=150); plt.close()

    top_markers = freq_df_shared.sum(axis=1).nlargest(args.marker_count_for_plots).index
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(freq_df_shared.loc[top_markers], ax=ax, cmap='crest', linewidths=0.3, vmin=0) 
    ax.set_title(f"Shared Marker Frequency by Dataset{model_modifier}")
    ax.set_xlabel("Dataset"); ax.set_ylabel("Marker")
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig(plots_dir / "1_shared_hedge_freq_per_ds_heatmap.png", dpi=150); plt.close()
    
    # plot MIC values -- heatmap
    top_markers = mic_df_shared.mean(axis=1).nlargest(args.marker_count_for_plots).index
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(mic_df_shared.loc[top_markers], ax=ax, cmap='crest', vmin=0, vmax=1, linewidths=0.3, annot=True, fmt='.2f')
    ax.set_title(f"MICs per Dataset{model_modifier}")
    ax.set_xlabel("Dataset"); ax.set_ylabel("Marker")
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='y', rotation=0)
    plt.tight_layout()
    plt.savefig(plots_dir / "2_mic_per_ds_heatmap.png", dpi=150); plt.close()

    # plot MIC distribution per model
    mic_long = mic_df.reset_index().melt(id_vars='hedges_per_sentence_mapped', var_name='dataset', value_name='mic').dropna()
    mic_long.rename(columns={'hedges_per_sentence_mapped': 'marker'}, inplace=True)
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.violinplot(data=mic_long, x='dataset', y='mic', ax=ax)
    ax.set_ylim(0, 1); ax.set_title(f"MIC Distribution by Dataset{model_modifier}")
    ax.set_xlabel("Dataset"); ax.set_ylabel("MIC"); ax.tick_params(axis='x', rotation=45)
    ax2 = ax.twinx()
    ax2.set_ylim(0, 1)
    dataset_acc  = pd.Series(dataset_acc_map).reindex(datasets)
    dataset_cmfg = pd.Series(dataset_cmfg_map).reindex(datasets)
    x_positions = range(len(datasets))
    ax2.plot(x_positions, dataset_acc.values,  '.', color='black',    markersize=8, label='Accuracy',  zorder=5)
    ax2.plot(x_positions, dataset_cmfg.values, '*', color='black', markersize=10, label='cMFG', zorder=5)
    ax2.set_ylabel("Accuracy / cMFG")
    ax2.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(plots_dir / "3_mic_distr_per_ds_violin.png", dpi=150); plt.close()

    # plot MIC value by acc per dataset
    scatter_df = mic_c_df.reset_index().melt(id_vars='hedges_per_sentence_mapped', var_name='dataset', value_name='mic_c').merge(mic_i_df.reset_index().melt(id_vars='hedges_per_sentence_mapped', var_name='dataset', value_name='mic_i'), on=['hedges_per_sentence_mapped', 'dataset']).dropna()
    fig, ax = plt.subplots(figsize=(7, 7))
    sns.scatterplot(data=scatter_df, x='mic_i', y='mic_c', hue='dataset', alpha=0.7, ax=ax)
    sns.regplot(data=scatter_df, x='mic_i', y='mic_c', scatter=False, ax=ax, color='black', line_kws={'linewidth':1.5})
    from scipy.stats import linregress
    slope, intercept, r_value, p_value, std_err = linregress(scatter_df['mic_i'], scatter_df['mic_c'])
    ax.text(0.05, 0.95, f"R²={r_value**2:.2f}\n$p$={p_value:.2f}", transform=ax.transAxes, verticalalignment='top', fontsize=14, bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.set_title(f"MIC-C vs. MIC-I{model_modifier}", fontdict={'fontsize':16})
    ax.set_xlabel("MIC (Incorrect)", fontdict={'fontsize':16}); ax.set_ylabel("MIC (Correct)", fontdict={'fontsize':16})
    plt.tight_layout()
    plt.savefig(plots_dir / "4_mic_c_vs_mic_i_scatter.png", dpi=150); plt.close()

    # same but subplot per dataset
    n_datasets = len(datasets)
    n_cols = 2
    n_rows = math.ceil(n_datasets / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 7*n_rows), sharey=False, sharex=False)
    axes = axes.flatten()
    for ax, dataset in zip(axes, datasets):
        ds_df = scatter_df[scatter_df['dataset'] == dataset]
        sns.scatterplot(data=ds_df, x='mic_i', y='mic_c', alpha=0.7, ax=ax)
        # ax.plot([0,1],[0,1], 'k--', linewidth=1)
        sns.regplot(data=ds_df, x='mic_i', y='mic_c', scatter=False, ax=ax, color='black', line_kws={'linewidth':1.5})
        ax.set_xlim(0,1); ax.set_ylim(0,1)
        ax.set_title(dataset, fontsize=18)
        ax.set_xlabel("MIC (Incorrect)", fontsize=18)
        ax.set_ylabel("MIC (Correct)", fontsize=18)
        ax.tick_params(labelsize=14)
        if len(ds_df) > 1:
            slope, intercept, r_value, p_value, std_err = linregress(ds_df['mic_i'], ds_df['mic_c'])
            ax.text(0.05, 0.95, f"R²={r_value**2:.2f}\n$p$={p_value:.2f}", transform=ax.transAxes, verticalalignment='top', fontsize=14, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    for ax in axes[n_datasets:]:
        ax.set_visible(False)
    fig.suptitle(f"MIC-C vs. MIC-I per Dataset{model_modifier}", fontsize=24, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    fig.subplots_adjust(hspace=0.2, wspace=0.15)
    plt.savefig(plots_dir / "4b_mic_c_vs_mic_i_per_ds_scatter_reg.png", dpi=150); plt.close()

    # same but diagonal instead of regression
    n_datasets = len(datasets)
    n_cols = 2
    n_rows = math.ceil(n_datasets / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 7*n_rows), sharey=False, sharex=False)
    axes = axes.flatten()
    for ax, dataset in zip(axes, datasets):
        ds_df = scatter_df[scatter_df['dataset'] == dataset]
        sns.scatterplot(data=ds_df, x='mic_i', y='mic_c', alpha=0.7, ax=ax)
        ax.plot([0,1],[0,1], 'k--', linewidth=1)
        ax.set_xlim(0,1); ax.set_ylim(0,1)
        ax.set_title(dataset, fontsize=18)
        ax.set_xlabel("MIC (Incorrect)", fontsize=18)
        ax.set_ylabel("MIC (Correct)", fontsize=18)
        ax.tick_params(labelsize=14)
    for ax in axes[n_datasets:]:
        ax.set_visible(False)
    fig.suptitle(f"MIC-C vs. MIC-I per Dataset{model_modifier}", fontsize=24, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    fig.subplots_adjust(hspace=0.2, wspace=0.15)
    plt.savefig(plots_dir / "4b_mic_c_vs_mic_i_per_ds_scatter_diag.png", dpi=150); plt.close()

    # plot MIC value by acc per dataset -- violin plots
    palette = {
        "correct": "cornflowerblue",
        "incorrect": "lightcoral"
    }
    mic_c_long = mic_c_df.reset_index().melt(id_vars='hedges_per_sentence_mapped', var_name='dataset', value_name='mic').dropna(); mic_c_long['condition'] = 'correct'
    mic_i_long = mic_i_df.reset_index().melt(id_vars='hedges_per_sentence_mapped', var_name='dataset', value_name='mic').dropna(); mic_i_long['condition'] = 'incorrect'
    acc_long = pd.concat([mic_c_long, mic_i_long])
    fig, ax = plt.subplots(figsize=(14, 5))
    sns.violinplot(data=acc_long, x='dataset', y='mic', hue='condition', split=True, ax=ax, palette=palette)
    ax.set_ylim(0,1); ax.set_title(f"MIC by Correctness per Dataset{model_modifier}", fontdict={'fontsize':16})
    ax.set_xlabel("Dataset", fontdict={'fontsize':14}); ax.set_ylabel("MIC", fontdict={'fontsize':14}); ax.tick_params(axis='x', rotation=45, labelsize=12)
    ax.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(plots_dir / "4c_mic_by_acc_per_ds_violin.png", dpi=150)
    plt.tight_layout()
    ax2 = ax.twinx()
    ax2.set_ylim(0, 1)
    dataset_acc  = pd.Series(dataset_acc_map).reindex(datasets)
    x_positions = range(len(datasets))
    ax2.plot(x_positions, dataset_acc.values,  '.', color='black',    markersize=8, label='Accuracy',  zorder=5)
    ax2.set_ylabel("Accuracy")
    ax2.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(plots_dir / "4c_mic_by_acc_per_ds_violinwith_acc.png", dpi=150); plt.close()

    # BIGGER FONT
    palette = {
        "correct": "cornflowerblue",
        "incorrect": "lightcoral"
    }
    mic_c_long = mic_c_df.reset_index().melt(id_vars='hedges_per_sentence_mapped', var_name='dataset', value_name='mic').dropna(); mic_c_long['condition'] = 'correct'
    mic_i_long = mic_i_df.reset_index().melt(id_vars='hedges_per_sentence_mapped', var_name='dataset', value_name='mic').dropna(); mic_i_long['condition'] = 'incorrect'
    acc_long = pd.concat([mic_c_long, mic_i_long])
    fig, ax = plt.subplots(figsize=(14, 5))
    sns.violinplot(data=acc_long, x='dataset', y='mic', hue='condition', split=True, ax=ax, palette=palette)
    ax.set_ylim(0,1); ax.set_title(f"MIC by Correctness per Dataset{model_modifier}", fontdict={'fontsize':22})
    ax.set_xlabel("Dataset", fontdict={'fontsize':18}); ax.set_ylabel("MIC", fontdict={'fontsize':18}); ax.tick_params(axis='x', rotation=45, labelsize=18); ax.tick_params(axis='y', labelsize=18)
    ax.legend(loc='upper left', fontsize=14)
    ax2 = ax.twinx()
    ax2.set_ylim(0, 1)
    dataset_acc  = pd.Series(dataset_acc_map).reindex(datasets)
    x_positions = range(len(datasets))
    ax2.plot(x_positions, dataset_acc.values,  '.', color='black',    markersize=10, label='Accuracy',  zorder=5)
    ax2.tick_params(axis='y', labelsize=18)
    ax2.set_ylabel("Accuracy", fontdict={'fontsize':18})
    ax2.legend(loc='lower right', fontsize=14)
    plt.tight_layout()
    plt.savefig(plots_dir / "4c_mic_by_acc_per_ds_violinwith_accBIGGER.png", dpi=150); plt.close()

    # plot MIC value by f_score per dataset
    scatter_df2 = mic_f_df.reset_index().melt(id_vars='hedges_per_sentence_mapped', var_name='dataset', value_name='mic_f').merge(mic_uf_df.reset_index().melt(id_vars='hedges_per_sentence_mapped', var_name='dataset', value_name='mic_uf'), on=['hedges_per_sentence_mapped', 'dataset']).dropna()
    fig, ax = plt.subplots(figsize=(7, 7))
    sns.scatterplot(data=scatter_df2, x='mic_uf', y='mic_f', hue='dataset', alpha=0.7, ax=ax)
    sns.regplot(data=scatter_df2, x='mic_uf', y='mic_f', scatter=False, ax=ax, color='black', line_kws={'linewidth':1.5})
    slope, intercept, r_value, p_value, std_err = linregress(scatter_df2['mic_uf'], scatter_df2['mic_f'])
    ax.text(0.95, 0.95, f"R²={r_value**2:.2f}\n$p$={p_value:.2f}", transform=ax.transAxes, verticalalignment='top', horizontalalignment='right', fontsize=14, bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.set_title(f"MIC-F vs. MIC-UF{model_modifier}", fontdict={'fontsize':16})
    ax.set_xlabel("MIC (Unfaithful)", fontdict={'fontsize':16}); ax.set_ylabel("MIC (Faithful)", fontdict={'fontsize':16})
    plt.tight_layout()
    plt.savefig(plots_dir / "5_mic_f_vs_mic_uf_scatter.png", dpi=150); plt.close()

    # same plot but per dataset
    n_datasets = len(datasets)
    n_cols = 2
    n_rows = math.ceil(n_datasets / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 7*n_rows), sharey=False, sharex=False)
    axes = axes.flatten()
    for ax, dataset in zip(axes, datasets):
        ds_df = scatter_df2[scatter_df2['dataset'] == dataset]
        sns.scatterplot(data=ds_df, x='mic_uf', y='mic_f', alpha=0.7, ax=ax)
        sns.regplot(data=ds_df, x='mic_uf', y='mic_f', scatter=False, ax=ax, color='black', line_kws={'linewidth':1.5})
        ax.set_xlim(0,1); ax.set_ylim(0,1)
        ax.set_title(dataset, fontsize=18)
        ax.set_xlabel("MIC (Unfaithful)", fontsize=18)
        ax.set_ylabel("MIC (Faithful)", fontsize=18)
        ax.tick_params(labelsize=14)
        if len(ds_df) > 1:
            slope, intercept, r_value, p_value, std_err = linregress(ds_df['mic_uf'], ds_df['mic_f'])
            ax.text(0.05, 0.95, f"R²={r_value**2:.2f}\n$p$={p_value:.2f}", transform=ax.transAxes, verticalalignment='top', fontsize=14, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    for ax in axes[n_datasets:]:
        ax.set_visible(False)
    fig.suptitle(f"MIC-F vs. MIC-UF per Dataset{model_modifier}", fontsize=24, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    fig.subplots_adjust(hspace=0.2, wspace=0.15)
    plt.savefig(plots_dir / "5b_mic_f_vs_mic_uf_per_ds_scatter_reg.png", dpi=150)
    plt.close()

    # same plot but per dataset with diagonal
    n_datasets = len(datasets)
    n_cols = 2
    n_rows = math.ceil(n_datasets / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 7*n_rows), sharey=False, sharex=False)
    axes = axes.flatten()
    for ax, dataset in zip(axes, datasets):
        ds_df = scatter_df2[scatter_df2['dataset'] == dataset]
        sns.scatterplot(data=ds_df, x='mic_uf', y='mic_f', alpha=0.7, ax=ax)
        ax.plot([0,1], [0,1], 'k--', linewidth=1)
        ax.set_xlim(0,1); ax.set_ylim(0,1)
        ax.set_title(dataset, fontsize=18)
        ax.set_xlabel("MIC (Unfaithful)", fontsize=18)
        ax.set_ylabel("MIC (Faithful)", fontsize=18)
        ax.tick_params(labelsize=14)
    for ax in axes[n_datasets:]:
        ax.set_visible(False)
    fig.suptitle(f"MIC-F vs. MIC-UF per Dataset{model_modifier}", fontsize=24, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    fig.subplots_adjust(hspace=0.2, wspace=0.15)
    plt.savefig(plots_dir / "5b_mic_f_vs_mic_uf_per_ds_scatter_diag.png", dpi=150)
    plt.close()

    # plot MIC value by f_score per dataset -- violin plots
    palette = {
        "faithful": "royalblue",
        "unfaithful": "firebrick"
    }
    mic_f_long = mic_f_df.reset_index().melt(id_vars='hedges_per_sentence_mapped', var_name='dataset', value_name='mic').dropna(); mic_f_long['condition'] = 'faithful'
    mic_uf_long = mic_uf_df.reset_index().melt(id_vars='hedges_per_sentence_mapped', var_name='dataset', value_name='mic').dropna(); mic_uf_long['condition'] = 'unfaithful'
    faith_long = pd.concat([mic_f_long, mic_uf_long])
    fig, ax = plt.subplots(figsize=(14, 5))
    sns.violinplot(data=faith_long, x='dataset', y='mic', hue='condition', split=True, ax=ax, palette=palette)
    ax.set_ylim(0,1); ax.set_title(f"MIC by Faithfulness per Dataset{model_modifier}", fontdict={'fontsize':16})
    ax.set_xlabel("Dataset", fontdict={'fontsize':14}); ax.set_ylabel("MIC", fontdict={'fontsize':14}); ax.tick_params(axis='x', rotation=45, labelsize=12)
    ax.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(plots_dir / "5c_mic_by_f_per_ds_violin.png", dpi=150)
    ax2 = ax.twinx()
    ax2.set_ylim(0, 1)
    dataset_cmfg = pd.Series(dataset_cmfg_map).reindex(datasets)
    x_positions = range(len(datasets))
    ax2.plot(x_positions, dataset_cmfg.values, '*', color='black', markersize=10, label='cMFG', zorder=5)
    ax2.set_ylabel("cMFG")
    ax2.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(plots_dir / "5c_mic_by_f_per_ds_violinwith_cmfg.png", dpi=150); plt.close()

    # BIGGER FONT
    palette = {
        "faithful": "royalblue",
        "unfaithful": "firebrick"
    }
    mic_f_long = mic_f_df.reset_index().melt(id_vars='hedges_per_sentence_mapped', var_name='dataset', value_name='mic').dropna(); mic_f_long['condition'] = 'faithful'
    mic_uf_long = mic_uf_df.reset_index().melt(id_vars='hedges_per_sentence_mapped', var_name='dataset', value_name='mic').dropna(); mic_uf_long['condition'] = 'unfaithful'
    faith_long = pd.concat([mic_f_long, mic_uf_long])
    fig, ax = plt.subplots(figsize=(14, 5))
    sns.violinplot(data=faith_long, x='dataset', y='mic', hue='condition', split=True, ax=ax, palette=palette)
    ax.set_ylim(0,1); ax.set_title(f"MIC by Faithfulness per Dataset{model_modifier}", fontdict={'fontsize':22})
    ax.set_xlabel("Dataset", fontdict={'fontsize':18}); ax.set_ylabel("MIC", fontdict={'fontsize':18}); ax.tick_params(axis='x', rotation=45, labelsize=18); ax.tick_params(axis='y', labelsize=18)
    ax.legend(loc='upper left', fontsize=14)
    ax2 = ax.twinx()
    ax2.set_ylim(0, 1)
    dataset_cmfg = pd.Series(dataset_cmfg_map).reindex(datasets)
    x_positions = range(len(datasets))
    ax2.plot(x_positions, dataset_cmfg.values, '*', color='black', markersize=10, label='cMFG', zorder=5)
    ax2.tick_params(axis='y', labelsize=18)
    ax2.set_ylabel("cMFG", fontdict={'fontsize':18})
    ax2.legend(loc='lower right', fontsize=14)
    plt.tight_layout()
    plt.savefig(plots_dir / "5c_mic_by_f_per_ds_violinwith_cmfgBIGGER.png", dpi=150); plt.close()

    # plot MIC value vs. MF value ACROSS ALL datasets 
    mic_long = mic_df.reset_index().melt(id_vars='hedges_per_sentence_mapped', var_name='dataset', value_name='mic').dropna()
    mf_long  = mf_df.reset_index().melt(id_vars='hedges_per_sentence_mapped', var_name='dataset', value_name='mf').dropna()
    combined = mic_long.merge(mf_long, on=['hedges_per_sentence_mapped','dataset']).rename(columns={'hedges_per_sentence_mapped':'marker'})
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.scatterplot(data=combined, x='mic', y='mf', hue='dataset', alpha=0.7, ax=ax)
    sns.regplot(data=combined, x='mic', y='mf', scatter=False, ax=ax, color='black', line_kws={'linewidth':1.5})
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.set_title(f"MIC vs. MF Across Datasets{model_modifier}", fontdict={'fontsize':16})
    ax.set_xlabel("MIC", fontdict={'fontsize':16}); ax.set_ylabel("MF", fontdict={'fontsize':16})
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    slope, intercept, r_value, p_value, std_err = linregress(combined['mic'], combined['mf'])
    ax.text(0.95, 0.95, f"R²={r_value**2:.2f}\n$p$={p_value:.2f}", transform=ax.transAxes, verticalalignment='top', horizontalalignment='right', fontsize=14, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    plt.savefig(plots_dir / "7_mic_vs_mf_scatter.png", dpi=150); plt.close()

    # same as above but per dataset
    n_datasets = len(datasets)
    n_cols = 2
    n_rows = math.ceil(n_datasets / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 7*n_rows), sharey=False, sharex=False)
    axes = axes.flatten()
    for ax, dataset in zip(axes, datasets):
        ds_df = combined[combined['dataset'] == dataset]
        sns.scatterplot(data=ds_df, x='mic', y='mf', alpha=0.7, ax=ax)
        sns.regplot(data=ds_df, x='mic', y='mf', scatter=False, ax=ax, color='black', line_kws={'linewidth':1.5})
        ax.set_xlim(0,1); ax.set_ylim(0,1)
        ax.set_title(dataset, fontsize=18)
        ax.set_xlabel("MIC", fontsize=18)
        ax.set_ylabel("MF", fontsize=18)
        ax.tick_params(labelsize=14)
        if len(ds_df) > 1:
            slope, intercept, r_value, p_value, std_err = linregress(ds_df['mic'], ds_df['mf'])
            ax.text(0.05, 0.95, f"R²={r_value**2:.2f}\n$p$={p_value:.2f}", transform=ax.transAxes, verticalalignment='top', fontsize=14, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    for ax in axes[n_datasets:]:
        ax.set_visible(False)
    fig.suptitle(f"MIC vs. MF per Dataset{model_modifier}", fontsize=24, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    fig.subplots_adjust(hspace=0.2, wspace=0.15)
    plt.savefig(plots_dir / "7b_mic_vs_mf_per_ds_scatter.png", dpi=150)
    plt.close()

if __name__ == "__main__":
    args = parse_args()
    run(args)
