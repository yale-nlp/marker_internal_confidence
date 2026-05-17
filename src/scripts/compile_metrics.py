import json
import glob
import os
import re
import pandas as pd

# Find all _scores.json files
RESULTS_DIR = "./_results"
pattern = os.path.join(RESULTS_DIR, "*", "__marker_thresh_*", "_scores", "_scores.json")
files = glob.glob(pattern)

OUTPUT_DIR = os.path.join(RESULTS_DIR, "_compiled_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

if not files:
    print("No files found. Check the path pattern.")
    print(f"Searched: {pattern}")
    exit(1)

print(f"Found {len(files)} files")

# Parse each file
records = []
for fpath in files:
    # Extract model name and threshold from path
    # Path: .../MODELNAME/__marker_thresh_N/_scores/_scores.json
    parts = fpath.split(os.sep)
    scores_idx = parts.index("_scores")
    thresh_part = parts[scores_idx - 1]   # __marker_thresh_N
    model_name = parts[scores_idx - 2]    # MODELNAME

    match = re.search(r"__marker_thresh_(\d+)", thresh_part)
    if not match:
        print(f"Could not parse threshold from: {thresh_part}, skipping {fpath}")
        continue
    thresh = int(match.group(1))

    with open(fpath) as f:
        data = json.load(f)

    # Flatten: skip list-valued keys (like mac_spear_stats etc.)
    flat = {"model": model_name}
    for k, v in data.items():
        if not isinstance(v, list):
            flat[k] = v

    flat["_thresh"] = thresh
    records.append(flat)

if not records:
    print("No records parsed.")
    exit(1)

df = pd.DataFrame(records)

# Split by threshold and save
for thresh, group in df.groupby("_thresh"):
    out = group.drop(columns=["_thresh"]).set_index("model").sort_index()
    out_path = os.path.join(OUTPUT_DIR, f"results_thresh{thresh}.csv")
    out.to_csv(out_path)
    print(f"Saved {out_path}  ({len(out)} models, {len(out.columns)} columns)")
