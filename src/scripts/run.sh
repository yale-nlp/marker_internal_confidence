### Environment Setup

export PYTHONPATH=.
conda activate mic_env

export GOOGLE_API_KEY="your_api_key_here"
export OPENAI_KEY="your_api_key_here"
export HF_KEY="your_huggingface_key_here"

### Example Commands to Run MIC Computation & Analysis for Llama3.1-8B-Instruct
### Settings: 1 GPU, max 256 output tokens, max 5000 samples from the train split, using generic system prompt (sys1), marker threshold 10

export CUDA_VISIBLE_DEVICES=0

# Get Model Predictions
bash ./src/scripts/run_inference.sh meta-llama/Llama-3.1-8B-Instruct 0.9 1 train sys1 5000 256

# Score Model Predictions (1 = extract hedges, 2 = score internal confidence & accuracy, 3 = score decisiveness, 4 = compile & score cMFG)
bash ./src/scripts/run_scoring.sh meta-llama/Llama-3.1-8B-Instruct 0.9 1 train sys1 5000 256 1 
bash ./src/scripts/run_scoring.sh meta-llama/Llama-3.1-8B-Instruct 0.9 1 train sys1 5000 256 2
bash ./src/scripts/run_scoring.sh meta-llama/Llama-3.1-8B-Instruct 0.9 1 train sys1 5000 256 3
bash ./src/scripts/run_scoring.sh meta-llama/Llama-3.1-8B-Instruct 0.9 1 train sys1 5000 256 4

# Compute MICs + Statistics
python ./src/scripts/compute_statistics.py  --model_dir=./_results/meta_llama_Llama_3.1_8B_Instruct  --model_name=meta-llama/Llama-3.1-8B-Instruct --marker_count_threshold=10  --marker_count_for_plots=30 --split=train
python ./src/scripts/compute_statistics.py  --model_dir=./_results/meta_llama_Llama_3.1_8B_Instruct  --model_name=meta-llama/Llama-3.1-8B-Instruct --marker_count_threshold=10  --marker_count_for_plots=30 --split=test

# Compute Analysis Metrics (add flag --ignore_blank_hedge if ignoring the special marker <no_hedge>)
python ./src/scripts/compute_metrics.py  --df_dir=./_results/meta_llama_Llama_3.1_8B_Instruct/__marker_thresh_10/train/_dfs

# Plot KDE Analysis
python ./src/scripts/plot_kde.py  --csv_path=./_results/meta_llama_Llama_3.1_8B_Instruct/__marker_thresh_10/train/_dfs/shared_mic.csv --model_name="meta-llama/Llama-3.1-8B-Instruct"

