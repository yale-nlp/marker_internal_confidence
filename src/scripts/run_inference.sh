#!/bin/bash
# Command-line arguments
MODEL_NAME=$1
GMU=${2:-0.9}
TPS=${3:-1} 
SPLIT=${4:-"train"}
PROMPT=${5:-"sys1"}
NUM_SAMPLES=${6:-5000}
MAX_TOKENS=${7:-256}

datasets=(
    popqa
    selfaware
    sciq
    simpleqa
    hallueval
    mmlu
    arc_challenge
    superglue
    truthfulqa
    ambignq
    wnli
)

for dataset in "${datasets[@]}"; do

    python ./src/scripts/inference_and_score.py --model_name=$MODEL_NAME --dataset_name=$dataset  --gpu_mem_utilization=$GMU  --tensor_parallel_size=$TPS  --num_samples=$NUM_SAMPLES  --split=$SPLIT  --max_output_tokens=$MAX_TOKENS --sys_prompt=$PROMPT  --no_score

done

