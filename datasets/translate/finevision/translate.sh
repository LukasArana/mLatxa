#!/bin/bash

set -euo pipefail

# ---- paths (edit these) -----------------------------------------------------
INPUT=/home/ehu/ehu152297/scratch/data/leonardo/jsonl/train_fixed_2.jsonl
OUTPUT_DIR=/home/ehu/ehu152297/scratch/data/leonardo/conversations_eu
SAMPLES_PER_JOB=20000
export HF_HOME=/home/ehu/ehu152297/scratch/hf_cache   # shared cache so models download once

# ---- environment ------------------------------------------------------------
module purge
module load miniforge cuda/12.6 cudnn

source activate megatron-swift


export TP_SIZE=4
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false

mkdir -p logs "$OUTPUT_DIR"

python3 finevision/translate.py \
    --input "$INPUT" \
    --output-dir "$OUTPUT_DIR" \
    --samples-per-job "$SAMPLES_PER_JOB" \
    --job-index 0 \
    --model HiTZ/Latxa-Llama-3.1-70B-Instruct \
    --tensor-parallel-size "$TP_SIZE" \
    --max-model-len 8192 \
    --max-num-seqs 256 \
    --gpu-memory-utilization 0.90 \