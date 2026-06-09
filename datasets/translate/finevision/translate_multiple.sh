#!/bin/bash
#SBATCH --job-name=latxa-mt
#SBATCH --partition=boost_usr_prod        # Leonardo Booster; adjust for your cluster
#SBATCH --account=YOUR_ACCOUNT            # <-- set your project account
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4                      # 4x A100-64GB per Booster node -> TP=4
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=logs/mt_%A_%a.out
#SBATCH --error=logs/mt_%A_%a.err
# NOTE: the --array range is supplied by submit.sh, e.g. sbatch --array=0-99%20 translate.slurm

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