#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
# --output / --error are set by translate_all.sh so each carries its [start,end) indexes.

set -euo pipefail

# ---- slice this job translates ---------------------------------------------
START="${1:?usage: translate.sh START END MODEL LANGUAGE [INPUT]}"   # inclusive, 0-based
END="${2:?usage: translate.sh START END MODEL LANGUAGE [INPUT]}"      # exclusive
MODEL="${3:?usage: translate.sh START END MODEL LANGUAGE [INPUT]}"
LANGUAGE="${4:?usage: translate.sh START END MODEL LANGUAGE [INPUT]}"

# ---- paths (edit these) -----------------------------------------------------
INPUT="${5:-$WORK/datasets/train/finevisionProcessed/train.jsonl}"
OUTPUT_DIR="$(dirname "$INPUT")/"
export HF_HOME=$WORK/hf_cache   # shared cache so models download once

source $WORK/environments/vllm_19/bin/activate

export TP_SIZE=4
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false

mkdir -p logs "$OUTPUT_DIR"

python3 $HOME/mLatxa/datasets/translate/finevision/translate.py \
    --input "$INPUT" \
    --output-dir "$OUTPUT_DIR" \
    --start "$START" \
    --end "$END" \
    --model "$MODEL" \
    --language "$LANGUAGE" \
    --tensor-parallel-size "$TP_SIZE" \
    --max-model-len 8192 \
    --max-num-seqs 64 \
    --gpu-memory-utilization 0.90
