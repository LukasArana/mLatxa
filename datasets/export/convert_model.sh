#!/bin/bash
# Convert Megatron (mcore) weights -> HuggingFace safetensors using ms-swift
# Usage: bash convert_mcore_to_hf.sh <mcore_checkpoint_dir> <output_dir>

set -e

if [ -z "$1" ]; then
    echo "Usage: bash $0 <mcore_checkpoint_dir> [output_dir]"
    echo ""
    echo "  <mcore_checkpoint_dir>  Path to the Megatron checkpoint. Point to the PARENT"
    echo "                          directory of iter_xxx (the one containing"
    echo "                          latest_checkpointed_iteration.txt),"
    echo "                          e.g. megatron_output/Qwen2.5-7B-Instruct/vx-xxx"
    echo "  [output_dir]            Where to save HF safetensors (default: <mcore_checkpoint_dir>-hf)"
    exit 1
fi

module purge
module load miniforge cuda/12.6 cudnn

source /gpfs/apps/MN5/ACC/MINIFORGE/24.3.0-0/etc/profile.d/conda.sh
conda activate megatron-swift

MCORE_MODEL="$1"
OUTPUT_DIR="${2:-${MCORE_MODEL%/}-hf}"

if [ ! -d "$MCORE_MODEL" ]; then
    echo "Error: checkpoint directory not found: $MCORE_MODEL"
    exit 1
fi

echo "Converting: $MCORE_MODEL"
echo "Output dir: $OUTPUT_DIR"

CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
megatron export \
  --mcore_model "$MCORE_MODEL" \
  --to_hf true \
  --torch_dtype bfloat16 \
  --output_dir "$OUTPUT_DIR" \
  --tensor_model_parallel_size 4