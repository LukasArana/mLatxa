#!/bin/bash
#SBATCH --job-name=convert_moe
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --time 1:00:00
#SBATCH --account=ehpc678
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --output=models/convert/logs/%j.out
#SBATCH --error=models/convert/logs/%j.err

if [ -z "$1" ]; then
    echo "Usage: sbatch $0 <mcore_checkpoint_dir> [output_dir]"
    echo ""
    echo "  <mcore_checkpoint_dir>  Path to the Megatron checkpoint. Point to the PARENT"
    echo "                          directory of iter_xxx (the one containing"
    echo "                          latest_checkpointed_iteration.txt),"
    echo "                          e.g. \$SCRATCH/Latxa3.5/v8-xxx/checkpoint-12937"
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

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29500
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

srun --cpu-bind=none bash -c '
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NNODES=$SLURM_NNODES \
NODE_RANK=$SLURM_PROCID \
NPROC_PER_NODE=4 \
megatron export \
  --mcore_model "'"$MCORE_MODEL"'" \
  --to_hf true \
  --torch_dtype bfloat16 \
  --output_dir "'"$OUTPUT_DIR"'" \
  --tensor_model_parallel_size 1 \
  --pipeline_model_parallel_size 4 \
  --expert_model_parallel_size 4 \
  --expert_tensor_parallel_size 1 \
  --save_safetensors true
'