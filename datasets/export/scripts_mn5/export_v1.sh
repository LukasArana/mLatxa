#!/bin/bash
#SBATCH --job-name=export_v1
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --account=ehpc678
#SBATCH --partition=gpp
#SBATCH --qos=acc_ehpc
#SBATCH --time=24:00:00
#SBATCH --output=log/v1.out
#SBATCH --error=log/v1.err

# 1. Load modules and activate environment
module purge
module load miniforge cuda/12.6 cudnn

source /gpfs/apps/MN5/ACC/MINIFORGE/24.3.0-0/etc/profile.d/conda.sh
conda activate megatron-swift

# 2. Create cache directories
SCRATCH=~/scratch
mkdir -p $SCRATCH/hf_cache
mkdir -p $SCRATCH/modelscope_cache
mkdir -p $SCRATCH/tmp

# 3. Set Environment Variables (CRITICAL)
export HF_DATASETS_CACHE="$SCRATCH/hf_cache"
export HF_HOME="$SCRATCH/hf_cache"
export MODELSCOPE_CACHE="$SCRATCH/modelscope_cache"
export TMPDIR="$SCRATCH/tmp"

# 4. WANDB Configuration
export WANDB_PROJECT="qwen3-vl-finetuning"
export WANDB_ENTITY="USER"
export WANDB_DIR="$SCRATCH/wandb_logs"
mkdir -p $WANDB_DIR

# 5. Clean existing small caches (Free up space if root disk is already full)
rm -rf ~/.cache/huggingface
rm -rf ~/.cache/modelscope

# 6. Fix Z3 library path
export Z3_LIB_PATH=$(python -c 'import z3, os; print(os.path.join(z3.__path__[0], "lib"))')
export LD_LIBRARY_PATH="$Z3_LIB_PATH:$LD_LIBRARY_PATH"

# 7. Export settings
export IMAGE_MAX_TOKEN_NUM=124
export video_min_token_num=0
export video_max_token_num=0

echo $HOME

swift export \
    --model ~/scratch/basemodels/Qwen3.5-9B \
    --custom_dataset_info $HOME/mLatxa/datasets/export/configs_mn5/v1.json \
    --dataset magpie_qwen hplt_v1 booktegi cultura-x egunkaria wikipedia euscrawl_v1.1 oscar \
    --template qwen \
    --max_length 8192 \
    --dataset_num_proc 32 \
    --to_cached_dataset \
    --output_dir $SCRATCH/data/v1/

echo "Export complete!"
