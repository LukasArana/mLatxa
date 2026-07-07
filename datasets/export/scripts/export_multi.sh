#!/bin/bash
#SBATCH --job-name=export_multimodal%A_%a
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --partition=boost_usr_prod
#SBATCH --account=AIFAC_5C0_261
#SBATCH --mem=60GB
#SBATCH --time=24:00:00
#SBATCH --output=~/mLatxa/datasets/export/log/log_multi.out
#SBATCH --error=~/mLatxa/datasets/export/log/log_multi.err

# 1. Activate environment
source $WORK/environments/env_torch_2_9_megratron/bin/activate

# 2. Create cache directories on the LARGE volume
mkdir -p $WORK/hf_cache
mkdir -p $WORK/modelscope_cache
mkdir -p $WORK/tmp

# 3. Set Environment Variables (CRITICAL)
# Hugging Face Cache
export HF_DATASETS_CACHE="$WORK/hf_cache"
export HF_HOME="$WORK/hf_cache"

# ModelScope Cache (Specific to ms-swift)
export MODELSCOPE_CACHE="$WORK/modelscope_cache"

# System Temp Directory
export TMPDIR="$WORK/tmp"

#WANBD
export WANDB_PROJECT="qwen3-vl-finetuning"
export WANDB_ENTITY="USER"
export WANDB_DIR="$WORK/lukas/wandb_logs"
mkdir -p $WANDB_DIR
# 4. Clean existing small caches (Free up space if root disk is already full)
rm -rf ~/.cache/huggingface
rm -rf ~/.cache/modelscope

export IMAGE_MAX_TOKEN_NUM=0
# 5. Run your export command
IMAGE_MAX_TOKEN_NUM=0 \
swift export \
    --model $WORK/baseModels/Qwen3.5-2B \
    --custom_dataset_info configs/multi.json \
    --dataset magpie_qwen magpie_qwen_dup aldizkariak berria bog booktegi bopv botha cc-bsc oscar-05 oscar-06 cultura-x egunkaria euscrawl_v1 euscrawl_2023 euscrawl_2025 euscrawl_v2 finepdf fineweb hplt_v1 hplt_v2 opensubtitles parleus wikipedia zelaihandi gl-1 gl-2 gl-3-00 gl-3-01 gl-3-02 gl-3-03 gl-3-04 gl-3-05 gl-3-06 ca-00 ca-01 ca-02 ca-03 ca-04 ca-05 ca-06 \
    --template qwen \
    --max_length 8192 \
    --dataset_num_proc 32 \
    --to_cached_dataset \
    --output_dir $WORK/datasets/train/preprocessed/multi/