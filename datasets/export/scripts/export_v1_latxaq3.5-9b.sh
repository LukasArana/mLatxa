#!/bin/bash
#SBATCH --job-name=export_multimodal%A_%a
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --partition=boost_usr_prod
#SBATCH --account=AIFAC_5C0_261
#SBATCH --mem=60GB
#SBATCH --output=/leonardo/home/userexternal/asagasti/mLatxa/datasets/export/log/log.out
#SBATCH --error=/leonardo/home/userexternal/asagasti/mLatxa/datasets/export/log/log.err


# 1. Activate environment
source /leonardo_work/AIFAC_5C0_261/environments/env_torch_2_9_megratron/bin/activate

# 2. Create cache directories on the LARGE volume
mkdir -p /leonardo_work/AIFAC_5C0_261/hf_cache
mkdir -p /leonardo_work/AIFAC_5C0_261/modelscope_cache
mkdir -p /leonardo_work/AIFAC_5C0_261/tmp

# 3. Set Environment Variables (CRITICAL)
# Hugging Face Cache
export HF_DATASETS_CACHE="/leonardo_work/AIFAC_5C0_261/hf_cache"
export HF_HOME="/leonardo_work/AIFAC_5C0_261/hf_cache"

# ModelScope Cache (Specific to ms-swift)
export MODELSCOPE_CACHE="/leonardo_work/AIFAC_5C0_261/modelscope_cache_"

# System Temp Directory
export TMPDIR="/leonardo_work/AIFAC_5C0_261/tmp"

#WANBD
export WANDB_PROJECT="latxa-qwen3.5-9b"
export WANDB_ENTITY="asagasti"
export WANDB_DIR="/leonardo_work/AIFAC_5C0_261/aimar/wandb_logs"
mkdir -p $WANDB_DIR
# 4. Clean existing small caches (Free up space if root disk is already full)
rm -rf ~/.cache/huggingface
rm -rf ~/.cache/modelscope

# 5. Run your export command
swift export \
    --model /leonardo_work/AIFAC_5C0_261/baseModels/Qwen3.5-9B \
    --custom_dataset_info ../configs/v1.json \
    --dataset magpie_qwen hplt_v1 booktegi cultura-x egunkaria wikipedia euscrawl_v1.1 oscar \
    --template qwen \
    --max_length 8192 \
    --dataset_num_proc 16 \
    --to_cached_dataset \
    --output_dir /leonardo_work/AIFAC_5C0_261/datasets/preprocess_v1
