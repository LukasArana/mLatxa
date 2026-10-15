#!/bin/bash
#SBATCH --job-name=export_v2_qwen35_qwen3vl
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --partition=boost_usr_prod
#SBATCH --account=AIFAC_5C0_261
#SBATCH --mem=60GB
#SBATCH --output=log/export_v2_qwen35_qwen3vl.out
#SBATCH --error=log/export_v2_qwen35_qwen3vl.err


# 1. Activate environment
source $WORK/environments/env_torch_2_9_megratron/bin/activate
set -eo pipefail

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
export WANDB_ENTITY="${WANDB_ENTITY:-USER}"
export WANDB_DIR="$WORK/lukas/wandb_logs"
mkdir -p $WANDB_DIR
# 4. Clean existing small caches (Free up space if root disk is already full)
rm -rf ~/.cache/huggingface
rm -rf ~/.cache/modelscope

# Same v2 sources as export_v2_qwen35.sh, tokenized for qwen3vl (caches are tokenizer-specific)
# 5. Run your export command
CFG=$(python $HOME/mLatxa/datasets/export/render_config.py $HOME/mLatxa/datasets/export/configs/v2_qwen.json)

swift export \
    --model ${EUHPC_WORK:-/leonardo_work/EUHPC_E04_042}/BaseModels/Qwen3-VL-32B-Instruct \
    --custom_dataset_info "$CFG" \
    --dataset qwen_35_conv aldizkariak berria bog booktegi bopv botha cc-bsc oscar-05 oscar-06 cultura-x egunkaria euscrawl_v1 euscrawl_2023 euscrawl_2025 euscrawl_v2 finepdf fineweb hplt_v1 hplt_v2 opensubtitles parleus wikipedia zelaihandi \
    --template qwen \
    --max_length 8192 \
    --dataset_num_proc 32 \
    --to_cached_dataset \
    --output_dir $WORK/datasets/train/preprocessed/v2_qwen35/