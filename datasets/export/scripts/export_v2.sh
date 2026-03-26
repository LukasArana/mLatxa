# 1. Activate environment
source /leonardo_work/AIFAC_5C0_261/environments/env_torch_2_9/bin/activate

# 2. Create cache directories on the LARGE volume
mkdir -p /leonardo_work/AIFAC_5C0_261/hf_cache
mkdir -p /leonardo_work/AIFAC_5C0_261/modelscope_cache
mkdir -p /leonardo_work/AIFAC_5C0_261/tmp

# 3. Set Environment Variables (CRITICAL)
# Hugging Face Cache
export HF_DATASETS_CACHE="/leonardo_work/AIFAC_5C0_261/hf_cache"
export HF_HOME="/leonardo_work/AIFAC_5C0_261/hf_cache"

# ModelScope Cache (Specific to ms-swift)
export MODELSCOPE_CACHE="/leonardo_work/AIFAC_5C0_261/modelscope_cache"

# System Temp Directory
export TMPDIR="/leonardo_work/AIFAC_5C0_261/tmp"

#WANBD
export WANDB_PROJECT="qwen3-vl-finetuning"
export WANDB_ENTITY="larana"
export WANDB_DIR="/leonardo_work/AIFAC_5C0_261/lukas/wandb_logs"
mkdir -p $WANDB_DIR
# 4. Clean existing small caches (Free up space if root disk is already full)
rm -rf ~/.cache/huggingface
rm -rf ~/.cache/modelscope

# 5. Run your export command
swift export \
    --model /leonardo_work/EUHPC_E04_042/BaseModels/Qwen3-VL-32B-Instruct \
    --custom_dataset_info configs/v2.json \
    --dataset aldizkariak berria bog booktegi bopv botha cc-bsc oscar-05 oscar-06 cultura-x egunkaria euscrawl_v1 euscrawl_2023 euscrawl_2025 euscrawl_v2 finepdf fineweb hplt_v1 hplt_v2 opensubtitles parleus wikipedia zelaihandi magpie_qwen magpie_llama \
    --template qwen \
    --max_length 8192 \
    --dataset_num_proc 32 \
    --to_cached_dataset \
    --output_dir /leonardo_work/AIFAC_5C0_261/datasets/train/preprocessed/latxa_v2/qwen32b/train/