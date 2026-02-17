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

# 4. Clean existing small caches (Free up space if root disk is already full)
rm -rf ~/.cache/huggingface
rm -rf ~/.cache/modelscope
rm -rf /tmp/*

# 5. Run your export command
swift export \
    --model /leonardo_work/EUHPC_E04_042/BaseModels/Qwen3-VL-4B-Instruct \
    --dataset /leonardo_work/AIFAC_5C0_261/datasets/train/finevisionjsonl/train.04_clean-01.onlytext.jsonl \
    --dataset /leonardo_work/AIFAC_5C0_261/datasets/train/finevisionjsonl/wikipedia.train.eu.04_clean-01.onlytext.jsonl \
    --dataset /leonardo_work/AIFAC_5C0_261/datasets/train/finevisionjsonl/eu.train.04_clean-01.onlytext.jsonl \
    --dataset /leonardo_work/AIFAC_5C0_261/datasets/train/finevisionjsonl/colossal_oscar_2023-14_eu.train.part-0001-of-0001.shuffled.04_clean-01.onlytext.jsonl \
    --dataset /leonardo_work/AIFAC_5C0_261/datasets/train/finevisionjsonl/botha_eu_18_09.train.04_clean-01.onlytext.jsonl \
    --dataset /leonardo_work/AIFAC_5C0_261/datasets/train/finevisionjsonl/bopv_eu_18_09.train.04_clean-01.onlytext.jsonl \
    --dataset /leonardo_work/AIFAC_5C0_261/datasets/train/finevisionjsonl/booktegi-bsc.train.04_clean-01.onlytext.jsonl \
    --dataset /leonardo_work/AIFAC_5C0_261/datasets/train/finevisionjsonl/bog_euskera_18_09.train.04_clean-01.onlytext.jsonl \
    --dataset /leonardo_work/AIFAC_5C0_261/datasets/train/finevisionjsonl/berria-202509.train.04_clean-01.onlytext.jsonl \
    --dataset /leonardo_work/AIFAC_5C0_261/datasets/train/finevisionjsonl/full.train.04_clean-01.onlytext.jsonl \
    --dataset /leonardo_work/EUHPC_E04_042/datasets/InstructDatasets/Magpie-Llama-3.1-70B-Instruct-Filtered-1M.jsonl \
    --exist_ok \
    --dataset_num_proc 16 \
    --split_dataset_ratio 0.1 \
    --to_cached_dataset \
    --output_dir /leonardo_work/AIFAC_5C0_261/datasets/train/preprocessed/latxa_v2