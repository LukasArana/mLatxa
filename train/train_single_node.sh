source /leonardo_work/AIFAC_5C0_261/environments/env_torch_2_9/bin/activate
# 2. Create cache directories on the LARGE volume
mkdir -p /leonardo_work/AIFAC_5C0_261/hf_cache
mkdir -p /leonardo_work/AIFAC_5C0_261/modelscope_cache
mkdir -p /leonardo_work/AIFAC_5C0_261/tmp

export SWIFT_PATCH_CONV3D=1
export WANDB_MODE=offline
export WANDB_PROJECT="qwen3-vl-finetuning"
export WANDB_ENTITY="larana"  # Optional
export WANDB_DIR="/leonardo_work/AIFAC_5C0_261/lukas/wandb_logs"
mkdir -p $WANDB_DIR

# Optional: Set a specific run name
export WANDB_NAME="qwen3-4b-full-train-$(date +%Y%m%d_%H%M)"
# 3. Set Environment Variables (CRITICAL)
# Hugging Face Cache
export HF_DATASETS_CACHE="/leonardo_work/AIFAC_5C0_261/hf_cache"
export HF_HOME="/leonardo_work/AIFAC_5C0_261/hf_cache"
# ModelScope Cache (Specific to ms-swift)
export MODELSCOPE_CACHE="/leonardo_work/AIFAC_5C0_261/modelscope_cache"
# System Temp Directory
export TMPDIR="/leonardo_work/AIFAC_5C0_261/tmp"

export OMP_NUM_THREADS=8

# 4. Clean existing small caches (Free up space if root disk is already full)
rm -rf ~/.cache/huggingface
rm -rf ~/.cache/modelscope
rm -rf /tmp/*

torchrun --nproc_per_node=4 \
    $(which swift) sft \
    --model /leonardo_work/EUHPC_E04_042/BaseModels/Qwen3-VL-4B-Instruct \
    --train_type full \
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
    --torch_dtype bfloat16 \
    --num_train_epochs 4 \
    --load_from_cache_file true \
    --split_dataset_ratio 0.001 \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 2 \
    --dataset_num_proc 8 \
    --gradient_checkpointing true \
    --save_total_limit 2 \
    --logging_steps 1 \
    --output_dir /leonardo_work/AIFAC_5C0_261/lukas/msoutput \
    --system 'You are a helpful assistant.' \
    --warmup_ratio 0.05 \
    --max_length 481926 \
    --packing \
    --freeze_llm false \
    --freeze_vit true \
    --vit_gradient_checkpointing false \
    --freeze_aligner false \
    --attn_impl flash_attention_2 \
    --dataloader_num_workers 7 \
    --model_author swift \
    --deepspeed zero2 \
    --cached_dataset /leonardo_work/AIFAC_5C0_261/datasets/train/preprocessed/latxa_v2/train \
    --load_from_cache_file true \
