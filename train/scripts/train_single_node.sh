source /leonardo_work/AIFAC_5C0_261/environments/env_torch_2_9/bin/activate
# 2. Create cache directories on the LARGE volume
mkdir -p /leonardo_work/AIFAC_5C0_261/hf_cache
mkdir -p /leonardo_work/AIFAC_5C0_261/modelscope_cache
mkdir -p /leonardo_work/AIFAC_5C0_261/tmp

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
export SWIFT_PATCH_CONV3D=1
export OMP_NUM_THREADS=8

# 4. Clean existing small caches (Free up space if root disk is already full)
rm -rf ~/.cache/huggingface
rm -rf ~/.cache/modelscope
rm -rf /tmp/*

torchrun --nproc_per_node=4 $(which swift) sft --config /leonardo/home/userexternal/laranaga/mLatxa/train/configs/train_32B_cache.yaml