#!/bin/bash
#SBATCH --job-name=train_single_node
#SBATCH --nodes=1
#SBATCH --cpus-per-task=80
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --time 24:00:00
#SBATCH --account=ehpc678
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --exclusive

# Load modules on the compute node
module purge
module load miniforge cuda/12.6 cudnn

# Activate env
source /gpfs/apps/MN5/ACC/MINIFORGE/24.3.0-0/etc/profile.d/conda.sh
conda activate megatron-swift

export SCRATCH=~/scratch

# Fix the Z3 library path dynamically
export Z3_LIB_PATH=$(python -c 'import z3, os; print(os.path.join(z3.__path__[0], "lib"))')
export LD_LIBRARY_PATH="$Z3_LIB_PATH:$LD_LIBRARY_PATH"

# --- Directory Setup & Variables ---
export TMPDIR="$SCRATCH/tmp"
export HF_HOME="$SCRATCH/hf_cache"
export MODELSCOPE_CACHE="$SCRATCH/modelscope_cache"
export WANDB_DIR="$SCRATCH/lukas/wandb_logs"

mkdir -p $TMPDIR $HF_HOME $MODELSCOPE_CACHE $WANDB_DIR

export WANDB_MODE=offline
export WANDB_PROJECT="qwen3-vl-finetuning"
export WANDB_NAME="${SLURM_JOB_NAME}_${SLURM_JOB_ID}"

export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
export LANGUAGE=en_US.UTF-8
export OMP_NUM_THREADS=16
export NNODES=1
export GPUS_PER_NODE=4
export NPROC_PER_NODE=$GPUS_PER_NODE

# --- Network & Distributed Config ---
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NCCL_TIMEOUT=28800
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=28800
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export SWIFT_PATCH_CONV3D=1

export MASTER_PORT=9327
export video_min_token_num=0
export video_max_token_num=0

echo "Launching single-node training ($NNODES node, $GPUS_PER_NODE GPUs)"

# --- Training Launch ---
megatron sft \
        --model $SCRATCH/basemodels/Llama3.1 \
        --model_type llama \
        --template llama3 \
        --save_safetensors true \
        --cached_dataset $SCRATCH/data/v2_llama31/train \
        --load_from_cache_file true \
        --split_dataset_ratio 0.01 \
        --tensor_model_parallel_size 1 \
        --pipeline_model_parallel_size 1 \
        --micro_batch_size 1 \
        --global_batch_size 4 \
        --recompute_granularity full \
        --packing true \
        --recompute_method uniform \
        --recompute_num_layers 1 \
        --num_train_epochs 1 \
        --finetune false \
        --cross_entropy_loss_fusion true \
        --lr 0.00001 \
        --lr_warmup_fraction 0.05 \
        --adam_beta1 0.9 \
        --adam_beta2 0.95 \
        --adam_eps 1e-8 \
        --lr_decay_style cosine \
        --output_dir $SCRATCH/multimodalModels \
        --save_steps 1 \
        --max_length 8192 \
        --dataloader_num_workers 1 \
        --dataset_num_proc 1 \
        --sequence_parallel true \
        --attention_backend flash \
        --no_load_optim false \
        --no_load_rng false \
        --save_total_limit 2 \
        --overlap_param_gather false \
        --overlap_grad_reduce true \
        --logging_steps 5 \
        --no_save_optim false \
        --freeze_llm true \
        --freeze_vit true \
        --freeze_aligner false \
        --dist_ckpt_optim_fully_reshardable \
        --fp8_format hybrid \
        --fp8_recipe delayed \
        --fp8_param_gather false
