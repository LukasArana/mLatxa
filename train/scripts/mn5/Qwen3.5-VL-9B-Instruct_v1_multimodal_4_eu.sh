#!/bin/bash
#SBATCH --job-name=swift-multinode
#SBATCH --nodes=32
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

# Load modules on the head node
module purge
module load miniforge cuda/12.6 cudnn

# Activate env on the head node (essential for Z3 path lookup)

export SCRATCH=~/scratch

# Fix the Z3 library path dynamically and export it so workers inherit it
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
export WANDB_ENTITY="USER"

export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
export LANGUAGE=en_US.UTF-8
export OMP_NUM_THREADS=16
export NNODES=$SLURM_NNODES
export GPUS_PER_NODE=4

# Pre-calculate the total processes so the worker node doesn't have to evaluate math
export TOTAL_PROCESSES=$(( NNODES * GPUS_PER_NODE ))

# --- Network & Distributed Config ---
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NCCL_TIMEOUT=28800
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=28800
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export SWIFT_PATCH_CONV3D=1

# Extract high-speed IB IP and assign it EXACTLY ONCE
nodes=($(scontrol show hostnames $SLURM_JOB_NODELIST))
head_node=${nodes[0]}
export MAIN_PROCESS_IP=$(srun -N 1 -n 1 -w $head_node hostname -I | grep -oE '10\.[0-9]+\.[0-9]+\.[0-9]+' | head -n 1)
export MASTER_PORT=9327

export GLOO_SOCKET_IFNAME=ib0
export video_min_token_num=0
export video_max_token_num=0
export IMAGE_MAX_TOKEN_NUM=128
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

# CRITICAL FIX: Ensure user-site packages (like accelerate and z3) are NOT blocked
unset PYTHONNOUSERSITE

echo "Launching $NNODES-node training on Head Node: $MAIN_PROCESS_IP at Port: $MASTER_PORT"
echo "Total processes: $TOTAL_PROCESSES"
# Resume from Leonardo checkpoint (rsync to MN5 first):
# --mcore_model $SCRATCH/multimodalModels/v304-20260721-140858/checkpoint-1250/

# --- Training Launch ---
# Notice the single quotes ('...') around the bash -c command.
srun --cpu-bind=none bash -c '
    # Clean the worker environment
    module purge
    module load miniforge cuda/12.6 cudnn

    # Initialize Conda cleanly ON THE WORKER NODE
    source /gpfs/apps/MN5/ACC/MINIFORGE/24.3.0-0/etc/profile.d/conda.sh
    conda activate megatron-swift
    export IMAGE_MAX_TOKEN_NUM=128

    # Use the absolute Python path and call accelerate as a module.
    /gpfs/apps/MN5/ACC/MINIFORGE/24.3.0-0/envs/megatron-swift/bin/python -m accelerate.commands.launch \
        --num_processes $TOTAL_PROCESSES \
        --num_machines $NNODES \
        --mixed_precision bf16 \
        --dynamo_backend "no" \
        --rdzv_backend c10d \
        --main_process_ip $MAIN_PROCESS_IP \
        --main_process_port $MASTER_PORT \
        --machine_rank $SLURM_NODEID \
        $HOME/MS-Swift/swift/cli/_megatron/sft.py \
        --model $SCRATCH/basemodels/Qwen3.5-9B \
        --mcore_model $SCRATCH/multimodalModels/v150-20260825-133949/checkpoint-250 \
        --save_safetensors false \
        --cached_dataset $SCRATCH/data/v1_multimodal_4_eu/train \
        --load_from_cache_file true \
        --add_non_thinking_prefix true \
        --loss_scale ignore_empty_think \
        --split_dataset_ratio 0.01 \
        --tensor_model_parallel_size 4 \
        --torch_dtype bfloat16 \
        --micro_batch_size 1 \
        --packing true \
        --padding_free true \
        --global_batch_size 512 \
        --recompute_granularity full \
        --recompute_num_layers 1 \
        --recompute_method uniform \
        --num_train_epochs 2 \
        --finetune false \
        --cross_entropy_loss_fusion true \
        --lr 0.00001 \
        --lr_warmup_fraction 0.05 \
        --adam_beta1 0.9 \
        --adam_beta2 0.95 \
        --adam_eps 1e-8 \
        --lr_decay_style cosine \
        --output_dir $SCRATCH/multimodalModels \
        --save_steps 250 \
        --max_length 8192 \
        --dataloader_num_workers 4 \
        --dataset_num_proc 4 \
        --sequence_parallel true \
        --attention_backend flash \
        --no_load_optim false \
        --no_load_rng false \
        --save_total_limit 2 \
        --overlap_param_gather false \
        --overlap_grad_reduce false \
        --logging_steps 10 \
        --no_save_optim false \
        --freeze_llm false \
        --freeze_vit true \
        --freeze_aligner false \
        --fp8_format hybrid \
        --fp8_recipe delayed \
        --fp8_param_gather false
'
