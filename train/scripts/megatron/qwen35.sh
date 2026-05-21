#!/bin/bash
#SBATCH --job-name=swift-qwen3.5
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --account=AIFAC_5C0_261
#SBATCH --partition=boost_usr_prod
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --exclusive

# --- Environment ---
source /leonardo_work/AIFAC_5C0_261/environments/env_torch_2_9_megratron/bin/activate

# --- Caches ---
export TMPDIR="/leonardo_work/AIFAC_5C0_261/tmp"
export HF_HOME="/leonardo_work/AIFAC_5C0_261/hf_cache"
export MODELSCOPE_CACHE="/leonardo_work/AIFAC_5C0_261/modelscope_cache"
export WANDB_DIR="/leonardo_work/AIFAC_5C0_261/lukas/wandb_logs"
mkdir -p $TMPDIR $HF_HOME $MODELSCOPE_CACHE $WANDB_DIR

# --- W&B ---
export WANDB_MODE=offline
export WANDB_PROJECT="qwen3.5-finetuning"
export WANDB_ENTITY="larana"

# --- Locale / threading ---
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
export OMP_NUM_THREADS=4

# --- Memory allocator (correct variable name!) ---
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# --- NCCL / InfiniBand on Leonardo Boost ---
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3
export NCCL_IB_GID_INDEX=3
export NCCL_NET_GDR_LEVEL=2
export NCCL_DEBUG=WARN
export GLOO_SOCKET_IFNAME=ib0
export NCCL_TIMEOUT=1800
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# --- Qwen3.5 multimodal token limits (per ms-swift docs) ---
export IMAGE_MAX_TOKEN_NUM=1024
export VIDEO_MAX_TOKEN_NUM=128
export FPS_MAX_FRAMES=12

# --- Distributed launch ---
export NNODES=$SLURM_NNODES
export GPUS_PER_NODE=4
MASTER_PORT=9327
MAIN_PROCESS_IP=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)

srun accelerate launch \
    --num_processes $(( $NNODES * $GPUS_PER_NODE )) \
    --num_machines $NNODES \
    --mixed_precision bf16 \
    --dynamo_backend no \
    --rdzv_backend c10d \
    --main_process_ip $MAIN_PROCESS_IP \
    --main_process_port $MASTER_PORT \
    --machine_rank $SLURM_NODEID \
    /leonardo/home/userexternal/laranaga/ms-swift/swift/cli/_megatron/sft.py \
    --model /leonardo_work/EUHPC_E04_042/BaseModels/Qwen3.5-9B-Instruct \
    --cached_dataset /leonardo_work/AIFAC_5C0_261/datasets/train/preprocessed/multimodal_v1/train \
    --load_from_cache_file true \
    --split_dataset_ratio 0.01 \
    --tensor_model_parallel_size 4 \
    --pipeline_model_parallel_size 1 \
    --sequence_parallel true \
    --micro_batch_size 1 \
    --global_batch_size 512 \
    --max_length 4096 \
    --packing true \
    --padding_free true \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --attention_backend flash \
    --cross_entropy_loss_fusion true \
    --finetune true \
    --freeze_llm false \
    --freeze_vit true \
    --freeze_aligner false \
    --add_non_thinking_prefix true \
    --loss_scale ignore_empty_think \
    --num_train_epochs 2 \
    --lr 1e-5 \
    --min_lr 1e-6 \
    --lr_warmup_fraction 0.05 \
    --lr_decay_style cosine \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --adam_eps 1e-8 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 8 \
    --output_dir /leonardo_work/AIFAC_5C0_261/multimodalModels \
    --save_steps 250 \
    --save_total_limit 2 \
    --save_safetensors true \
    --logging_steps 5