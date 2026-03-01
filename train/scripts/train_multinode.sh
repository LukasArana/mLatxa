#!/bin/bash
#SBATCH --job-name=swift-multinode
#SBATCH --nodes=16
#SBATCH --cpus-per-task=16
#SBATCH --nodes=64
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --account AIFAC_5C0_261
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --output=logs/%j.out
#SBATCH --requeue
#SBATCH --mem=494000MB

# --- Directory Setup ---
mkdir -p /leonardo_work/AIFAC_5C0_261/hf_cache
mkdir -p /leonardo_work/AIFAC_5C0_261/modelscope_cache
mkdir -p /leonardo_work/AIFAC_5C0_261/tmp

# --- Environment Variables ---
export WANDB_MODE=offline
export WANDB_PROJECT="qwen3-vl-finetuning"
export WANDB_ENTITY="larana"
export WANDB_DIR="/leonardo_work/AIFAC_5C0_261/lukas/wandb_logs"
mkdir -p $WANDB_DIR

export HF_HOME="/leonardo_work/AIFAC_5C0_261/hf_cache"
export MODELSCOPE_CACHE="/leonardo_work/AIFAC_5C0_261/modelscope_cache"
export TMPDIR="/leonardo_work/AIFAC_5C0_261/tmp"
export OMP_NUM_THREADS=8

# --- Network & Distributed Config ---
# Get the first node's name for the Master
nodes=($(scontrol show hostnames $SLURM_JOB_NODELIST))
head_node=${nodes[0]}

export MASTER_ADDR=$head_node
export MASTER_PORT=29505
export NCCL_TIMEOUT=18000

# Fix the deprecated warning from your logs
export PYTORCH_ALLOC_CONF="expandable_segments:True"

# Leonardo Boost Specific NCCL tuning
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3
#export NCCL_SOCKET_IFNAME=hsn0
export NCCL_DEBUG=INFO
export SWIFT_PATCH_CONV3D=1


# Load qwen3-vl specific environment variables 8187712460
export video_min_token_num=0
export video_max_token_num=0
# --- Environment Activation ---
source /leonardo_work/AIFAC_5C0_261/environments/env_torch_2_9/bin/activate

# --- Training Launch ---
# Key Fix: We pass --node_rank=$SLURM_PROCID so each node knows its place.
# Key Fix: We use --rdzv_endpoint to ensure all nodes find the Master.
srun --nodes=$SLURM_NNODES --ntasks-per-node=1 \
    torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=4 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    --node_rank=$SLURM_PROCID \
    $(which swift) sft --config /leonardo/home/userexternal/laranaga/mLatxa/train/configs/train_32B_cache.yaml