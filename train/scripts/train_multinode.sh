#!/bin/bash
#SBATCH --job-name=swift-multinode
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --account AIFAC_5C0_261
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --output=logs/%j.out
#SBATCH --requeue

# --- Network & Distributed Config (From Script 2) ---
nodes=($(scontrol show hostnames $SLURM_JOB_NODELIST))
head_node=${nodes[0]}
# Specifically grab the internal high-speed network IP
head_node_ip=$(srun -w $head_node hostname -i | grep -oE '10\.[0-9]+\.[0-9]+\.[0-9]+' | head -n 1)

export MASTER_ADDR=$head_node_ip
export MASTER_PORT=29502
export NCCL_TIMEOUT=18000
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Leonardo-specific IB and NCCL tuning
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3
export NCCL_SOCKET_IFNAME=hsn0,ib0
export NCCL_NET_DISABLE_INTRA=1
export NCCL_DEBUG=INFO

# --- Environment Setup ---
source /leonardo_work/AIFAC_5C0_261/environments/env_torch_2_9/bin/activate

# --- Training Launch ---
# Using srun to wrap torchrun ensures proper process tracking across nodes
srun --nodes=$SLURM_NNODES --ntasks-per-node=1 \
    torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=4 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    $(which swift) sft --config /leonardo/home/userexternal/laranaga/mLatxa/train/configs/train_32B_cache.yaml