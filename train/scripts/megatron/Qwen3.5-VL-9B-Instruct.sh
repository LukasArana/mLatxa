#!/bin/bash
#SBATCH --job-name=swift-multinode
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --time 24:00:00
#SBATCH --account=AIFAC_5C0_261
#SBATCH --partition=boost_usr_prod
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --exclusive

# --- Environment Activation ---
source /leonardo_work/AIFAC_5C0_261/environments/env_torch_2_9_megratron/bin/activate

# --- Directory Setup & Variables ---
export TMPDIR="/leonardo_work/AIFAC_5C0_261/tmp"
export HF_HOME="/leonardo_work/AIFAC_5C0_261/hf_cache"
export MODELSCOPE_CACHE="/leonardo_work/AIFAC_5C0_261/modelscope_cache"
export WANDB_DIR="/leonardo_work/AIFAC_5C0_261/lukas/wandb_logs"

mkdir -p $TMPDIR $HF_HOME $MODELSCOPE_CACHE $WANDB_DIR

export WANDB_MODE=offline
export WANDB_PROJECT="qwen3-vl-finetuning"
export WANDB_ENTITY="larana"

export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
export LANGUAGE=en_US.UTF-8
export OMP_NUM_THREADS=16
export NNODES=$SLURM_NNODES
export GPUS_PER_NODE=4

# --- Network & Distributed Config (Leonardo Specifics) ---
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Prevent crashing when loading heavy multimodal datasets
export NCCL_TIMEOUT=28800
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=28800
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# Leonardo quirk: Prevent NICs from being used for inter-CPU communication
export NCCL_NET_DISABLE_INTRA=1

# InfiniBand setups for Leonardo Boost
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3
export NCCL_IB_GID_INDEX=3
export NCCL_NET_GDR_LEVEL=2
export SWIFT_PATCH_CONV3D=1

# Get the Master Node's IP address directly (fixes hostname resolution issues in torchrun)
nodes=($(scontrol show hostnames $SLURM_JOB_NODELIST))
head_node=${nodes[0]}

# Added -N 1 -n 1 for cleaner execution
head_node_ip=$(srun -N 1 -n 1 -w $head_node hostname -I | grep -oE '10\.[0-9]+\.[0-9]+\.[0-9]+' | head -n 1)

#export MASTER_ADDR=$head_node_ip
# Deterministic port based on SLURM_JOB_ID to prevent race conditions
#export MASTER_PORT=$(( 10000 + ($SLURM_JOB_ID % 50000) ))

# CRITICAL FOR TORCHRUN: Force Gloo (the rendezvous backend) to use the InfiniBand interface
export GLOO_SOCKET_IFNAME=ib0  # Change to eno1 if ib0 still times out

# Qwen3-VL specific variables
export video_min_token_num=0
export video_max_token_num=0

nvidia-smi topo -m

export MASTER_PORT=9327
export MAIN_PROCESS_IP=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)


# --- Training Launch ---
echo "Launching $NNODES-node training on Head Node: $MAIN_PROCESS_IP at Port: $MASTER_PORT"

# For more information on multi-node training launch methods, refer to:
# https://github.com/modelscope/ms-swift/tree/main/examples/train/multi-node

MASTER_PORT=9327
MAIN_PROCESS_IP=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export ACCELERATE_FSDP_SHARDING_STRATEGY="1"
srun accelerate launch \
    --num_processes $(( $NNODES * $GPUS_PER_NODE )) \
    --num_machines $NNODES \
    --mixed_precision bf16 \
    --dynamo_backend "no" \
    --rdzv_backend c10d \
    --main_process_ip $MAIN_PROCESS_IP \
    --main_process_port $MASTER_PORT \
    --machine_rank $SLURM_NODEID \
    /leonardo/home/userexternal/laranaga/ms-swift/swift/cli/_megatron/sft.py \
    --model /leonardo_work/EUHPC_E04_042/BaseModels/Qwen3.5-9B-Instruct \
    --save_safetensors true \
    --cached_dataset /leonardo_work/AIFAC_5C0_261/datasets/train/preprocessed/latxa_v2/qwen32b/train \
    --load_from_cache_file true \
    --split_dataset_ratio 0.01 \
    --tensor_model_parallel_size 2 \
    --pipeline_model_parallel_size 1 \
    --micro_batch_size 4 \
    --packing true \
    --global_batch_size 512 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --num_train_epochs 4 \
    --finetune false \
    --cross_entropy_loss_fusion true \
    --lr 0.00001 \
    --lr_warmup_fraction 0.05 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --adam_eps 1e-8 \
    --lr_decay_style cosine \
    --lr 0.00001 \
    --output_dir /leonardo_work/AIFAC_5C0_261/multimodalModels \
    --save_steps 250 \
    --max_length 8192 \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8 \
    --sequence_parallel true \
    --attention_backend flash \
    --no_load_optim false \
    --no_load_rng false \
    --save_total_limit 2 \
    --overlap_param_gather true \
    --overlap_grad_reduce true \
    --logging_steps 5 \
    --no_save_optim false \
    --freeze_llm false \
    --freeze_vit true \
    --freeze_aligner false \
    --dist_ckpt_save_pre_mcore_014 true