#!/bin/bash
#SBATCH --job-name=swift-multinode
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --time 24:00:00
#SBATCH --account=AIFAC_5C0_261
#SBATCH --partition=boost_usr_prod
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --exclusive

# --- Environment Activation ---
source $WORK/environments/env_torch_2_9_megratron2/bin/activate

# --- Directory Setup & Variables ---
export TMPDIR="$WORK/tmp"
export HF_HOME="$WORK/hf_cache"
export MODELSCOPE_CACHE="$WORK/modelscope_cache"
export WANDB_DIR="$WORK/lukas/wandb_logs"

mkdir -p $TMPDIR $HF_HOME $MODELSCOPE_CACHE $WANDB_DIR

export WANDB_MODE=offline
export WANDB_PROJECT="qwen3-vl-finetuning"
export WANDB_ENTITY="USER"

export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
export LANGUAGE=en_US.UTF-8
export OMP_NUM_THREADS=4
export NNODES=$SLURM_NNODES
export GPUS_PER_NODE=4

export MALLOC_ARENA_MAX=2
# --- Network & Distributed Config (Leonardo Specifics) ---
export PYTORCH_ALLOC_CONF="expandable_segments:True"

# Prevent crashing when loading heavy multimodal datasets
# Lowered from 28800 -> 600 so the watchdog fires before manual scancel,
# producing a real "rank X stuck in collective Y" line instead of silent hang.
export NCCL_TIMEOUT=1800
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# Diagnostic: surface NCCL hangs with a stack + flight-recorder dump
export TORCH_NCCL_BLOCKING_WAIT=0
export TORCH_NCCL_TRACE_BUFFER_SIZE=2000
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export NCCL_DEBUG=WARN
export CUDA_LAUNCH_BLOCKING=0

# Leonardo quirk: Prevent NICs from being used for inter-CPU communication
export NCCL_NET_DISABLE_INTRA=1

# InfiniBand setups for Leonardo Boost
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3
export NCCL_IB_GID_INDEX=3
export NCCL_NET_GDR_LEVEL=2

export SWIFT_PATCH_CONV3D=1
# Diagnostic: disabled to test whether MCore GDN compile path is the hang root.
# Re-enable if removing it does not change the symptom.
export SWIFT_USE_MCORE_GDN=1

# Get the Master Node's IP address directly (fixes hostname resolution issues in torchrun)
nodes=($(scontrol show hostnames $SLURM_JOB_NODELIST))
head_node=${nodes[0]}

export SKIP_MULTIMODAL_MTP_VALIDATION=1


# CRITICAL FOR TORCHRUN: Force Gloo (the rendezvous backend) to use the InfiniBand interface
export GLOO_SOCKET_IFNAME=ib0  # Change to eno1 if ib0 still times out

# Qwen3-VL specific variables
export video_min_token_num=0
export video_max_token_num=0
export FPS_MIN_FRAMES=0

export USE_MCORE_GDN=1
nvidia-smi topo -m
export IMAGE_MAX_TOKEN_NUM=128

MASTER_PORT=9327
MAIN_PROCESS_IP=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)

IMAGE_MAX_TOKEN_NUM=128 \
srun accelerate launch \
    --num_processes $(( $NNODES * $GPUS_PER_NODE )) \
    --num_machines $NNODES \
    --mixed_precision bf16 \
    --dynamo_backend "no" \
    --rdzv_backend c10d \
    --main_process_ip $MAIN_PROCESS_IP \
    --main_process_port $MASTER_PORT \
    --machine_rank $SLURM_NODEID \
    $HOME/ms-swift/swift/cli/_megatron/sft.py \
    --model $WORK/baseModels/Qwen3.5-9B \
    --save_safetensors false \
    --cached_dataset $WORK/datasets/train/preprocessed/multimodal_4_v2_eu/train/ \
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
    --output_dir $WORK/multimodalModels \
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
    --freeze_aligner false