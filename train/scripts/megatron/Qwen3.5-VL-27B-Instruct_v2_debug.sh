#!/bin/bash
#SBATCH --job-name=swift-multinode
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time 00:30:00
#SBATCH --account=AIFAC_5C0_261
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --exclusive
#SBATCH --requeue

# --- Environment Activation ---
source $WORK/environments/env_torch_2_9_megratron/bin/activate

# --- Directory Setup & Variables ---
export TMPDIR="$WORK/tmp"
export HF_HOME="$WORK/hf_cache"
export MODELSCOPE_CACHE="$WORK/modelscope_cache"
export WANDB_DIR="$WORK/lukas/wandb_logs"


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


# Diagnostic: surface NCCL hangs with a stack + flight-recorder dump
export PYTHONUNBUFFERED=1                          # flush logs instead of buffering them away
export NCCL_DEBUG=WARN                             # which collective stalled (noisy on 256 ranks, but worth it)
export TORCH_NCCL_TRACE_BUFFER_SIZE=2000           # flight recorder ON
export TORCH_NCCL_DUMP_ON_TIMEOUT=1                # dump the recorder when a rank times out
export NCCL_TIMEOUT=1800
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=0                  # keep 0 — do NOT also set this to 1, the two modes conflict
export CUDA_LAUNCH_BLOCKING=0

# Temporarily drop timeouts to 60 seconds for debugging
export NCCL_TIMEOUT=60
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=60

# Enable maximum distributed debugging verbosity
export TORCH_NCCL_ENABLE_COALESCING=0
# Tell PyTorch to dump the exact hang location when the 60s timeout hits
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=2000


export NCCL_DUMP_DIR="$WORK/lukas/nccl_dump_27"
mkdir -p $TMPDIR $HF_HOME $MODELSCOPE_CACHE $WANDB_DIR $NCCL_DUMP_DIR
export TORCH_NCCL_DEBUG_INFO_TEMP_FILE="${NCCL_DUMP_DIR}/rank_"               # keep 0 (1 serializes every kernel — kills a 256-GPU run)

export GLOO_TIMEOUT_SECONDS=1800  # 30 minutes for large multi-node setup

mkdir -p $TMPDIR $HF_HOME $MODELSCOPE_CACHE $WANDB_DIR $TORCH_NCCL_DEBUG_INFO_TEMP_FILE


# Get the Master Node's IP address directly (fixes hostname resolution issues in torchrun)
nodes=($(scontrol show hostnames $SLURM_JOB_NODELIST))
head_node=${nodes[0]}

export SKIP_MULTIMODAL_MTP_VALIDATION=1

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

export IMAGE_MAX_TOKEN_NUM=0

export NCCL_NET_DISABLE_INTRA=1

# InfiniBand setups for Leonardo Boost
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3
export NCCL_IB_GID_INDEX=3
export NCCL_NET_GDR_LEVEL=2

export SWIFT_PATCH_CONV3D=1
# Diagnostic: disabled to test whether MCore GDN compile path is the hang root.
# Re-enable if removing it does not change the symptom.
export SWIFT_USE_MCORE_GDN=1

nvidia-smi topo -m

export MASTER_PORT=9327
export MAIN_PROCESS_IP=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)


# --- Training Launch ---
echo "Launching $NNODES-node training on Head Node: $MAIN_PROCESS_IP at Port: $MASTER_PORT"


export ACCELERATE_FSDP_SHARDING_STRATEGY="1"
IMAGE_MAX_TOKEN_NUM=0 \
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
    --model $WORK/baseModels/Qwen3.5-0.8B \
    --save_safetensors false \
    --cached_dataset $WORK/datasets/train/preprocessed/v2/qwen32b/train/ \
    --load_from_cache_file true \
    --split_dataset_ratio 0.01 \
    --add_non_thinking_prefix true \
    --tensor_model_parallel_size 4 \
    --pipeline_model_parallel_size 1 \
    --micro_batch_size 1 \
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
    --loss_scale ignore_empty_think \
    --adam_beta2 0.95 \
    --adam_eps 1e-8 \
    --lr_decay_style cosine \
    --output_dir  $WORK/multimodalModels \
    --save_steps 100 \
    --max_length 8192 \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8 \
    --attention_backend flash \
    --no_load_optim false \
    --no_load_rng false \
    --save_total_limit 10 \
    --logging_steps 5 \
    --no_save_optim false \
    --freeze_llm false \
    --freeze_vit true \
    --freeze_aligner false \
    --padding_free true \
    --use_distributed_optimizer \
    --use_precision_aware_optimizer true \
    --optimizer_cpu_offload true