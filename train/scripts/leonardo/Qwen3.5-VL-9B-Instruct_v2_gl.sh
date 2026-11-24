#!/bin/bash
# Leonardo port of scripts/mn5/Qwen3.5-VL-9B-Instruct_v2_gl.sh (generated 2026-11-24).
# Training args are verbatim from MN5 except: paths, and save_steps capped at 250
# (24h wall limit -> resumes lose up to save_steps of work).
# MN5 time limit was 24:00:00; Leonardo max is 24h, so this run auto-resumes
# from the newest checkpoint in $OUT_DIR. Chain it with:
#   bash scripts/leonardo/submit_chain.sh scripts/leonardo/Qwen3.5-VL-9B-Instruct_v2_gl.sh 1
#SBATCH --job-name=Qwen3.5-VL-9B-Instruct_v2_gl
# NOTE: MN5 file says 64, but the queued MN5 job was submitted with -N 32
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time 24:00:00
#SBATCH --account=AIFAC_S07_127
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --exclusive

# --- Paths ---
# Env + base models still live in the ended project's work area (AIFAC_5C0_261);
# data, caches and checkpoints go to the active project's $WORK (AIFAC_S07_127).
export OLD_WORK=/leonardo_work/AIFAC_5C0_261
export WORK=/leonardo_work/AIFAC_S07_127
export DATA_DIR="$WORK/lukas/data_mn5"
export OUT_DIR="$WORK/lukas/multimodalModels/${SLURM_JOB_NAME}"

# --- Environment Activation ---
source $OLD_WORK/environments/env_torch_2_9_megratron2/bin/activate

# --- Directory Setup & Variables ---
export TMPDIR="$WORK/lukas/tmp"
export HF_HOME="$WORK/lukas/hf_cache"
export MODELSCOPE_CACHE="$WORK/lukas/modelscope_cache"
export TRITON_CACHE_DIR="$WORK/lukas/triton_cache"
export WANDB_DIR="$WORK/lukas/wandb_logs"
mkdir -p $TMPDIR $HF_HOME $MODELSCOPE_CACHE $TRITON_CACHE_DIR $WANDB_DIR $OUT_DIR

export WANDB_MODE=offline
export WANDB_PROJECT="qwen3-vl-finetuning"
export WANDB_NAME="${SLURM_JOB_NAME}_${SLURM_JOB_ID}"

export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
export LANGUAGE=en_US.UTF-8
export OMP_NUM_THREADS=4
export NNODES=$SLURM_NNODES
export GPUS_PER_NODE=4

# --- Network & Distributed Config (Leonardo Booster) ---
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NCCL_TIMEOUT=28800
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=28800
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_NET_DISABLE_INTRA=1
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3
export NCCL_IB_GID_INDEX=3
export NCCL_NET_GDR_LEVEL=2
export GLOO_SOCKET_IFNAME=ib0

# --- ms-swift / Qwen3.5-VL knobs ---
export SWIFT_PATCH_CONV3D=1
export SWIFT_USE_MCORE_GDN=1
export video_min_token_num=0
export video_max_token_num=0
export IMAGE_MAX_TOKEN_NUM=512

export MASTER_PORT=9327
export MAIN_PROCESS_IP=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)

# --- Auto-resume from the newest checkpoint of this job name ---
RESUME_ARGS=()
LAST_CKPT=$(ls -d $OUT_DIR/v*/checkpoint-* 2>/dev/null \
    | awk -F'checkpoint-' '{print $NF" "$0}' | sort -n | tail -n 1 | cut -d' ' -f2-)
if [ -n "$LAST_CKPT" ] && [ -f "$LAST_CKPT/latest_checkpointed_iteration.txt" ]; then
    echo "Resuming from $LAST_CKPT"
    RESUME_ARGS=(--mcore_model "$LAST_CKPT")
else
    echo "No checkpoint in $OUT_DIR, starting from scratch"
fi

echo "Launching $NNODES-node training on Head Node: $MAIN_PROCESS_IP at Port: $MASTER_PORT"
nvidia-smi topo -m

# --- Training Launch ---
srun accelerate launch \
    --num_processes $(( NNODES * GPUS_PER_NODE )) \
    --num_machines $NNODES \
    --mixed_precision bf16 \
    --dynamo_backend "no" \
    --rdzv_backend c10d \
    --main_process_ip $MAIN_PROCESS_IP \
    --main_process_port $MASTER_PORT \
    --machine_rank $SLURM_NODEID \
    $HOME/ms-swift/swift/cli/_megatron/sft.py \
    --model $OLD_WORK/baseModels/Qwen3.5-9B \
    --save_safetensors false \
    --cached_dataset $DATA_DIR/v2_gl/train \
    --load_from_cache_file true \
    --add_non_thinking_prefix true \
    --loss_scale ignore_empty_think \
    --split_dataset_ratio 0.01 \
    --tensor_model_parallel_size 4 \
    --torch_dtype bfloat16 \
    --micro_batch_size 4 \
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
    --output_dir $OUT_DIR \
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
    "${RESUME_ARGS[@]}"
