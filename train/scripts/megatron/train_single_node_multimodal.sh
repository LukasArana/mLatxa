
#!/bin/bash
#SBATCH --job-name=swift-single-node
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --time 24:00:00
#SBATCH --account=AIFAC_5C0_261
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
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
export NNODES=1
export GPUS_PER_NODE=4

# --- Network & Distributed Config (Leonardo Specifics) ---
export PYTORCH_ALLOC_CONF="expandable_segments:True"

export SWIFT_PATCH_CONV3D=1


MAX_PIXELS=1003520

nvidia-smi topo -m



# --- Training Launch ---
echo "Launching $NNODES-node training on Head Node: $MAIN_PROCESS_IP at Port: $MASTER_PORT"

# For more information on multi-node training launch methods, refer to:
# https://github.com/modelscope/ms-swift/tree/main/examples/train/multi-node

MASTER_PORT=9327
MAIN_PROCESS_IP=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export ACCELERATE_FSDP_SHARDING_STRATEGY="1"

SWIFT_USE_MCORE_GDN=1

megatron sft \
    --model /leonardo_work/AIFAC_5C0_261/baseModels/Qwen3.5-0.8B \
    --save_safetensors true \
    --cached_dataset /leonardo_work/AIFAC_5C0_261/datasets/train/preprocessed/multimodal_debug/train \
    --load_from_cache_file true \
    --add_non_thinking_prefix true \
    --loss_scale ignore_empty_think \
    --split_dataset_ratio 0.01 \
    --tensor_model_parallel_size 1 \
    --pipeline_model_parallel_size 1 \
    --micro_batch_size 1 \
    --global_batch_size 4 \
    --packing true \
    --padding_free \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --num_train_epochs 1 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-5 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-6 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --adam_eps 1e-8 \
    --lr_decay_style cosine \
    --output_dir /leonardo_work/AIFAC_5C0_261/multimodalModels \
    --save_steps 250 \
    --max_length 8192 \
    --dataloader_num_workers 1 \
    --dataset_num_proc 1 \
    --sequence_parallel true \
    --attention_backend flash \
    --no_load_optim false \
    --no_load_rng false \
    --save_total_limit 2 \
    --overlap_param_gather true \
    --overlap_grad_reduce true \
    --logging_steps 5 \
    --no_save_optim false \
    --dist_ckpt_save_pre_mcore_014 true \
    --freeze_llm true \
    --freeze_vit true \
    --freeze_aligner false

