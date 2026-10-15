#!/bin/bash
#SBATCH --job-name=qwen3.5_eval
#SBATCH --array=0-26%4             # 9 models x 3 task groups (task id = model_idx * 3 + group)
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/slurm_%A_%a.out

# Submit with: sbatch multimodal_harness.sh  (keep --array in sync with len(MODEL) * 3)
#
# pip3 install ".[qwen]" (for Qwen's dependencies)
# pip3 install transformers>=5.3.0 (to get Qwen3_5 models)
# pip3 install flash-linear-attention (optional, make inference faster)

source eval_env/bin/activate

MODEL=(
    'Qwen/Qwen3.5-2B'
    'Latxa-Qwen3.5-9B-v1-multimodal'
    'Latxa-Qwen3.5-9B-v1-multimodal_eu_en'
    'Latxa-Qwen3.5-9B-v2'
    'Latxa-Qwen3.5-9B-v2-2epoch'
    'Latxa-Qwen3.5-9B-v2-multimodal_1024'
    'Latxa-Qwen3.5-9B-v1'
    'Latxa-Qwen3.5-2B-multi'
    'Latxa-Qwen3-32B-v2'
)


MAX_PIXELS=$((2048 * 32 * 32))
MIN_PIXELS=$((256 * 32 * 32))
MAX_NUM_FRAMES=1
BATCH_SIZE=8
PORT=$((12346 + SLURM_ARRAY_TASK_ID))

GROUP=$((SLURM_ARRAY_TASK_ID % 3))
MODEL_NAME="${MODEL[$((SLURM_ARRAY_TASK_ID / 3))]}"
echo "Model: $MODEL_NAME | group: $GROUP"

# Group 0: text-only (knowledge, math, instruction following)
if [ "$GROUP" -eq 0 ]; then
    accelerate launch --num_processes=1 --main_process_port=$PORT -m lmms_eval \
        --model qwen3_5 \
        --model_args pretrained=$MODEL_NAME,max_pixels=$MAX_PIXELS,min_pixels=$MIN_PIXELS,max_frames=$MAX_NUM_FRAMES,enable_thinking=false \
        --gen_kwargs max_new_tokens=8192,temperature=0.7,top_p=0.8,top_k=20,repetition_penalty=1.0 \
        --tasks mmlu_pro,aime_figures,ifeval \
        --log_samples \
        --output_path ./logs/ \
        --batch_size $BATCH_SIZE
fi

# Group 1: image reasoning (visual math, multimodal understanding)
if [ "$GROUP" -eq 1 ]; then
    accelerate launch --num_processes=1 --main_process_port=$PORT -m lmms_eval \
        --model qwen3_5 \
        --model_args pretrained=$MODEL_NAME,max_pixels=$MAX_PIXELS,min_pixels=$MIN_PIXELS,max_frames=$MAX_NUM_FRAMES,enable_thinking=false \
        --gen_kwargs max_new_tokens=8192,temperature=0.7,top_p=0.8,top_k=20,repetition_penalty=1.0 \
        --tasks mmmu_val,mmmu_pro,mmstar \
        --log_samples \
        --output_path ./logs/ \
        --batch_size $BATCH_SIZE
fi

# Group 2: OCR & document parsing
if [ "$GROUP" -eq 2 ]; then
    accelerate launch --num_processes=1 --main_process_port=$PORT -m lmms_eval \
        --model qwen3_5 \
        --model_args pretrained=$MODEL_NAME,max_pixels=$MAX_PIXELS,min_pixels=$MIN_PIXELS,max_frames=$MAX_NUM_FRAMES,enable_thinking=false \
        --gen_kwargs max_new_tokens=2048,temperature=0.7,top_p=0.8,top_k=20,repetition_penalty=1.0 \
        --tasks ocrbench,omnidocbench \
        --log_samples \
        --output_path ./logs/ \
        --batch_size $BATCH_SIZE
fi