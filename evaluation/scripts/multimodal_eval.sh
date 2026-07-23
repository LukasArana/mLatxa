#!/bin/bash
#SBATCH --job-name=multimodal_eval
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=2-00:00:00
#SBATCH --mem=100GB
#SBATCH --gres=gpu:1
#SBATCH --array=0-12%4
#SBATCH --output=.slurm/multimodal_eval_%A_%a.out
#SBATCH --error=.slurm/multimodal_eval_%A_%a.err

#source /gscratch6/users/USER/vllmenv/bin/activate
source /gaueko1/users/USER/py312/bin/activate
export CUDA_HOME=/usr/local/cuda-12.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# ─── Models ───────────────────────────────────────────────────────────────────
# Each entry: "model_path|temp|top_p|top_k|min_p|presence_penalty|rep_penalty"
# OCR always uses greedy (temp=0); mmstar uses the per-model values below.
declare -a MODELS=(

    # ── Qwen3.5-9B family ──────────────────────────────────────────────────
    "/proiektuak/ilenia/models-instruct/Latxa-Qwen3.5-9B-v1|0.7|0.95|20|0.0|1.5|1.0"
    "/proiektuak/ilenia/models-instruct/Latxa-Qwen3.5-9B-v1_gal_multimodal_gal|0.7|0.8|20|0.0|1.5|1.0"
    "/proiektuak/ilenia/models-instruct/Latxa-Qwen3.5-9B-v1-multimodal_en|0.7|0.8|20|0.0|1.5|1.0"
    "/proiektuak/ilenia/models-instruct/Latxa-Qwen3.5-9B-v1_multimodal_en_4|0.7|0.8|20|0.0|1.5|1.0"
    "/proiektuak/ilenia/models-instruct/Latxa-Qwen3.5-9B-v1_multimodal_eu_en|0.7|0.8|20|0.0|1.5|1.0"
    "/proiektuak/ilenia/models-instruct/Latxa-Qwen3.5-9B-v2-2epoch|0.7|0.95|20|0.0|1.5|1.0"
    "/proiektuak/ilenia/models-instruct/Latxa-Qwen3.5-9B-v2-4epoch|0.7|0.8|20|0.0|1.5|1.0"
    "/proiektuak/ilenia/models-instruct/Latxa-Qwen3.5-9B-v2-multimodal_en|0.7|0.8|20|0.0|1.5|1.0"

    "Qwen/Qwen3.5-9B|0.7|0.8|20|0.0|1.5|1.0"

    # ── Qwen3/3.5-VL-32B/27B family ────────────────────────────────────────────────
    #"Qwen/Qwen3-VL-32B-Instruct|0.7|0.8|20|0.0|1.5|1.0"
    "Qwen/Qwen3.5-27B|0.7|0.8|20|0.0|1.5|1.0"
    "/proiektuak/ilenia/latxa-ereduak/Latxa-Qwen3.5-27B-v2-2epoch|0.7|0.8|20|0.0|1.5|1.0"

    # ── Qwen3-VL-8B family ─────────────────────────────────────────────────
    "/proiektuak/ilenia/models-instruct/multilingual-multimodal/8b.mono.eu|0.7|0.8|20|0.0|1.5|1.0"
    "Qwen/Qwen3-VL-8B-Instruct|0.7|0.8|20|0.0|1.5|1.0"
  #  "/proiektuak/ilenia-scratch/models-instruct/multilingual-multimodal/8b.mono.eu|0.7|0.8|20|0.0|1.5|1.0"
)

# ─── Tasks ────────────────────────────────────────────────────────────────────
# Available: mmstar_en  mmstar_eu  ocr_eu pixmo_eu pixmo_en
tasks_selected=(
    "animals_eu"
    "animals_en"
)
labels_path="/ikerlariak/USER/latxa_multimodal/animals_dataset/animals_name.txt"
image_path="/ikerlariak/USER/latxa_multimodal/animals_dataset/animals"

# ─── Evaluation loop ──────────────────────────────────────────────────────────

entry="${MODELS[$SLURM_ARRAY_TASK_ID]}"

IFS='|' read -r model_path temp top_p top_k min_p pres_pen rep_pen <<< "$entry"
model_short="$(basename "$model_path")"

echo "Running model ${SLURM_ARRAY_TASK_ID}: ${model_short}"

for task in "${tasks_selected[@]}"; do
    echo "=== Model: $model_path | Task: $task ==="

    if [[ $task == "mmstar_en" ]]; then
        python3 tasks/mmstar/mmstar_offline.py \
            --model_path "$model_path" \
            --max_new_tokens 10 \
            --temperature "$temp" \
            --top_p "$top_p" \
            --top_k "$top_k" \
            --min_p "$min_p" \
            --presence_penalty "$pres_pen" \
            --repetition_penalty "$rep_pen" \
            --dataset_name "HiTZ/mmstar_eu" \
            --output_file "results/${model_short}/mmstar_results_en.jsonl" \
            --batch_size 8 \
            --language en

    elif [[ $task == "mmstar_eu" ]]; then
        python3 tasks/mmstar/mmstar_offline.py \
            --model_path "$model_path" \
            --max_new_tokens 10 \
            --temperature "$temp" \
            --top_p "$top_p" \
            --top_k "$top_k" \
            --min_p "$min_p" \
            --presence_penalty "$pres_pen" \
            --repetition_penalty "$rep_pen" \
            --dataset_name "HiTZ/mmstar_eu" \
            --output_file "results/${model_short}/mmstar_results_eu.jsonl" \
            --batch_size 8 \
            --language eu

    elif [[ $task == "mmstar_eu_reviewed" ]]; then
        python3 tasks/mmstar/mmstar_offline.py \
            --model_path "$model_path" \
            --max_new_tokens 10 \
            --temperature "$temp" \
            --top_p "$top_p" \
            --top_k "$top_k" \
            --min_p "$min_p" \
            --presence_penalty "$pres_pen" \
            --repetition_penalty "$rep_pen" \
            --dataset_name "HiTZ/mmstar_eu" \
            --output_file "results/${model_short}/mmstar_results_eu_reviewed.jsonl" \
            --batch_size 8 \
            --language eu_reviewed

    elif [[ $task == "ocr_eu" ]]; then
        python3 tasks/ocr_eu/ocr_offline.py \
            --model_path "$model_path" \
            --max_new_tokens 256 \
            --temperature 0.0 \
            --dataset_name "HiTZ/ocr-eu" \
            --output_file "results/${model_short}/ocr_eu_results.jsonl" \
            --batch_size 8

    elif [[ $task == "pixmo_eu" ]]; then
        python3 tasks/pixmo_ama/pixmo_offline.py \
            --model_path "$model_path" \
            --max_new_tokens 2000 \
            --temperature "$temp" \
            --top_p "$top_p" \
            --top_k "$top_k" \
            --min_p "$min_p" \
            --presence_penalty "$pres_pen" \
            --repetition_penalty "$rep_pen" \
            --dataset_name "HiTZ/pixmo-ask-model-anything_eu" \
            --image_path "$image_path" \
            --output_file "results/${model_short}/pixmo_results_eu.jsonl" \
            --batch_size 8 \
            --language eu

    elif [[ $task == "pixmo_en" ]]; then
        python3 tasks/pixmo_ama/pixmo_offline.py \
            --model_path "$model_path" \
            --max_new_tokens 2000 \
            --temperature "$temp" \
            --top_p "$top_p" \
            --top_k "$top_k" \
            --min_p "$min_p" \
            --presence_penalty "$pres_pen" \
            --repetition_penalty "$rep_pen" \
            --dataset_name "HiTZ/pixmo-ask-model-anything_eu" \
            --image_path "$image_path" \
            --output_file "results/${model_short}/pixmo_results_en.jsonl" \
            --batch_size 8 \
            --language en

    elif [[ $task == "docci_eu" ]]; then
        python3 tasks/docci/docci_offline.py \
            --model_path "$model_path" \
            --max_new_tokens 2000 \
            --temperature "$temp" \
            --top_p "$top_p" \
            --top_k "$top_k" \
            --min_p "$min_p" \
            --presence_penalty "$pres_pen" \
            --repetition_penalty "$rep_pen" \
            --dataset_name "google/docci" \
            --output_file "results/${model_short}/docci_results_eu_paragraph.jsonl" \
            --batch_size 8 \
            --language eu

    elif [[ $task == "docci_en" ]]; then
        python3 tasks/docci/docci_offline.py \
            --model_path "$model_path" \
            --max_new_tokens 2000 \
            --temperature "$temp" \
            --top_p "$top_p" \
            --top_k "$top_k" \
            --min_p "$min_p" \
            --presence_penalty "$pres_pen" \
            --repetition_penalty "$rep_pen" \
            --dataset_name "google/docci" \
            --output_file "results/${model_short}/docci_results_en_paragraph.jsonl" \
            --batch_size 8 \
            --language en

    elif [[ $task == "animals_eu" ]]; then
        python3 tasks/grounding/animals_dataset_offline.py \
            --model_path "$model_path" \
            --max_new_tokens 2000 \
            --temperature "$temp" \
            --top_p "$top_p" \
            --top_k "$top_k" \
            --min_p "$min_p" \
            --presence_penalty "$pres_pen" \
            --repetition_penalty "$rep_pen" \
            --dataset_path "$image_path" \
            --labels_path "$labels_path" \
            --output_file "results/${model_short}/animals_results_eu.jsonl" \
            --batch_size 8 \
            --language eu

    elif [[ $task == "animals_en" ]]; then
        python3 tasks/grounding/animals_dataset_offline.py \
            --model_path "$model_path" \
            --max_new_tokens 2000 \
            --temperature "$temp" \
            --top_p "$top_p" \
            --top_k "$top_k" \
            --min_p "$min_p" \
            --presence_penalty "$pres_pen" \
            --repetition_penalty "$rep_pen" \
            --dataset_path "$image_path" \
            --labels_path "$labels_path" \
            --output_file "results/${model_short}/animals_results_en.jsonl" \
            --batch_size 8 \
            --language en
    fi

    echo "Task '$task' for '$model_path' complete!"
done

echo "Array task ${SLURM_ARRAY_TASK_ID} finished!"

echo "All evaluations complete!"
