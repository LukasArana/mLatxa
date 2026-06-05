#!/bin/bash
#SBATCH --job-name=multimodal_eval
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mem=50GB
#SBATCH --gres=gpu:1
#SBATCH --output=.slurm/multimodal_eval.out
#SBATCH --error=.slurm/multimodal_eval.err

source /gscratch6/users/asagasti036/vllmenv/bin/activate

# ─── Models ───────────────────────────────────────────────────────────────────
# Each entry: "model_path|temp|top_p|top_k|min_p|presence_penalty|rep_penalty"
# OCR always uses greedy (temp=0); mmstar uses the per-model values below.
declare -a MODELS=(

    # ── Qwen3.5-9B family ──────────────────────────────────────────────────
    "/proiektuak/ilenia-scratch/models-instruct/Latxa-Qwen3.5-9B-v2-multimodal_1024|0.7|0.8|20|0.0|1.5|1.0"
    "/proiektuak/ilenia-scratch/models-instruct/Latxa-Qwen3.5-9B-v2|0.7|0.8|20|0.0|1.5|1.0"
    "/proiektuak/ilenia-scratch/models-instruct/Latxa-Qwen3.5-2B-v2/|0.7|0.8|20|0.0|1.5|1.0"
    #"/proiektuak/ilenia-scratch/models-instruct/Latxa-Qwen3.5-9B-v1|0.7|0.95|20|0.0|1.5|1.0"
    #"/proiektuak/ilenia-scratch/models-instruct/Latxa-Qwen3.5-9B-v2-2epoch|0.7|0.95|20|0.0|1.5|1.0"

    # ── Qwen3-VL-32B family ────────────────────────────────────────────────
    #"Qwen/Qwen3-VL-32B-Instruct|0.7|0.8|20|0.0|1.5|1.0"
   # "/proiektuak/ilenia-scratch/models-instruct/Latxa-Qwen3-32B-v2|0.7|0.8|20|0.0|1.5|1.0"

    # ── Qwen3-VL-8B family ─────────────────────────────────────────────────
  #  "Qwen/Qwen3-VL-8B-Instruct|0.7|0.8|20|0.0|1.5|1.0"
  #  "/proiektuak/ilenia-scratch/models-instruct/multilingual-multimodal/8b.mono.eu|0.7|0.8|20|0.0|1.5|1.0"
)

# ─── Tasks ────────────────────────────────────────────────────────────────────
# Available: mmstar_en  mmstar_eu  ocr_eu
tasks_selected=(
    "mmstar_en"
    "mmstar_eu"
    "ocr_eu"
    "mmstar_eu_reviewed"
)

# ─── Evaluation loop ──────────────────────────────────────────────────────────
for entry in "${MODELS[@]}"; do
    IFS='|' read -r model_path temp top_p top_k min_p pres_pen rep_pen <<< "$entry"
    model_short="$(basename "$model_path")"

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
        fi

        echo "Task '$task' for '$model_path' complete!"
    done
done

echo "All evaluations complete!"
