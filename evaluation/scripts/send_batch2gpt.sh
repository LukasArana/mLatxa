#!/usr/bin/env bash
# send_batch2gpt.sh
#
# Full judge pipeline: prepare → submit → poll → download
#
# Usage:
#   VENV=/path/to/venv bash evaluation/scripts/send_batch2gpt.sh [PREDICTIONS_FILE]
#
# The API key is read from $OPENAI_API_KEY or, if unset, from evaluation/OPENAI_API_KEY.txt (gitignored).
#
# Adjust the variables in the CONFIG section before running.


source "${VENV:?set VENV to the evaluation virtualenv}/bin/activate"
set -euo pipefail

EVAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$EVAL_DIR"
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    OPENAI_API_KEY="$(cat "$EVAL_DIR/OPENAI_API_KEY.txt")"
fi
[[ -n "$OPENAI_API_KEY" ]] || { echo "OPENAI_API_KEY is empty" >&2; exit 1; }
export OPENAI_API_KEY

# ─── CONFIG ────────────────────────────────────────────────────────────────────

PREDICTIONS_FILE="${1:-results/Latxa-Qwen3.5-9B-v2-multimodal_1024/docci_results_eu2.jsonl}"     # your input JSONL
MODEL_NAME="gpt-5.5"                              # judge model
HF_SPLIT="test"                                 # HuggingFace dataset split
HF_DATASET="google/docci"                   # HuggingFace dataset name

# Scratch paths (batch input + ID storage)
BATCH_DIR="tmp/judge_batch.jsonl"
BATCH_ID_FILE="tmp/judge_batch_id.txt"

# Final output
model_dir="$(basename "$(dirname "$PREDICTIONS_FILE")")"
pred_name="$(basename "$PREDICTIONS_FILE" .jsonl)"
OUT_FILE="results/${model_dir}_${pred_name}_judge_gpt.jsonl"

POLL_INTERVAL=30   # seconds between status checks

# ───────────────────────────────────────────────────────────────────────────────

echo "=== Step 1: Prepare batch ==="

#python judge/prepare_batch.py \
 #   --predictions_file "$PREDICTIONS_FILE" \
 #   --batch_file "$BATCH_DIR" \
 #   --model_name "$MODEL_NAME" \
#   --hf_split "$HF_SPLIT" \
 #   --hf_dataset "$HF_DATASET" \
 #   --ground_in_eus  # Include this flag to use 'ground_truth' from predictions as-is in Basque

python judge/prepare_batch_wo_imgs.py \
  --predictions_file "$PREDICTIONS_FILE" \
  --batch_file "$BATCH_DIR" \
  --model_name "$MODEL_NAME"

echo "=== Step 2: Submit batch ==="
python judge/submit_batch.py \
    --batch_file "$BATCH_DIR" \
    --batch_id_file "$BATCH_ID_FILE"

echo "=== Step 3: Poll until complete ==="
while true; do
    STATUS=$(python judge/check_batch.py \
        --batch_id_file "$BATCH_ID_FILE" \
        | grep "Status:" | awk '{print $2}')

    if [[ "$STATUS" == "completed" || "$STATUS" == "failed" \
       || "$STATUS" == "cancelled" || "$STATUS" == "expired" ]]; then
        echo "Batch finished with status: $STATUS"
        break
    fi

    echo "Batch status: $STATUS. Waiting ${POLL_INTERVAL}s..."
    sleep "$POLL_INTERVAL"
done

echo "=== Step 4: Download and parse results ==="
python judge/download_batch.py \
    --batch_id_file "$BATCH_ID_FILE" \
    --predictions_file "$PREDICTIONS_FILE" \
    --out_file "$OUT_FILE"

echo ""
echo "Done! Results saved to: $OUT_FILE"

