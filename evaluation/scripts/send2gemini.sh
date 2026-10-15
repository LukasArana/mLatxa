#!/usr/bin/env bash
# send2gemini.sh
#
# Full judge pipeline: prepare → submit → poll → download
#
# Usage:
#   VENV=/path/to/venv bash evaluation/scripts/send2gemini.sh [PREDICTIONS_FILE]
#
# The API key is read from $GEMINI_API_KEY or, if unset, from evaluation/GEMINI_API_KEY.txt (gitignored).
#
# Adjust the variables in the CONFIG section before running.


source "${VENV:?set VENV to the evaluation virtualenv}/bin/activate"
set -euo pipefail

EVAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$EVAL_DIR"
if [[ -z "${GEMINI_API_KEY:-}" ]]; then
    GEMINI_API_KEY="$(cat "$EVAL_DIR/GEMINI_API_KEY.txt")"
fi
[[ -n "$GEMINI_API_KEY" ]] || { echo "GEMINI_API_KEY is empty" >&2; exit 1; }
export GEMINI_API_KEY

# ─── CONFIG ────────────────────────────────────────────────────────────────────

PREDICTIONS_FILE="${1:-results/Latxa-Qwen3.5-9B-v2-multimodal_1024/docci_results_eu2.jsonl}"     # your input JSONL
MODEL_NAME="gemini-3.5-flash"                              # judge model
HF_SPLIT="test"                                 # HuggingFace dataset split
HF_DATASET="google/docci"                   # HuggingFace dataset name


# Final output
model_dir="$(basename "$(dirname "$PREDICTIONS_FILE")")"
pred_name="$(basename "$PREDICTIONS_FILE" .jsonl)"
OUT_FILE="results/${model_dir}_${pred_name}_judge_gemini.jsonl"

POLL_INTERVAL=30   # seconds between status checks

# ───────────────────────────────────────────────────────────────────────────────


python judge/judge_gemini.py \
  --predictions_file "$PREDICTIONS_FILE" \
  --output_file "$OUT_FILE" \
  --model_name "$MODEL_NAME" \
  --sleep 0.5

