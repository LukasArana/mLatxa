#!/usr/bin/env bash
# run_judge.sh
#
# Full judge pipeline: prepare → submit → poll → download
#
# Usage:
#   bash run_judge.sh
#
# Adjust the variables in the CONFIG section before running.


source /gscratch6/users/USER/vllmenv/bin/activate
export GEMINI_API_KEY="$(cat /ikerlariak/USER/latxa_multimodal/mLatxa/evaluation/GEMINI_API_KEY.txt)"
set -euo pipefail

# ─── CONFIG ────────────────────────────────────────────────────────────────────

PREDICTIONS_FILE="/ikerlariak/USER/latxa_multimodal/mLatxa/evaluation/results/Latxa-Qwen3.5-9B-v2-multimodal_1024/docci_results_eu2.jsonl"     # your input JSONL
MODEL_NAME="gemini-3.5-flash"                              # judge model
HF_SPLIT="test"                                 # HuggingFace dataset split
HF_DATASET="google/docci"                   # HuggingFace dataset name


# Final output
model_dir="$(basename "$(dirname "$PREDICTIONS_FILE")")"
pred_name="$(basename "$PREDICTIONS_FILE" .jsonl)"
OUT_FILE="/ikerlariak/USER/latxa_multimodal/mLatxa/evaluation/results/${model_dir}_${pred_name}_judge_output2.jsonl"

POLL_INTERVAL=30   # seconds between status checks

# ───────────────────────────────────────────────────────────────────────────────


python judge/judge_gemini.py \
  --predictions_file "$PREDICTIONS_FILE" \
  --output_file "$OUT_FILE" \
  --model_name "$MODEL_NAME" \
  --sleep 0.5

