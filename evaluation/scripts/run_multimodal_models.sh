#!/bin/bash
# Evaluate a list of multimodal models, each in its own set of SLURM jobs, by calling
# LatxaTxat's run_multimodal_nothink.sh once per model (no thinking). Per model that submits:
#   vision  MMStar, A-OKVQA, VQAv2 (eu + en)                         multimodal
#   core    basque_v2 core tasks, first 1,000 items per task          text only
#   extra   NoRobots, MMLU-Pro open, EusProficiency, SimpleQA, DocMT  text only
# plus that model's own text-grading (2 GPUs) and VQAv2-grading (1 GPU) judge jobs,
# which start when its generation jobs end.
#
#   bash run_multimodal_models.sh                # the MODELS list below
#   bash run_multimodal_models.sh models.txt     # same format, one model per line
#   DRY_RUN=1 bash run_multimodal_models.sh      # print the commands, submit nothing
#
# A model line is:  <path or HF id> [gpus|auto] [label...]
# gpus = GPUs per generation job; "auto" (or missing) picks it from the size in the name
# (<=9B: 1, <=35B: 2, larger: 4). The label names the model on the results page.
# SKIP_SAFETY=0 also runs the safety/bias tasks; LOGS_ROOT changes where logs go.
#
# GPU budget: at most MAX_GPUS (default 4) GPUs in use at once by these jobs, judges included.
# Every submission goes through gpu_lanes.sh, which chains the jobs with afterany
# dependencies; everything is queued at once and SLURM starts each job when its lanes free.
# The state (lanes' last jobs) is in $LOGS_ROOT/.gpu_lanes, so a second launch queues
# behind the first instead of adding GPUs. MAX_GPUS=0 turns the limit off.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RUN=/ikerlariak/jetxaniz007/LatxaTxat/evaluation/scripts/run_multimodal_nothink.sh
P=/proiektuak/ilenia/models-instruct
LOGS_ROOT="${LOGS_ROOT:-/sorgin1/users/larana/mLatxa/evaluation/scripts/multimodal_eval_logs}"
export MAX_GPUS="${MAX_GPUS:-4}"

if [[ "${DRY_RUN:-0}" != 1 && "$MAX_GPUS" != 0 ]]; then
  REAL_SBATCH=$(command -v sbatch) || { echo "sbatch not found (submit from durunda)"; exit 1; }
  export REAL_SBATCH GPU_LANES_STATE="$LOGS_ROOT/.gpu_lanes"
  mkdir -p "$GPU_LANES_STATE/bin"
  ln -sfn "$HERE/gpu_lanes.sh" "$GPU_LANES_STATE/bin/sbatch"
  export PATH="$GPU_LANES_STATE/bin:$PATH"
fi

MODELS=(
  "Qwen/Qwen3.5-9B                                   auto  Qwen3.5-9B (base)"
  "$P/Latxa-Qwen3.5-9B-v2_multimodal_eu_en           auto  Latxa v2 + multimodal eu/en"
  "$P/Latxa-Qwen3.5-9B-v2_multimodal_eu              auto  Latxa v2 + multimodal eu"
  "$P/Latxa-Qwen3.5-9B-v2-multimodal_en              auto  Latxa v2 + multimodal en"
  "$P/Latxa-Qwen3.5-9B-v2_multimodal_4               auto  Latxa v2 + multimodal_4"
  "$P/Latxa-Qwen3.5-9B-v1_multimodal_eu_en           auto  Latxa v1 + multimodal eu/en"
  "$P/Latxa-Qwen3.5-9B-v1_multimodal_eu              auto  Latxa v1 + multimodal eu"
  "$P/Latxa-Qwen3.5-9B-v1-multimodal_en              auto  Latxa v1 + multimodal en"
  "$P/Latxa-Qwen3.5-9B-v1_multimodal_eu_4            auto  Latxa v1 + multimodal eu_4"
  "$P/Latxa-Qwen3.5-9B-v1_multimodal_4_new           auto  Latxa v1 + multimodal_4 (new)"
  "$P/Latxa-Qwen3.5-2B-multi                         auto  Latxa 2B multimodal"
  "$P/Latxa-Qwen3.5-27B-v2_multimodal_eu/checkpoint-10000-hf  auto  Latxa 27B v2 + multimodal eu"
)
if [[ $# -ge 1 ]]; then
  mapfile -t MODELS < <(grep -vE '^\s*(#|$)' "$1")
fi

auto_gpus() {  # largest "<n>B" in the name -> GPUs per generation job
  local b
  b=$(grep -oiE '[0-9]+(\.[0-9]+)?b' <<< "$1" | tr -d 'bB' | sort -g | tail -1)
  b=${b%.*}
  local g
  if [[ -z "$b" ]]; then g=1
  elif (( b <= 9 )); then g=1
  elif (( b <= 35 )); then g=2
  else g=4
  fi
  (( MAX_GPUS > 0 && g > MAX_GPUS )) && g=$MAX_GPUS
  echo $g
}

for line in "${MODELS[@]}"; do
  read -r model gpus label <<< "$line"
  [[ -z "${gpus:-}" || "$gpus" == auto ]] && gpus=$(auto_gpus "$model")
  label="${label:-$(basename "$model")}"
  logs="$LOGS_ROOT/$(sed "s#^$P/##; s#/#__#g" <<< "$model")"
  echo "== $model  gpus=$gpus  label='$label'  logs=$logs"
  if [[ "${DRY_RUN:-0}" == 1 ]]; then
    echo "   GPUS=$gpus LABEL='$label' LOGS=$logs bash $RUN $model"
    continue
  fi
  # One call per model: its jobs and judges depend only on its own generation jobs.
  GPUS=$gpus LABEL="$label" LOGS="$logs" bash "$RUN" "$model" \
    || echo "!! submission failed for $model, continuing" >&2
done
