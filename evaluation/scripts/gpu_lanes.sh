#!/bin/bash
# Stand-in for sbatch that keeps the GPU jobs it submits within MAX_GPUS GPUs at once.
# run_multimodal_models.sh puts it first on PATH (as "sbatch") for run_multimodal_nothink.sh.
#
# The budget is split into MAX_GPUS lanes. A job asking for g GPUs takes g lanes and gets
# --dependency=afterany on the last job in each of them (merged with any dependency it
# already had), then becomes the last job of those lanes. Each lane runs one job at a time,
# so the running jobs never hold more than MAX_GPUS GPUs. Lanes whose last job has left
# the queue count as free, so a later launch with the same state dir continues the chain.
#
# Env: REAL_SBATCH (the real binary), GPU_LANES_STATE (state dir), MAX_GPUS (default 4).
# Inside a SLURM job, or without GPU_LANES_STATE, it is plain sbatch.
set -euo pipefail
real="${REAL_SBATCH:?REAL_SBATCH not set}"
[[ -n "${SLURM_JOB_ID:-}" || -z "${GPU_LANES_STATE:-}" ]] && exec "$real" "$@"
max="${MAX_GPUS:-4}"

orig=("$@") args=() g=0 dep=""
while (($#)); do
  case "$1" in
    --gres=gpu*) g=${1##*:}; args+=("$1") ;;   # --gres=gpu:N or --gres=gpu:<type>:N
    --dependency=*) dep=${1#--dependency=} ;;
    --dependency|-d) dep=$2; shift ;;
    *) args+=("$1") ;;
  esac
  shift
done
[[ "$g" =~ ^[0-9]+$ ]] || g=1
(( g == 0 )) && exec "$real" "${orig[@]}"
(( g <= max )) || { echo "gpu_lanes: job asks for $g GPUs, budget is $max" >&2; exit 1; }

mkdir -p "$GPU_LANES_STATE"
exec 9>"$GPU_LANES_STATE/lock"
flock 9

# Each lane: "<last job id> <lane>"; free lanes (empty or job gone from the queue) sort first,
# then the lanes whose last job was submitted earliest.
cands=()
for ((i = 0; i < max; i++)); do
  id=$(cat "$GPU_LANES_STATE/lane_$i" 2>/dev/null || true)
  if [[ -z "$id" ]] || ! squeue -h -j "$id" -o %i 2>/dev/null | grep -q .; then id=0; fi
  cands+=("$id $i")
done
mapfile -t chosen < <(printf '%s\n' "${cands[@]}" | sort -n -k1,1 | head -n "$g")

wait_on=""
for c in "${chosen[@]}"; do
  id=${c% *}
  (( id > 0 )) && wait_on="$wait_on:$id"
done
if [[ -n "$wait_on" ]]; then dep="${dep:+$dep,}afterany$wait_on"; fi

out=$("$real" "${args[@]}" ${dep:+--dependency="$dep"})
jid=${out%%;*}
for c in "${chosen[@]}"; do echo "$jid" > "$GPU_LANES_STATE/lane_${c#* }"; done
echo "gpu_lanes: job $jid ($g GPUs) lanes [$(printf '%s ' "${chosen[@]#* }")] after '${wait_on#:}'" >&2
echo "$out"
