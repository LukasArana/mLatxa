#!/bin/bash
#SBATCH --job-name=submit_when_ready
#SBATCH --partition=lrd_all_serial
#SBATCH --qos=normal
#SBATCH --account=AIFAC_S07_127
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=04:00:00
#SBATCH --output=submit_when_ready_%j.out
#SBATCH --error=submit_when_ready_%j.err

# Submits the MN5-ported training chains as soon as (a) boost_usr_prod has nodes
# again (during maintenance it has none and sbatch rejects every booster job with
# "More processors requested than permitted") and (b) each run's datasets are on
# Leonardo (state.json present; multimodal caches also need the .leonardo_paths
# marker from fix_image_paths.py). Polls every 5 min for up to ~3h50m.
#
# Submit from mLatxa/train (logs/ of the training jobs land there):
#   cd ~/mLatxa/train && sbatch scripts/leonardo/submit_when_ready.sbatch
# Runs already submitted are recorded in $STATE and never submitted twice, so it
# is safe to resubmit this job if it times out before everything went in.
# Edit RUNS to choose which runs (and how many 24h pieces) to submit.

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/mLatxa/train}"
L=scripts/leonardo
DATA_DIR=/leonardo_work/AIFAC_S07_127/lukas/data_mn5
MULTIMODAL="multimodal_v1 v1_multimodal_eu_en multimodal_124 multimodal_eu_124"
STATE="$HOME/.submit_when_ready.submitted"
touch "$STATE"

# script : number of 24h pieces
RUNS=(
  Qwen3.5-VL-27B-Instruct_v2_ca:3
  Qwen3.5-VL-27B-Instruct_v2_gl:3
  Qwen3.5-VL-27B-Instruct_v2_no:3
  Qwen3.5-VL-27B-Instruct_v2:2
  Qwen3.5-VL-9B-Instruct_v2:2
  Qwen3.5-VL-9B-Instruct_v2_ca:2
  Qwen3.5-VL-9B-Instruct_v2_gl:2
  Qwen3.5-VL-9B-Instruct_v2_no:2
  Qwen3.5-VL-9B-Instruct_v1_multimodal:2
  Qwen3.5-VL-9B-Instruct_v1_multimodal_eu_en:2
  Qwen3.5-VL-9B-Instruct_v2_multimodal_en_124:2
  Qwen3.5-VL-9B-Instruct_v2_multimodal_eu_124:2
  Qwen3.5-VL-9B-Instruct_v2_multimodal_eu_en_124:2
)

log() { echo "[$(date '+%F %T')] $*"; }

data_ready() {  # data_ready <script.sh>: all cached_dataset dirs present (+ rewritten if multimodal)
  local d name
  for d in $(grep -- '--cached_dataset' "$1" | sed 's/.*--cached_dataset//; s/\\$//'); do
    d=${d//\$DATA_DIR/$DATA_DIR}
    [ -f "$d/state.json" ] || return 1
    name=$(basename "$(dirname "$d")")
    if [[ " $MULTIMODAL " == *" $name "* ]] && [ ! -f "$d/.leonardo_paths" ]; then return 1; fi
  done
}

deadline=$(( $(date +%s) + 230*60 ))
while :; do
  nodes=$(sinfo -h -p boost_usr_prod -o %D 2>/dev/null | awk '{s+=$1} END {print s+0}')
  pending=0
  if [ "$nodes" -gt 0 ]; then
    for r in "${RUNS[@]}"; do
      s=${r%%:*}; n=${r##*:}
      grep -qx "$s" "$STATE" && continue
      if data_ready "$L/$s.sh"; then
        log "submitting $s ($n pieces)"
        if bash "$L/submit_chain.sh" "$L/$s.sh" "$n"; then
          echo "$s" >> "$STATE"
        else
          log "sbatch failed for $s, will retry"; pending=1
        fi
      else
        pending=1
      fi
    done
  else
    log "boost_usr_prod has no nodes yet (maintenance)"; pending=1
  fi
  [ "$pending" -eq 0 ] && { log "all runs submitted"; break; }
  [ "$(date +%s)" -ge "$deadline" ] && { log "time limit: not submitted yet: $(for r in "${RUNS[@]}"; do s=${r%%:*}; grep -qx "$s" "$STATE" || echo -n "$s "; done) — resubmit this job"; break; }
  sleep 300
done
