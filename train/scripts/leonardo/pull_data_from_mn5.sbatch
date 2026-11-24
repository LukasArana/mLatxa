#!/bin/bash
#SBATCH --job-name=mn5_pull_data
#SBATCH --partition=lrd_all_serial   # data transfer, no GPU
#SBATCH --qos=normal
#SBATCH --account=AIFAC_S07_127
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G               # image-path rewrite (fix_image_paths.py) runs here too
#SBATCH --time=04:00:00              # lrd_all_serial max walltime
#SBATCH --output=mn5_pull_data_%j.out
#SBATCH --error=mn5_pull_data_%j.err

# Pull the cached datasets used by the scripts/leonardo/ MN5 ports from MN5 (transfer1.bsc.es)
# into $DATA_DIR on Leonardo. Same pattern as ~/submit_transfer_jsonl.sbatch,
# in the opposite direction: uses ~/.ssh/mn5_transfer.
#
# Submit ONCE from a Leonardo login node, from the directory holding this file:
#   MN5_USER=<mn5 user> sbatch pull_data_from_mn5.sbatch
# (sbatch exports the environment, so MN5_USER reaches the job and its chained successors.)
#
# SELF-CHAINING: each job queues the next one (afterany) before working, so if
# the 4h wall kills it, the next job resumes rsync. The chain stops once every
# dataset is verified (done-flag file) or after CHAIN_MAX jobs.
# Stop manually: scancel the pending mn5_pull_data job, or touch $DONE_FLAG.
#
# Fallback if this fails: copy_data_from_mn5.sh (relays through your laptop).

set -uo pipefail

SSH="ssh -i $HOME/.ssh/mn5_transfer -o StrictHostKeyChecking=accept-new -o BatchMode=yes"
SRC="${MN5_USER:?export MN5_USER (MN5 username) before sbatch}@transfer1.bsc.es:/gpfs/scratch/ehu26/data"   # = ~/scratch/data on MN5
DATA_DIR=/leonardo_work/AIFAC_S07_127/lukas/data_mn5         # = $DATA_DIR in the sbatch scripts

# sizes on MN5: v2 23G, v2_ca 15G, v2_gl 15G, v2_no 26G, multimodal_v1 12G,
#               v1_multimodal_eu_en 13G, multimodal_124 1.3G, multimodal_eu_124 1.2G
DATASETS=(v2 v2_ca v2_gl v2_no multimodal_v1 v1_multimodal_eu_en multimodal_124 multimodal_eu_124)
# Multimodal caches store absolute MN5 image paths; they are rewritten to the
# Leonardo copy of the images by fix_image_paths.py once copied and verified.
MULTIMODAL=(multimodal_v1 v1_multimodal_eu_en multimodal_124 multimodal_eu_124)
FIX_SCRIPT="${SLURM_SUBMIT_DIR:-.}/fix_image_paths.py"
PY_ENV=/leonardo_work/AIFAC_5C0_261/environments/env_torch_2_9_megratron2

DONE_FLAG="$HOME/.mn5_pull_data.done"
CHAIN_MAX=8
CHAIN_N=${CHAIN_N:-0}
SELF="$(realpath "${SLURM_SUBMIT_DIR:-.}/pull_data_from_mn5.sbatch")"

log() { echo "[$(date '+%F %T')] $*"; }

if [ -f "$DONE_FLAG" ]; then
  log "Done flag present ($DONE_FLAG) — nothing to do."
  exit 0
fi
if [ "$CHAIN_N" -ge "$CHAIN_MAX" ]; then
  log "Chain limit reached ($CHAIN_MAX jobs) — NOT resubmitting. Investigate and resubmit manually."
else
  sbatch --dependency=afterany:"$SLURM_JOB_ID" --export=ALL,CHAIN_N=$((CHAIN_N+1)) "$SELF"
  log "Queued chained job #$((CHAIN_N+1))."
fi

FAILED=0
for d in "${DATASETS[@]}"; do
  log "=== $d ==="
  if [ -f "$DATA_DIR/$d/train/.leonardo_paths" ]; then
    log "$d: already copied and image paths rewritten — skipping (re-syncing would undo the rewrite)"
    continue
  fi
  mkdir -p "$DATA_DIR/$d/train"
  if ! rsync -a --partial --info=progress2 -e "$SSH" "$SRC/$d/train/" "$DATA_DIR/$d/train/"; then
    log "rsync failed for $d"; FAILED=1; continue
  fi
  # Verify with checksums: nothing may be left to transfer.
  pending=$(rsync -a --checksum --dry-run --itemize-changes -e "$SSH" \
      "$SRC/$d/train/" "$DATA_DIR/$d/train/" | grep -c '^[<>c]')
  if [ "$pending" != "0" ]; then
    log "$d: $pending files still differ after copy"; FAILED=1
  else
    log "$d: verified"
  fi
done

if [ "$FAILED" -eq 0 ]; then
  log "=== Rewriting MN5 image paths in multimodal caches ==="
  source "$PY_ENV/bin/activate"
  export HF_DATASETS_CACHE="$DATA_DIR/.hf_datasets_cache" TMPDIR="$DATA_DIR/.tmp"
  mkdir -p "$TMPDIR"
  MM_PATHS=()
  for d in "${MULTIMODAL[@]}"; do MM_PATHS+=("$DATA_DIR/$d/train"); done
  python "$FIX_SCRIPT" "${MM_PATHS[@]}" || FAILED=1
fi

if [ "$FAILED" -eq 0 ]; then
  touch "$DONE_FLAG"
  log "All datasets verified — done flag written. The queued chained job will exit immediately."
else
  log "Transfer NOT complete — the queued chained job will resume it."
fi
