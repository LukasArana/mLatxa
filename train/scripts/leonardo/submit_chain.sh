#!/bin/bash
# Submit a training script as N chained 24h jobs (Leonardo's wall-time limit).
# --dependency=singleton runs same-named jobs one after another; each one resumes
# from the newest checkpoint in its $OUT_DIR. Run from mLatxa/train/ so that
# logs/ lands in the usual place.
#
#   bash scripts/leonardo/submit_chain.sh scripts/leonardo/<script>.sh <N>
#
# A100s are slower than MN5's H100s, so N = MN5 days is a lower bound; add more
# chunks later the same way (they queue behind the running one). If training
# finishes early, scancel the leftover chunks.
set -euo pipefail
script=${1:?usage: submit_chain.sh <script.sh> <num_chunks>}
n=${2:?usage: submit_chain.sh <script.sh> <num_chunks>}
mkdir -p logs
for i in $(seq 1 "$n"); do
    sbatch --dependency=singleton "$script"
done
