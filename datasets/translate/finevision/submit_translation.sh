#!/bin/bash
# Auto-submit the EN->Basque shard translation as a SLURM array job.
#
#   ./submit_translation.sh INPUT OUTPUT_DIR [SAMPLES_PER_JOB] [MAX_CONCURRENT]
#
# It counts the samples in INPUT, works out how many array tasks (= nodes) are
# needed, and submits one array job. Each task translates its own slice based
# on $SLURM_ARRAY_TASK_ID, so no manual split list is required.
set -euo pipefail

INPUT="${1:?usage: ./submit_translation.sh INPUT OUTPUT_DIR [SAMPLES_PER_JOB] [MAX_CONCURRENT]}"
OUTPUT_DIR="${2:?missing OUTPUT_DIR}"
SAMPLES_PER_JOB="${3:-20000}"
MAX_CONCURRENT="${4:-8}"   # cap on tasks running at once (be nice to the queue)

# --- count total samples, auto-detecting JSON array vs JSONL ------------------
FIRST=$(grep -m1 -o '[^[:space:]]' "${INPUT}" | head -c1 || true)
if [ "${FIRST}" = "[" ]; then
    TOTAL=$(python -c "import json,sys; print(len(json.load(open(sys.argv[1]))))" "${INPUT}")
else
    TOTAL=$(grep -cve '^[[:space:]]*$' "${INPUT}")
fi

if [ "${TOTAL}" -eq 0 ]; then
    echo "Input ${INPUT} has 0 samples; nothing to submit." >&2
    exit 1
fi

NJOBS=$(( (TOTAL + SAMPLES_PER_JOB - 1) / SAMPLES_PER_JOB ))
LAST=$(( NJOBS - 1 ))

echo "Total samples : ${TOTAL}"
echo "Samples / job : ${SAMPLES_PER_JOB}"
echo "Array tasks   : ${NJOBS}  (indices 0-${LAST}, up to ${MAX_CONCURRENT} concurrent)"

mkdir -p .slurm
sbatch --array=0-${LAST}%${MAX_CONCURRENT} translate.sh \
    "${INPUT}" "${OUTPUT_DIR}" "${SAMPLES_PER_JOB}"