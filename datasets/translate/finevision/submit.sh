# submit.sh -- compute the SLURM array size from the input and submit.
#
# Usage:
#   ./submit.sh <input.jsonl> <samples_per_job> [max_parallel_jobs]
#
# Example (2M samples, 20k per job => 100 jobs, at most 25 running at once):
#   ./submit.sh /path/conversations_en.jsonl 20000 25

set -euo pipefail

INPUT="${1:?usage: submit.sh <input.jsonl> <samples_per_job> [max_parallel]}"
N="${2:?usage: submit.sh <input.jsonl> <samples_per_job> [max_parallel]}"
MAXP="${3:-20}"

echo "Counting lines in $INPUT ..."
TOTAL=$(wc -l < "$INPUT")
# number of jobs = ceil(TOTAL / N); array indices are 0-based, so last index = JOBS-1
JOBS=$(( (TOTAL + N - 1) / N ))
LAST=$(( JOBS - 1 ))

echo "Total samples : $TOTAL"
echo "Samples/job   : $N"
echo "Array jobs    : $JOBS  (indices 0-$LAST)"
echo "Max parallel  : $MAXP"

# Make sure translate.slurm uses the same INPUT/SAMPLES_PER_JOB, then submit:
sbatch --array=0-${LAST}%${MAXP} translate.slurm
