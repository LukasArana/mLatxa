#!/bin/bash
# Merge the 8 exported parts of multimodal_4 or multimodal_4_eu (export_multimodal_4_part.sh) into one cached dataset.
# Submit from ~/mLatxa/datasets/export (or run with bash on a compute node), a few minutes:
#   sbatch --job-name=merge_multimodal_4_eu scripts_mn5/merge_multimodal_4_parts.sh multimodal_4_eu
# Output: $SCRATCH/data/<name>/train. The parts are kept; delete them yourself once the merge is checked.
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --account=ehu26
#SBATCH --partition=gpp
#SBATCH --qos=acc_resa
#SBATCH --time=02:00:00
#SBATCH --output=log/%x-%j.out
#SBATCH --error=log/%x-%j.err

NAME=$1   # multimodal_4 or multimodal_4_eu
module purge
module load miniforge
source /gpfs/apps/MN5/ACC/MINIFORGE/24.3.0-0/etc/profile.d/conda.sh
conda activate /gpfs/projects/ehu26/envs/megatron-swift
set -eo pipefail

export SCRATCH=~/scratch
export TMPDIR=$SCRATCH/tmp HF_DATASETS_CACHE=$SCRATCH/hf_cache

python - "$NAME" <<'PY'
import os, sys
from datasets import concatenate_datasets, load_from_disk

name = sys.argv[1]
data = os.path.expanduser('~/scratch/data')
out = f'{data}/{name}/train'
if os.path.exists(out):
    sys.exit(f'{out} already exists')
parts = [load_from_disk(f'{data}/{name}_p{k}/train') for k in range(8)]
for k, p in enumerate(parts):
    print(f'{name}_p{k}: {len(p)} rows')
merged = concatenate_datasets(parts)
merged.save_to_disk(out, num_proc=8)
check = load_from_disk(out)
assert len(check) == sum(len(p) for p in parts) and check.column_names == parts[0].column_names
print(f'{out}: {len(check)} rows, columns {check.column_names}')
PY
echo "Merge complete!"
