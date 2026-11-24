#!/bin/bash
# Export one of the 8 parts of multimodal_4 or multimodal_4_eu (10M rows each do not fit in the 3-day limit).
# Submit from ~/mLatxa/datasets/export:
#   sbatch --job-name=export_multimodal_4_eu_p0 scripts_mn5/export_multimodal_4_part.sh multimodal_4_eu 0
# Output: $SCRATCH/data/<name>_p<part>/train
#SBATCH --cpus-per-task=64
#SBATCH --nodes=1
#SBATCH --account=ehu26
#SBATCH --partition=gpp
#SBATCH --qos=acc_resa
#SBATCH --time=72:00:00
#SBATCH --output=log/%x-%j.out
#SBATCH --error=log/%x-%j.err

NAME=$1   # multimodal_4 or multimodal_4_eu
PART=$2   # 0..7
NPARTS=8

# 1. Environment
module purge
module load miniforge cuda/12.6 cudnn
source /gpfs/apps/MN5/ACC/MINIFORGE/24.3.0-0/etc/profile.d/conda.sh
conda activate /gpfs/projects/ehu26/envs/megatron-swift
set -eo pipefail

export SCRATCH=~/scratch
export HF_DATASETS_CACHE=$SCRATCH/hf_cache
export HF_HOME=$SCRATCH/hf_cache
export MODELSCOPE_CACHE=$SCRATCH/modelscope_cache
export TMPDIR=$SCRATCH/tmp
mkdir -p $HF_HOME $MODELSCOPE_CACHE $TMPDIR

export Z3_LIB_PATH=$(python -c 'import z3, os; print(os.path.join(z3.__path__[0], "lib"))')
export LD_LIBRARY_PATH="$Z3_LIB_PATH:$LD_LIBRARY_PATH"

export IMAGE_MAX_TOKEN_NUM=124
export video_min_token_num=0
export video_max_token_num=0

# 2. The input file of the dataset
DIR=$SCRATCH/data/leonardo/InstructionDatasets/multimodal
case $NAME in
  multimodal_4)    SRC=$DIR/train_multimodal4_fixed.jsonl ;;
  multimodal_4_eu) SRC=$DIR/train_multimodal_eu_4.jsonl ;;
  *) echo "unknown dataset: $NAME" >&2; exit 1 ;;
esac
[[ $PART =~ ^[0-7]$ ]] || { echo "part must be 0..7, got: $PART" >&2; exit 1; }

# 3. This part: every 8th line, starting at line PART+1
mkdir -p $DIR/parts
PART_FILE=$DIR/parts/${NAME}_p${PART}.jsonl
if [ ! -s $PART_FILE ]; then
  awk -v n=$NPARTS -v k=$PART '(NR - 1) % n == k' $SRC > $PART_FILE.tmp
  mv $PART_FILE.tmp $PART_FILE
fi
echo "part $PART of $NAME: $(wc -l < $PART_FILE) rows"

# 4. Export
CFG=$(mktemp $TMPDIR/${NAME}_p${PART}.XXXX.json)
echo "[{\"dataset_name\": \"${NAME}_p${PART}\", \"dataset_path\": \"$PART_FILE\"}]" > $CFG

swift export \
    --model $SCRATCH/basemodels/Qwen3.5-9B \
    --custom_dataset_info $CFG \
    --dataset ${NAME}_p${PART} \
    --template qwen3_5 \
    --max_length 8192 \
    --dataset_num_proc 128 \
    --to_cached_dataset \
    --output_dir $SCRATCH/data/${NAME}_p${PART}/

echo "Export complete!"
