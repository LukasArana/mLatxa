#!/bin/bash
#SBATCH --job-name=finevision_pipeline
#SBATCH --partition=boost_usr_prod
#SBATCH --account=AIFAC_5C0_261
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=450GB
#SBATCH --time=24:00:00
#SBATCH --output=logs/finevision_pipeline_%j.out
#SBATCH --error=logs/finevision_pipeline_%j.err

mkdir -p logs

# 1. Activate environment
source /leonardo_work/AIFAC_5C0_261/environments/env_torch_2_9_megratron/bin/activate

# 2. Create cache directories on the LARGE volume
mkdir -p /leonardo_work/AIFAC_5C0_261/hf_cache
mkdir -p /leonardo_work/AIFAC_5C0_261/modelscope_cache
mkdir -p /leonardo_work/AIFAC_5C0_261/tmp

# 3. Set environment variables
export HF_DATASETS_CACHE="/leonardo_work/AIFAC_5C0_261/hf_cache"
export HF_HOME="/leonardo_work/AIFAC_5C0_261/hf_cache"
export MODELSCOPE_CACHE="/leonardo_work/AIFAC_5C0_261/modelscope_cache"
export TMPDIR="/leonardo_work/AIFAC_5C0_261/tmp"

# 4. Clean small caches on the root disk
rm -rf ~/.cache/huggingface
rm -rf ~/.cache/modelscope

cd $HOME/mLatxa/datasets

echo "Starting FineVision pipeline..."
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURM_NODELIST"
echo "CPUs:    $SLURM_CPUS_PER_TASK"
echo "Started: $(date)"

python finevision_pipeline.py \
    --dataset_name /leonardo_work/AIFAC_5C0_261/datasets/train/finevision \
    --output_path /leonardo_work/AIFAC_5C0_261/datasets/train/finevisionProcessed/train.jsonl \
    --images_dir /leonardo_work/AIFAC_5C0_261/datasets/train/finevisionProcessed/images \
    --tokenizer_path /leonardo_work/AIFAC_5C0_261/baseModels/Qwen3.5-9B \
    --instruct_path /leonardo_work/AIFAC_5C0_261/datasets/datasets/InstructDatasets/magpie.qwen3.32b.en.noreasoning.onlyconver.jsonl \
    --num_workers $SLURM_CPUS_PER_TASK \
    --seed 42

echo "Finished: $(date)"
