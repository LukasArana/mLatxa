#!/bin/bash
#SBATCH --job-name=sample_finevision
#SBATCH --partition=boost_usr_prod
#SBATCH --account=AIFAC_5C0_261
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400GB
#SBATCH --time=12:00:00
#SBATCH --output=logs/sample_finevision_%j.out
#SBATCH --error=logs/sample_finevision_%j.err

# Create logs directory if it doesn't exist
mkdir -p logs

# Load modules if needed
# module load python/3.10

# Activate environment if needed

source /leonardo_work/AIFAC_5C0_261/environments/env_torch_2_9_megratron/bin/activate

cd $HOME/mLatxa/datasets

echo "Starting FineVision sampling..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Time: $(date)"

python sample_finevision.py \
    --finevision_path /leonardo_work/AIFAC_5C0_261/datasets/train/finevision \
    --instruct_path /leonardo_work/AIFAC_5C0_261/datasets/datasets/InstructDatasets/magpie.qwen3.32b.en.noreasoning.onlyconver.jsonl \
    --tokenizer_path $WORK/baseModels/Qwen3.5-9B \
    --output_path $WORK/datasets/finevision_sampled.jsonl \
    --num_workers 32 \
    --seed 42

echo "Finished at: $(date)"
