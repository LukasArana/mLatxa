#!/bin/bash
#SBATCH --job-name=verify_multimodal_eu_124
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --account=ehpc678
#SBATCH --partition=gpp
#SBATCH --qos=acc_ehpc
#SBATCH --time=10:00:00
#SBATCH --output=log/verify_multimodal_eu_124_%j.out
#SBATCH --error=log/verify_multimodal_eu_124_%j.err

module purge
module load miniforge cuda/12.6 cudnn

source /gpfs/apps/MN5/ACC/MINIFORGE/24.3.0-0/etc/profile.d/conda.sh
conda activate megatron-swift
set -eo pipefail
export SCRATCH=~/scratch

cd ~/mLatxa/datasets
which python3
python3 verify/tokenization.py $SCRATCH/data/multimodal_eu_124/train/ 124