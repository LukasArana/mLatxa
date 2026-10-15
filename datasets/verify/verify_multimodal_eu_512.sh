#!/bin/bash
#SBATCH --job-name=verify_multimodal_eu_512
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --account=ehpc678
#SBATCH --partition=gpp
#SBATCH --qos=acc_ehpc
#SBATCH --time=10:00:00
#SBATCH --output=log/verify_multimodal_eu_512_%j.out
#SBATCH --error=log/verify_multimodal_eu_512_%j.err

module purge
module load miniforge cuda/12.6 cudnn

source /gpfs/apps/MN5/ACC/MINIFORGE/24.3.0-0/etc/profile.d/conda.sh
conda activate megatron-swift
set -eo pipefail
export SCRATCH=~/scratch

cd ~/mLatxa/datasets
$SCRATCH/ehu26/megatron-swift-v2/megatron-swift/bin/python3 verify/tokenization.py $SCRATCH/ehu26/data/multimodal_eu_512/train/ 512