#!/bin/bash
#SBATCH --job-name=export_v1_multimodal_4
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --account=ehpc678
#SBATCH --partition=gpp
#SBATCH --qos=acc_ehpc
#SBATCH --time=10:00:00
#SBATCH --output=log/%j_verify.out
#SBATCH --error=log/%j_verify.err

module purge
module load miniforge cuda/12.6 cudnn

source /gpfs/apps/MN5/ACC/MINIFORGE/24.3.0-0/etc/profile.d/conda.sh
conda activate /gpfs/projects/ehu26/envs/megatron-swift

cd ~/mLatxa/datasets
which python3
python3 verify/tokenization.py ~/scratch/data/multimodal_eu_124/train/ 124