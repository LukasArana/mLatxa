#!/bin/bash
#SBATCH --cpus-per-task=2
#SBATCH --time=72:00:00

#SBATCH --partition=hitz-exclusive
#SBATCH --account=hitz-exclusive
#SBATCH --job-name=download_finevision
#SBATCH --output=out/download_finevision_%j.out
#SBATCH --error=err/download_finevision_%j.err

source env/bin/activate
HF_XET_HIGH_PERFORMANCE=1 python3 download_finevision.py