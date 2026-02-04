#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --partition=lrd_all_serial
#SBATCH --qos=normal
#SBATCH --time=4:00:00
#SBATCH --account=AIFAC_5C0_261
#SBATCH --job-name=download_finevision
#SBATCH --output=out/download_finevision_%j.out
#SBATCH --error=err/download_finevision_%j.err

HF_XET_HIGH_PERFORMANCE=1 python3 download_finevision.py