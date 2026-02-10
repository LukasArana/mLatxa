#!/bin/bash
#SBATCH --cpus-per-task=16
#SBATCH --time=72:00:00
#SBATCH --mem=20G
#SBATCH --partition=hitz-exclusive
#SBATCH --account=hitz-exclusive
#SBATCH --job-name=download_finevision
#SBATCH --output=out/move_finevision_%j.out
#SBATCH --error=err/move_finevision_%j.err
#SBATCH --export=ALL,SSH_AUTH_SOCK

# 1. Define your password (WARNING: This is visible in this file)
# 2. Use sshpass to feed the password to the rsync/ssh command
rsync -avz -P -e "ssh -i $HOME/.ssh/cineca -o IdentitiesOnly=yes" \
/hitz_data/larana/finevision/ laranaga@dmover1.leonardo.cineca.it:/leonardo_scratch/fast/AIFAC_5C0_261/datasets/train/finevision/
