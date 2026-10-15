#!/bin/bash
#SBATCH --cpus-per-task=16
#SBATCH --time=72:00:00
#SBATCH --mem=20G
#SBATCH --partition=hitz-exclusive
#SBATCH --account=hitz-exclusive
#SBATCH --job-name=move_finevision
#SBATCH --output=out/move_finevision_%j.out
#SBATCH --error=err/move_finevision_%j.err
#SBATCH --export=ALL,SSH_AUTH_SOCK

# Copy the FineVision download (HiTZ) to Leonardo. Set DATA_DIR, LEO_USER and LEO_FAST
# (Leonardo's $FAST, e.g. /leonardo_scratch/fast/<account>) before submitting.
rsync -avz -P -e "ssh -i $HOME/.ssh/cineca -o IdentitiesOnly=yes" \
"${DATA_DIR:?set DATA_DIR}/finevision/" "${LEO_USER:?set LEO_USER}@dmover1.leonardo.cineca.it:${LEO_FAST:?set LEO_FAST}/datasets/train/finevision/"
