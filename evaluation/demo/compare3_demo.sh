#!/bin/bash
#SBATCH --job-name=compare3
#SBATCH --output=compare3_output.txt
#SBATCH --error=compare3_error.txt
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --mem=100G
#SBATCH --time=24:00:00

cd ~/mLatxa/evaluation/demo

source ~/multimodal/envDemo/bin/activate

python3 compare3_demo.py --share
