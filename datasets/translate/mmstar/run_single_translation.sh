#!/bin/bash
#SBATCH --job-name=mmstar_translation
#SBATCH --cpus-per-task=10
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --output=/home/larana/mLatxa/datasets/translate/mmstar/out/mm_star.out
#SBATCH --error=/home/larana/mLatxa/datasets/translate/mmstar/err/mm_star.err

export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
export LANGUAGE=en_US.UTF-8
export TRANSFORMERS_NO_ADVISORY_WARNINGS=true
export OMP_NUM_THREADS=16
export VLLM_WORKER_MULTIPROC_METHOD=spawn

cd /home/larana/mLatxa/datasets/translate

source /home/larana/mLatxa/datasets/translate/mmstar/env/bin/activate

echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"

# Run the Python script
python3 mmstar/mm_star.py \
  --model_path HiTZ/Latxa-Llama-3.1-70B-Instruct\
  --dataset_path Lin-Chen/MMStar \
  --prompt_lang eu \
  --output_path "output/mm_star_eu.json" \
  --frequency_penalty 0.05 \
  --tensor_parallel_size 2