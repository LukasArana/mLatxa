#!/bin/bash
#SBATCH --job-name=dataset_translation_%A_%a
#SBATCH --cpus-per-task=10
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --mem=100G
#SBATCH --output=/sorgin1/users/larana/Translation/slurm/dataset_translation_eu.out
#SBATCH --error=/sorgin1/users/larana/Translation/slurm/dataset_translation_eu.err

export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
export LANGUAGE=en_US.UTF-8
export TRANSFORMERS_NO_ADVISORY_WARNINGS=true
export OMP_NUM_THREADS=16
export VLLM_WORKER_MULTIPROC_METHOD=spawn

conda activate translation
cd /home/larana/mLatxa/datasets/translate
ls
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"

# Run the Python script
python3 mmstar/mm_star.py \
  --model_path HiTZ/Latxa-Llama-3.1-70B-Instruct\
  --dataset_path Lin-Chen/MMStar \
  --prompt_lang eu \
  --output_path "output/mm_star_eu.json" \
  --frequency_penalty 0.05 \
  --tensor_parallel_size 2