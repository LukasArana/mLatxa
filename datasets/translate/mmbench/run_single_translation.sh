#!/bin/bash
#SBATCH --job-name=mmbench_translation_es
#SBATCH --cpus-per-task=10
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --mem=100G
#SBATCH --output=out/mmbench_translation_es.out
#SBATCH --error=err/mmbench_translation_es.err
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
export LANGUAGE=en_US.UTF-8
export TRANSFORMERS_NO_ADVISORY_WARNINGS=true
export OMP_NUM_THREADS=16
export VLLM_WORKER_MULTIPROC_METHOD=spawn

cd "$HOME/mLatxa/datasets/translate/mmbench" || exit

# Activate conda environment
source "${CONDA_ROOT:-$HOME/miniforge3}/etc/profile.d/conda.sh"
conda activate translation2

echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"

# Run the Python script
python3 mm_bench.py \
  --model_path meta-llama/Llama-3.1-70B-Instruct \
  --dataset_path ~/LMUData/MMBench_dev_en.tsv \
  --prompt_lang es \
  --output_path "output/mm_bench_es.json" \
  --frequency_penalty 0.15 \
  --tensor_parallel_size 2
