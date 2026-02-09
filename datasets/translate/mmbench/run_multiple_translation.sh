#!/bin/bash
#SBATCH --job-name=dataset_translation_%A_%a
#SBATCH --cpus-per-task=10
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --mem=100G
#SBATCH --output=/sorgin1/users/larana/Translation/slurm/dataset_translation_%j.out
#SBATCH --error=/sorgin1/users/larana/Translation/slurm/dataset_translation_%j.err
#SBATCH --array=0-4%4  # 5 tasks for 5 languages, max 4 concurrent jobs
sh
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
export LANGUAGE=en_US.UTF-8
export TRANSFORMERS_NO_ADVISORY_WARNINGS=true
export OMP_NUM_THREADS=16
export VLLM_WORKER_MULTIPROC_METHOD=spawn

echo "Job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} starting on $(hostname)"

cd /sorgin1/users/larana/Translation || exit

# Activate conda environment
source "/sorgin1/users/larana/miniforge3/etc/profile.d/conda.sh"
conda activate translation2

# Language and model mapping based on SLURM array task ID
case $SLURM_ARRAY_TASK_ID in
0)
  LANG="eu"
  MODEL="HiTZ/Latxa-Llama-3.1-70B-Instruct"
  ;;
1)
  LANG="es"
  MODEL="meta-llama/Llama-3.1-70B-Instruct"
  ;;
*)
  echo "Unexpected SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
  exit 1
  ;;
esac

echo "Running language: ${LANG}, model: ${MODEL}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"

# Run the Python script
python3 mm_bench.py \
  --model_path "${MODEL}" \
  --dataset_path ~/LMUData/MMBench_dev_en.tsv \
  --prompt_lang "${LANG}" \
  --output_path "output/mm_bench_${LANG}.json" \
  --frequency_penalty 0.15 \
  --tensor_parallel_size 2