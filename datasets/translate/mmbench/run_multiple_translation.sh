#!/bin/bash
#SBATCH --job-name=mmbench_translation
#SBATCH --cpus-per-task=10
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --mem=100G
#SBATCH --output=out/mmbench_translation_%A_%a.out
#SBATCH --error=err/mmbench_translation_%A_%a.err
#SBATCH --array=0-1  # one task per target language (see case below)
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
export LANGUAGE=en_US.UTF-8
export TRANSFORMERS_NO_ADVISORY_WARNINGS=true
export OMP_NUM_THREADS=16
export VLLM_WORKER_MULTIPROC_METHOD=spawn

echo "Job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} starting on $(hostname)"

cd "$HOME/mLatxa/datasets/translate/mmbench" || exit

# Activate conda environment
source "${CONDA_ROOT:-$HOME/miniforge3}/etc/profile.d/conda.sh"
conda activate translation2

# Language and model mapping based on SLURM array task ID
case $SLURM_ARRAY_TASK_ID in
0)
  TGT_LANG="eu"
  MODEL="HiTZ/Latxa-Llama-3.1-70B-Instruct"
  ;;
1)
  TGT_LANG="es"
  MODEL="meta-llama/Llama-3.1-70B-Instruct"
  ;;
*)
  echo "Unexpected SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
  exit 1
  ;;
esac

echo "Running language: ${TGT_LANG}, model: ${MODEL}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"

# Run the Python script
python3 mm_bench.py \
  --model_path "${MODEL}" \
  --dataset_path ~/LMUData/MMBench_dev_en.tsv \
  --prompt_lang "${TGT_LANG}" \
  --output_path "output/mm_bench_${TGT_LANG}.json" \
  --frequency_penalty 0.15 \
  --tensor_parallel_size 2
