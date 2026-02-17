source /leonardo_work/AIFAC_5C0_261/environments/env_torch_2_9/bin/activate

swift export \
    --model /leonardo_work/EUHPC_E04_042/BaseModels/Qwen3-VL-4B-Instruct \
    --dataset '/leonardo_work/EUHPC_E04_042/datasets/InstructDatasets/Magpie-Llama-3.1-70B-Instruct-Filtered-100K.jsonl' \
    --exist_ok true \
    --dataset_num_proc 64 \
    --split_dataset_ratio 0.1 \
    --to_cached_dataset true \
    --output_dir /leonardo_work/AIFAC_5C0_261/datasets/train/preprocessed/ins/