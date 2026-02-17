source /leonardo_work/AIFAC_5C0_261/environments/env_torch_2_9/bin/activate

swift export \
    --model /leonardo_work/EUHPC_E04_042/BaseModels/Qwen3-VL-4B-Instruct \
    --dataset '/leonardo_work/AIFAC_5C0_261/datasets/train/finevisionjsonl/geo3k.jsonl' \
    --dataset_num_proc 64 \
    --exist_ok true \
    --split_dataset_ratio 0.1 \
    --to_cached_dataset true \
    --output_dir /leonardo_work/AIFAC_5C0_261/datasets/train/preprocessed/geo3k/