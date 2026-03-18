swift export \
    --model /leonardo_work/EUHPC_E04_042/BaseModels/Qwen3-VL-32B-Instruct \
    --to_mcore true \
    --torch_dtype bfloat16 \
    --output_dir /leonardo_work/EUHPC_E04_042/BaseModels/Qwen3-VL-32B-Instruct_megatron \
    --test_convert_precision true
