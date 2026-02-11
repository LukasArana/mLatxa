# 22GB
source /leonardo_work/AIFAC_5C0_261/environments/env_torch_2_9/bin/activate

swift sft \
    --model /leonardo_work/EUHPC_E04_042/BaseModels/Qwen3-VL-4B-Instruct \
    --train_type full \
    --dataset '/leonardo_scratch/fast/AIFAC_5C0_261/datasets/train/finevisionjsonl/geo3k.jsonl#500' '/leonardo_work/EUHPC_E04_042/datasets/InstructDatasets/Magpie-Llama-3.1-70B-Instruct-Filtered-100K.jsonl#500' '/leonardo_work/EUHPC_E04_042/datasets/PretrainDatasets/Latxa_booktegi_test.jsonl#500'\
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 1 \
    --save_total_limit 2 \
    --logging_steps 1 \
    --output_dir /leonardo_work/AIFAC_5C0_261/lukas/msoutput \
    --system 'You are a helpful assistant.' \
    --warmup_ratio 0.05 \
    --max_length 8192 \
    --packing \
    --attn_impl flash_attention_2 \
    --dataloader_num_workers 8 \
    --model_author swift \
    --model_name swift-robot \
    --max_pixels 65536 \