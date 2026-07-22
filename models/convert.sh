CUDA_VISIBLE_DEVICES=0 torchrun \
    --nproc_per_node=1 \
    --master_addr=localhost \
    --master_port=29500 \
    $(which megatron) export \
        --mcore_model $WORK/multimodalModels/v241-20260615-170621/checkpoint-19250 \
        --output_dir $WORK/multimodalModels/v241-20260615-170621/checkpoint-19250-hf \
        --to_hf