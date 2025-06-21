export CUDA_VISIBLE_DEVICES=3,2
export WANDB_DISABLED=true
set -ex
LOG_PATH=./log.txt
SAVE_PATH=/data/zhuotaodeng/yzj/alpha-research/model/qwen25_grm_iclr_boxed
mkdir -p $SAVE_PATH

torchrun --nproc_per_node=2 \
    --master_port=20011 \
    train.py \
    --model_name_or_path /data/zhuotaodeng/yzj/download_from_modelscope/Qwen/Qwen2___5-7B-Instruct \
    --data_path /data/zhuotaodeng/yzj/alpha-research/data/iclr_train_all_boxed.json \
    --bf16 True \
    --tf32 True \
    --output_dir $SAVE_PATH \
    --num_train_epochs 2 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 32 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 60 \
    --save_total_limit 17 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_steps 100 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'Qwen2DecoderLayer' \
    --model_max_length 16384 \
    --gradient_checkpointing True \
    --lazy_preprocess False
     
    > $LOG_PATH 2>&1
