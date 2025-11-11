#!/bin/bash

export CUDA_VISIBLE_DEVICES=2,3
cd /data/lzm/DrugRecommend
export PATH="/home/202312150002/anaconda3/bin:$PATH"
source /home/202312150002/anaconda3/etc/profile.d/conda.sh
conda activate llm

export PYTHONPATH="/data/lzm/DrugRecommend:$PYTHONPATH"

NUM_GPUS=2
MASTER_PORT=29500
LOG_FILE="/data/lzm/DrugRecommend/resource/output/checkpoint_save/baseline_1106/train_$(date +%Y%m%d_%H%M%S).log"
echo "🚀 Distributed training using deepspeed..."

torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    -m src.trainer.baseline_trainer \
    --train_file /data/lzm/DrugRecommend/src/worker/dataset/train.jsonl \
    --drug_file /data/lzm/DrugRecommend/src/worker/dataset/pre_drug.json \
    --output_dir /data/lzm/DrugRecommend/resource/output/checkpoint_save/baseline_1106 \
    --evaluation_strategy steps \
    --model_name_or_path /data1/nuist_llm/TrainLLM/ModelCkpt/qwen3-other/1point5B \
    --max_seq_length 2500 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-5 \
    --num_train_epochs 2 \
    --use_focal_loss \
    --use_lora \
    --bf16 \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --logging_steps 50 \
    --save_steps 200 \
    --save_total_limit 3 \
    --is_train \
    --deepspeed_config /data/lzm/DrugRecommend/src/configs/zero3.json \
    --distributed \
    2>&1 | tee "$LOG_FILE"