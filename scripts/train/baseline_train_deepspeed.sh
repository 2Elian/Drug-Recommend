#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,2
cd /data/lzm/DrugRecommend
export PATH="/home/202312150002/anaconda3/bin:$PATH"
source /home/202312150002/anaconda3/etc/profile.d/conda.sh
conda activate llm

export PYTHONPATH="/data/lzm/DrugRecommend:$PYTHONPATH"

NUM_GPUS=2
MASTER_PORT=29500

echo "🚀 Distributed training using deepspeed..."

torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    src/models/trainer/baseline_trainer.py \
    --train_file /data/lzm/DrugRecommend/src/worker/dataset/train.jsonl \
    --drug_file /data/lzm/DrugRecommend/src/worker/dataset/pre_drug.json \
    --output_dir /data/lzm/DrugRecommend/resource/output/checkpoint_save/baseline_1105 \
    --evaluation_strategy steps \
    --model_name_or_path /data1/nuist_llm/TrainLLM/ModelCkpt/glm/glm4-8b-chat \
    --max_seq_length 2000 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-5 \
    --num_train_epochs 2 \
    --use_focal_loss \
    --use_lora \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --logging_steps 50 \
    --save_steps 200 \
    --save_total_limit 3 \
    --use_deepspeed \
    --deepspeed_config /data/lzm/DrugRecommend/src/configs/zero3.json \
    --distributed