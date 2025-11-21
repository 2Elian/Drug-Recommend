#!/bin/bash

export CUDA_VISIBLE_DEVICES=2,3
cd /data/lzm/DrugRecommend
export PATH="/home/202312150002/anaconda3/bin:$PATH"
source /home/202312150002/anaconda3/etc/profile.d/conda.sh
conda activate llm

export PYTHONPATH="/data/lzm/DrugRecommend:$PYTHONPATH"

NUM_GPUS=2
MASTER_PORT=29501
LOG_FILE="/data/lzm/DrugRecommend/resource/output/checkpoint_save/q3_model_check/train_$(date +%Y%m%d_%H%M%S).log"
echo "🚀 Drug Model training..."

torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    -m src.trainer.q3_trainer \
    --train_file /data/lzm/DrugRecommend/src/data/CDrugRed-A-v1/CDrugRed_train.jsonl \
    --eval_file /data/lzm/DrugRecommend/src/worker/dataset/eval.jsonl \
    --drug_file /data/lzm/DrugRecommend/src/worker/dataset/pre_drug.json \
    --output_dir /data/lzm/DrugRecommend/resource/output/checkpoint_save/q3_model_check \
    --evaluation_strategy steps \
    --model_name_or_path /data1/nuist_llm/TrainLLM/ModelCkpt/glm/glm4-8b-chat \
    --max_seq_length 2500 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-5 \
    --num_train_epochs 10 \
    --use_focal_loss \
    --use_lora \
    --bf16 \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --logging_steps 1 \
    --eval_steps 2 \
    --save_steps 2 \
    --save_total_limit 3 \
    --is_gener \
    2>&1 | tee "$LOG_FILE"