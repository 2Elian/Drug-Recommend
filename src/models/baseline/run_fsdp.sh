#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,2
cd /data/lzm/DrugRecommend
export PATH="/home/202312150002/anaconda3/bin:$PATH"
source /home/202312150002/anaconda3/etc/profile.d/conda.sh
conda activate llm

export PYTHONPATH="/data/lzm/DrugRecommend:$PYTHONPATH"

NUM_GPUS=2
MASTER_PORT=29500

echo "PYTHONPATH: $PYTHONPATH"
echo "Current directory: $(pwd)"
echo "🚀 Distributed training using FSDP..."
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    src/models/baseline/baseline_main.py \
    --train_file /data/lzm/DrugRecommend/src/models/baseline/dataset/train.jsonl \
    --eval_file /data/lzm/DrugRecommend/src/models/baseline/dataset/eval.jsonl \
    --drug_file /data/lzm/DrugRecommend/src/models/baseline/dataset/pre_drug.json \
    --output_dir outputs/drug_prediction_fsdp \
    --evaluation_strategy steps \
    --model_name_or_path /data1/nuist_llm/TrainLLM/ModelCkpt/glm/glm4-8b-chat \
    --max_seq_length 11234 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-5 \
    --num_train_epochs 10 \
    --use_focal_loss \
    --use_lora \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --logging_steps 50 \
    --eval_steps 500 \
    --save_steps 500 \
    --load_best_model_at_end \
    --fp16 \
    --use_fsdp \
    --fsdp_sharding_strategy FULL_SHARD \
    --fsdp_offload_params \
    --gradient_checkpointing \
    --distributed