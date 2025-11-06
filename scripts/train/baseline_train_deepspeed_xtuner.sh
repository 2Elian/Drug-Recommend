#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,2
export NPROC_PER_NODE=2
export NNODES=1
export NODE_RANK=0
export PORT=10088
export NODE_0_ADDR=172.16.107.15

export PYTHONPATH="/data/lzm/DrugRecommend:$PYTHONPATH"

# cuda
export PATH="/usr/local/cuda-12.4/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH"
export CUDA_HOME="/usr/local/cuda-12.4"

# conda
export PATH="/home/202312150002/anaconda3/bin:$PATH"
source /home/202312150002/anaconda3/etc/profile.d/conda.sh
conda activate llm
cd /data/lzm/DrugRecommend

# set api key for swanLab
export SWANLAB_API_KEY=97BX7B3wovtGCgxYJBSJJ
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export NCCL_SOCKET_IFNAME=lo,eth0 
LOG_FILE="/data/lzm/DrugRecommend/resource/output/checkpoint_save/baseline_1106_xtuner/train_$(date +%Y%m%d_%H%M%S).log"

# start training
xtuner train /data/lzm/DrugRecommend/scripts/train/xtuner_config.py \
    --deepspeed deepspeed_zero3 \
    --work-dir "/data/lzm/DrugRecommend/resource/output/checkpoint_save/baseline_1106_xtuner" \
    --cfg-options model.llm._attn_implementation=flash_attention_2
    2>&1 | tee "$LOG_FILE"