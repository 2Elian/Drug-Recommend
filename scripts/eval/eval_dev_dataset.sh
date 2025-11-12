# eval dev dataset, and the dev dataset is obtained by splitting the test set by 20%
#!/bin/bash
echo "🚀 start eval..."
python -m src.evaluation.evaler \
  --base_model_path /data1/nuist_llm/TrainLLM/ModelCkpt/qwen3-other/1point5B \
  --model_path /data/lzm/DrugRecommend/resource/output/checkpoint_save/baseline_1106 \
  --test_file /data/lzm/DrugRecommend/src/worker/dataset/eval.jsonl \
  --save_path /data/lzm/DrugRecommend/resource/output/val/qwen1point7-baseline/predict_resullt.json \
  --pre_drug_path /data/lzm/DrugRecommend/src/worker/dataset/pre_drug.json \
  --batch_size 1 \
  --gpu_id 2 \
  --max_seq_length 2500 \
  # --is_submit