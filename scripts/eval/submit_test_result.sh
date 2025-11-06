# submit the final result on the official website
# eval dev dataset, and the dev dataset is obtained by splitting the test set by 20%
#!/bin/bash
echo "🚀 start eval..."
python -m src.evaluation.evaler \
  --base_model_path /data1/nuist_llm/TrainLLM/ModelCkpt/glm/glm4-8b-chat \
  --model_path /data/lzm/DrugRecommend/src/worker/baseline/output/drug_prediction_deepspeed \
  --test_file /data/lzm/DrugRecommend/src/data/CDrugRed-B-v1/CDrugRed_test-B.jsonl \
  --save_path /data/lzm/DrugRecommend/resource/output/submit/predict_resullt_1106.json \
  --pre_drug_path /data/lzm/DrugRecommend/src/worker/dataset/pre_drug.json \
  --batch_size 1 \
  --gpu_id 2 \
  --max_seq_length 2000 \
  --threshold 0.5 \
  --is_submit