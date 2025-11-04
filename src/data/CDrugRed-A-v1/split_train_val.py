#!/usr/bin/env python3
import json
import random
import sys

def split_simple(input_file, output_dir, eval_ratio=0.2, seed=42):
    random.seed(seed)
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    random.shuffle(lines)
    split_idx = int(len(lines) * (1 - eval_ratio))
    
    train_lines = lines[:split_idx]
    eval_lines = lines[split_idx:]
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'train.jsonl'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(train_lines))
    
    with open(os.path.join(output_dir, 'eval.jsonl'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(eval_lines))
    
    print(f"✅ 划分完成!")
    print(f"📊 训练集: {len(train_lines)} 条")
    print(f"📊 评估集: {len(eval_lines)} 条")

if __name__ == "__main__":
    split_simple("/data/lzm/DrugRecommend/src/data/CDrugRed-A-v1/CDrugRed_train.jsonl", "/data/lzm/DrugRecommend/src/models/baseline/dataset")