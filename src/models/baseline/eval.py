import json
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from os.path import join
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import AutoTokenizer
import argparse

from src.models.baseline.utils import load_drug_vocab

def setup_device():
    """设置设备"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"使用 GPU: {torch.cuda.get_device_name()}")
        print(f"GPU 内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        device = torch.device("cpu")
        print("使用 CPU")
    return device

def load_model_and_tokenizer(model_path , device):
    """加载模型和tokenizer到指定设备"""
    from src.models.baseline.model import GenericForMultiLabelClassification
    
    drug_to_idx, idx_to_drug, all_drugs = load_drug_vocab("/data/lzm/DrugRecommend/src/models/baseline/dataset/pre_drug.json")
    num_drugs = len(all_drugs)
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    
    # 加载模型
    model = GenericForMultiLabelClassification(
        model_name_or_path=model_path,
        num_labels=num_drugs,
        use_focal_loss=True,
        focal_alpha=0.75,
        focal_gamma=2.0,
    )
    
    # 加载模型权重并移动到GPU
    model.load_state_dict(torch.load(join(model_path, "pytorch_model.bin"), map_location="cpu"))
    model.to(device)
    model.eval()
    
    print(f"模型已加载到 {device}")
    
    return model, tokenizer, drug_to_idx, idx_to_drug, num_drugs

def compute_metrics_fast(predictions, labels, thresholds=[0.3, 0.4, 0.5]):
    """快速计算多标签分类指标"""
    metrics = {}
    
    for threshold in thresholds:
        preds = (predictions > threshold).astype(int)
        
        # 向量化计算指标
        intersection = np.sum(labels & preds, axis=1)
        union = np.sum(labels | preds, axis=1)
        true_positives = np.sum(labels, axis=1)
        pred_positives = np.sum(preds, axis=1)
        
        # 避免除零
        jaccard = np.divide(intersection, union, out=np.zeros_like(intersection, dtype=float), where=union!=0)
        precision = np.divide(intersection, pred_positives, out=np.zeros_like(intersection, dtype=float), where=pred_positives!=0)
        recall = np.divide(intersection, true_positives, out=np.zeros_like(intersection, dtype=float), where=true_positives!=0)
        
        # F1 score
        f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(precision), where=(precision + recall)!=0)
        
        avg_jaccard = np.mean(jaccard)
        avg_precision = np.mean(precision)
        avg_recall = np.mean(recall)
        avg_f1 = np.mean(f1)
        score = 0.5 * (avg_jaccard + avg_f1)

        metrics.update({
            f"jaccard_th{threshold}": avg_jaccard,
            f"precision_avg_th{threshold}": avg_precision,
            f"recall_avg_th{threshold}": avg_recall,
            f"f1_avg_th{threshold}": avg_f1,
            f"score_th{threshold}": score,
        })
        
        # 添加micro指标
        metrics.update({
            f"f1_micro_th{threshold}": f1_score(labels, preds, average="micro", zero_division=0),
            f"precision_micro_th{threshold}": precision_score(labels, preds, average="micro", zero_division=0),
            f"recall_micro_th{threshold}": recall_score(labels, preds, average="micro", zero_division=0),
        })
    
    # 额外统计信息
    preds_05 = (predictions > 0.5).astype(int)
    metrics.update({
        "avg_labels_per_sample": np.mean(np.sum(labels, axis=1)),
        "avg_preds_per_sample": np.mean(np.sum(preds_05, axis=1)),
        "exact_match_accuracy": np.mean(np.all(preds_05 == labels, axis=1)),
        "total_samples": len(labels)
    })
    
    return metrics

def evaluate_test_set(model, tokenizer, test_file, drug_to_idx, num_drugs, device, max_seq_length=512, batch_size=32):
    """评估测试集 - GPU加速版本"""
    # 加载测试数据
    test_data = pd.read_json(test_file, lines=True)
    print(f"加载测试数据: {len(test_data)} 个样本")
    
    all_predictions = []
    all_labels = []
    all_texts = []
    
    # 收集所有真实标签
    for _, row in tqdm(test_data.iterrows(), total=len(test_data), desc="准备标签"):
        text = row.get("text", "")
        drugs = row.get("drugs", [])
        
        # 创建多标签向量
        label_vector = np.zeros(num_drugs, dtype=int)
        for drug in drugs:
            if drug in drug_to_idx:
                label_vector[drug_to_idx[drug]] = 1
        
        all_labels.append(label_vector)
        all_texts.append(text)
    
    all_labels = np.array(all_labels)
    
    # 批量推理 - GPU加速
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(all_texts), batch_size), desc="GPU推理进度"):
            batch_texts = all_texts[i:i+batch_size]
            batch_input_ids = []
            batch_attention_mask = []
            
            # 处理批次数据
            for text in batch_texts:
                encoding = tokenizer(
                    text,
                    truncation=True,
                    padding=False,
                    max_length=max_seq_length,
                    return_tensors="pt"
                )
                batch_input_ids.append(encoding["input_ids"].squeeze(0))
                batch_attention_mask.append(encoding["attention_mask"].squeeze(0))
            
            # 填充批次并移动到GPU
            batch_input_ids = torch.nn.utils.rnn.pad_sequence(
                batch_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
            ).to(device)
            batch_attention_mask = torch.nn.utils.rnn.pad_sequence(
                batch_attention_mask, batch_first=True, padding_value=0
            ).to(device)
            
            # GPU推理
            outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
            logits = outputs.logits
            
            # 应用sigmoid得到概率并移回CPU
            batch_probs = torch.sigmoid(logits).cpu().numpy()
            all_predictions.extend(batch_probs)
            
            # 清理GPU内存
            del batch_input_ids, batch_attention_mask, outputs, logits
            
    # 强制清理GPU缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    all_predictions = np.array(all_predictions)
    
    # 计算指标
    metrics = compute_metrics_fast(all_predictions, all_labels)
    
    return metrics, all_predictions, all_labels

def evaluate_test_set_optimized(model, tokenizer, test_file, drug_to_idx, num_drugs, device, max_seq_length=512, batch_size=64):
    """进一步优化的评估版本，使用更大的batch_size"""
    test_data = pd.read_json(test_file, lines=True)
    print(f"加载测试数据: {len(test_data)} 个样本")
    
    all_predictions = []
    all_labels = []
    
    # 预处理所有标签
    all_labels = np.zeros((len(test_data), num_drugs), dtype=int)
    texts = []
    
    for idx, row in tqdm(test_data.iterrows(), total=len(test_data), desc="预处理数据"):
        text = row.get("text", "")
        drugs = row.get("drugs", [])
        
        for drug in drugs:
            if drug in drug_to_idx:
                all_labels[idx, drug_to_idx[drug]] = 1
        texts.append(text)
    
    # 批量推理
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="GPU批量推理"):
            batch_texts = texts[i:i+batch_size]
            
            # 使用tokenizer的批量处理
            encoding = tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=max_seq_length,
                return_tensors="pt"
            )
            
            # 移动到GPU
            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)
            
            # 推理
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            batch_probs = torch.sigmoid(outputs.logits).cpu().numpy()
            all_predictions.append(batch_probs)
            
            # 清理
            del input_ids, attention_mask, outputs
    
    # 合并预测结果
    all_predictions = np.vstack(all_predictions)
    
    # 清理GPU内存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 计算指标
    metrics = compute_metrics_fast(all_predictions, all_labels)
    
    return metrics, all_predictions, all_labels

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/data/lzm/DrugRecommend/src/models/baseline/output/drug_prediction_deepspeed/merge", help="训练好的模型路径")
    parser.add_argument("--test_file", type=str, default="/data/lzm/DrugRecommend/src/models/baseline/dataset/eval.jsonl", help="测试集文件路径")
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小（GPU可设置更大）")
    parser.add_argument("--max_seq_length", type=int, default=2000, help="最大序列长度")
    parser.add_argument("--optimized", action="store_true", help="使用优化版本（更快）")
    
    args = parser.parse_args()
    
    # 设置设备
    device = setup_device()
    
    print("加载模型和tokenizer...")
    model, tokenizer, drug_to_idx, idx_to_drug, num_drugs = load_model_and_tokenizer(
        args.model_path, device
    )
    
    print("开始评估测试集...")
    if args.optimized:
        print("使用优化版本评估...")
        metrics, predictions, labels = evaluate_test_set_optimized(
            model=model,
            tokenizer=tokenizer,
            test_file=args.test_file,
            drug_to_idx=drug_to_idx,
            num_drugs=num_drugs,
            device=device,
            max_seq_length=args.max_seq_length,
            batch_size=args.batch_size
        )
    else:
        metrics, predictions, labels = evaluate_test_set(
            model=model,
            tokenizer=tokenizer,
            test_file=args.test_file,
            drug_to_idx=drug_to_idx,
            num_drugs=num_drugs,
            device=device,
            max_seq_length=args.max_seq_length,
            batch_size=args.batch_size
        )
    
    print("\n" + "="*50)
    print("测试集评估结果:")
    print("="*50)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    # 保存详细结果
    results_file = join(args.model_path, "test_evaluation_results.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    print(f"\n详细结果已保存至: {results_file}")

if __name__ == "__main__":
    main()