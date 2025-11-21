import numpy as np
import json
from sklearn.metrics import f1_score, precision_score, recall_score
import torch
from typing import Optional
from transformers import PreTrainedTokenizer
with open('/data/lzm/DrugRecommend/src/data/CDrugRed-A-v1/候选药物列表.json', 'r', encoding='utf-8') as f:
        ORIGINAL_DRUG_LIST = json.load(f)
VALID_DRUGS_SET = set()
for drug in ORIGINAL_DRUG_LIST:
    standardized_drug_name = drug.replace('，', '').replace('、', '').replace(' ', '')
    if standardized_drug_name:
        VALID_DRUGS_SET.add(standardized_drug_name)
GLOBAL_VALID_DRUGS_SET = VALID_DRUGS_SET
GLOBAL_TOKENIZER: Optional[PreTrainedTokenizer] = None

def genercommend_compute_metrics(eval_preds):
    if GLOBAL_TOKENIZER is None:
        raise ValueError("GLOBAL_TOKENIZER 必须在 compute_metrics 外部被初始化！")

    predictions, label_ids = eval_preds
    # 移除loss ignore index 和 pad token
    true_labels_tokens = np.where(label_ids != -100, label_ids, GLOBAL_TOKENIZER.pad_token_id)
    predicted_texts = GLOBAL_TOKENIZER.batch_decode(predictions, skip_special_tokens=True)
    true_texts = GLOBAL_TOKENIZER.batch_decode(true_labels_tokens, skip_special_tokens=True)
    def extract_drugs(text_list, valid_drugs_set: set = None, filter: bool = False):
        drug_sets = []      
        for text in text_list:
            if not text:
                drug_sets.append(set())
                continue
            normalized_text = text.replace('，', ',').replace('、', ',')
            no_space_text = normalized_text.replace(' ', '')
            drug_items = [item for item in no_space_text.split(',') if item]
            current_drug_set = set(drug_items)
            if filter:
                current_drug_set = current_drug_set.intersection(valid_drugs_set)
            drug_sets.append(current_drug_set)
                
        return drug_sets
    pred_drug_sets = extract_drugs(predicted_texts, GLOBAL_VALID_DRUGS_SET, True)
    true_drug_sets = extract_drugs(true_texts)
    all_precision, all_recall, all_f1, all_jaccard = [], [], [], []
    
    for P, T in zip(pred_drug_sets, true_drug_sets):
        if not T:
            continue

        intersection = P.intersection(T)
        union = P.union(T)
        precision = len(intersection) / len(P) if len(P) > 0 else 0
        recall = len(intersection) / len(T) if len(T) > 0 else 0
        
        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        jaccard = len(intersection) / len(union) if len(union) > 0 else 0
        
        all_precision.append(precision)
        all_recall.append(recall)
        all_f1.append(f1)
        all_jaccard.append(jaccard)
    avg_f1 = np.mean(all_f1)
    avf_jaccard = np.mean(all_jaccard)
    
    return {
        "final_score": 0.5 * (avg_f1+avf_jaccard),
        "avg_f1": avg_f1,
        "avg_jaccard": avf_jaccard
    }
