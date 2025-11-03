#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/11/1 20:33
# @Author  : lizimo@nuist.edu.cn
# @File    : eval.py
# @Description: Thanks to LastWh1sper for contributing the evaluation code. https://tianchi.aliyun.com/forum/post/937062
import argparse, json, sys
from typing import Dict, List, Tuple, Set

def load_gt_jsonl(path: str, id_field: str, label_field: str, lower: bool, max_samples: int = -1) -> Dict[str, Set[str]]:
    gt = {}
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            if not line.strip(): 
                continue
            if max_samples > 0 and len(gt) >= max_samples:
                break
                
            obj = json.loads(line)
            if id_field not in obj:
                raise KeyError(f"[GT] 行{ln}缺少字段: {id_field}")
            if label_field not in obj:
                raise KeyError(f"[GT] 行{ln}缺少字段: {label_field}")
            _id = str(obj[id_field])
            labels = obj[label_field]
            if isinstance(labels, str):
                labels = [labels]
            if not isinstance(labels, list):
                raise TypeError(f"[GT] 行{ln}的 {label_field} 不是list/str")
            items = [str(x).strip() for x in labels if str(x).strip()!=""]
            if lower: 
                items = [x.lower() for x in items]
            gt[_id] = set(items)
    return gt

def load_pred(path: str, id_field: str, label_field: str, lower: bool, max_samples: int = -1) -> Dict[str, List[str]]:
    """加载预测文件，支持 JSON 和 JSONL 格式"""
    pred = {}
    sample_count = 0
    
    # 首先尝试作为 JSONL 文件读取
    try:
        with open(path, "r", encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                if not line.strip():
                    continue
                if max_samples > 0 and sample_count >= max_samples:
                    break
                    
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"[警告] 预测文件第{ln}行JSON解析错误: {e}", file=sys.stderr)
                    continue
                
                # 处理 JSONL 格式
                if id_field not in obj:
                    print(f"[警告] 预测文件第{ln}行缺少字段: {id_field}，跳过该行", file=sys.stderr)
                    continue
                if label_field not in obj:
                    print(f"[警告] 预测文件第{ln}行缺少字段: {label_field}，跳过该行", file=sys.stderr)
                    continue
                
                _id = str(obj[id_field])
                lst = obj[label_field] or []
                if isinstance(lst, str):
                    lst = [lst]
                if not isinstance(lst, list):
                    print(f"[警告] 预测文件第{ln}行的 {label_field} 不是list/str，跳过该行", file=sys.stderr)
                    continue
                
                items = [str(x).strip() for x in lst if str(x).strip()!=""]
                if lower: 
                    items = [x.lower() for x in items]
                pred[_id] = items
                sample_count += 1
                
        return pred
        
    except (UnicodeDecodeError, FileNotFoundError) as e:
        print(f"[警告] JSONL读取失败，尝试作为普通JSON读取: {e}", file=sys.stderr)
    
    # 作为普通 JSON 文件读取
    with open(path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"预测文件既不是有效的JSON也不是JSONL格式: {e}")

    if isinstance(data, dict):
        # 形如 {ID: [..], ...}
        for _id, lst in data.items():
            if max_samples > 0 and sample_count >= max_samples:
                break
            items = [str(x).strip() for x in (lst or []) if str(x).strip()!=""]
            if lower: 
                items = [x.lower() for x in items]
            pred[str(_id)] = items
            sample_count += 1
    elif isinstance(data, list):
        # 形如 [{"ID":..,"prediction":[..]}, ...]
        for i, obj in enumerate(data, 1):
            if max_samples > 0 and sample_count >= max_samples:
                break
            if id_field not in obj:
                print(f"[警告] 预测文件第{i}个样本缺少字段: {id_field}，跳过该样本", file=sys.stderr)
                continue
            if label_field not in obj:
                print(f"[警告] 预测文件第{i}个样本缺少字段: {label_field}，跳过该样本", file=sys.stderr)
                continue
            _id = str(obj[id_field])
            lst = obj[label_field] or []
            if isinstance(lst, str):
                lst = [lst]
            if not isinstance(lst, list):
                print(f"[警告] 预测文件第{i}个样本的 {label_field} 不是list/str，跳过该样本", file=sys.stderr)
                continue
            items = [str(x).strip() for x in lst if str(x).strip()!=""]
            if lower: 
                items = [x.lower() for x in items]
            pred[_id] = items
            sample_count += 1
    else:
        raise TypeError("[PRED] 不支持的JSON结构：应为对象或数组")

    return pred

def jaccard(p: Set[str], y: Set[str]) -> float:
    if not p and not y: 
        return 1.0
    return len(p & y) / max(1, len(p | y))

def precision(p: Set[str], y: Set[str]) -> float:
    if not p:
        return 1.0 if not y else 0.0
    return len(p & y) / len(p)

def recall(p: Set[str], y: Set[str]) -> float:
    if not y:
        return 1.0 if not p else 0.0
    return len(p & y) / len(y)

def f1(p: Set[str], y: Set[str]) -> float:
    P = precision(p, y)
    R = recall(p, y)
    if P + R == 0: 
        return 0.0
    return 2 * P * R / (P + R)

def evaluate(gt: Dict[str, Set[str]], pred: Dict[str, List[str]]) -> Tuple[float,float,float,float,float]:
    # 确保只评估两个文件中都存在的ID
    common_ids = set(gt.keys()) & set(pred.keys())
    gt_only_ids = set(gt.keys()) - set(pred.keys())
    pred_only_ids = set(pred.keys()) - set(gt.keys())
    
    if gt_only_ids:
        print(f"[警告] 预测缺少 {len(gt_only_ids)} 个GT中的ID（将忽略）示例: {list(gt_only_ids)[:5]}", file=sys.stderr)
    if pred_only_ids:
        print(f"[提示] 预测包含 {len(pred_only_ids)} 个不在GT中的ID（将忽略）示例: {list(pred_only_ids)[:5]}", file=sys.stderr)
    
    if not common_ids:
        raise ValueError("没有找到匹配的ID，无法进行评估！请检查ID字段是否一致")

    J_list, P_list, R_list, F1_list = [], [], [], []
    for _id in common_ids:
        y = gt[_id]  # GT标签
        p = set(pred[_id])  # 预测标签
        J_list.append(jaccard(p, y))
        P_list.append(precision(p, y))
        R_list.append(recall(p, y))
        F1_list.append(f1(p, y))

    AVG_J = sum(J_list)/len(J_list)
    AVG_P = sum(P_list)/len(P_list)
    AVG_R = sum(R_list)/len(R_list)
    AVG_F1 = sum(F1_list)/len(F1_list)
    SCORE = 0.5 * (AVG_J + AVG_F1)
    
    return AVG_J, AVG_P, AVG_R, AVG_F1, SCORE

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default="/data/lzm/DrugRecommend/src/data/CDrugRed-A-v1/CDrugRed_val_split_train_18present.jsonl", help="GT jsonl（每行一个样本）")
    ap.add_argument("--pred", default="/data/lzm/DrugRecommend/src/data/CDrugRed-A-v1/CDrugRed_predict-val.jsonl", help="预测 json（list或dict）")
    ap.add_argument("--gt-id-field", default="就诊标识")
    ap.add_argument("--gt-label-field", default="出院带药列表")
    ap.add_argument("--pred-id-field", default="就诊标识")  # 修改为与GT相同的字段名
    ap.add_argument("--pred-label-field", default="出院带药列表")  # 修改为与GT相同的字段名
    ap.add_argument("--case-sensitive", action="store_true", help="区分大小写；默认不区分")
    ap.add_argument("--num-samples", type=int, default=-1, help="评估样本数量，-1代表全部样本")
    
    args = ap.parse_args()

    lower = not args.case_sensitive
    gt = load_gt_jsonl(args.gt, args.gt_id_field, args.gt_label_field, lower, args.num_samples)
    pred = load_pred(args.pred, args.pred_id_field, args.pred_label_field, lower, args.num_samples)
    
    print(f"评估配置:", file=sys.stderr)
    print(f"  GT文件: {args.gt}", file=sys.stderr)
    print(f"  预测文件: {args.pred}", file=sys.stderr)
    print(f"  GT样本数: {len(gt)}", file=sys.stderr)
    print(f"  预测样本数: {len(pred)}", file=sys.stderr)
    print(f"  共同样本数: {len(set(gt.keys()) & set(pred.keys()))}", file=sys.stderr)
    print(f"  大小写敏感: {args.case_sensitive}", file=sys.stderr)
    print("", file=sys.stderr)

    AVG_J, AVG_P, AVG_R, AVG_F1, SCORE = evaluate(gt, pred)
    print(f"AVG_Jaccard: {AVG_J:.6f}")
    print(f"AVG_Precision: {AVG_P:.6f}")
    print(f"AVG_Recall: {AVG_R:.6f}")
    print(f"AVG_F1: {AVG_F1:.6f}")
    print(f"Final Score = 0.5 * (Jaccard + F1) = {SCORE:.6f}")

if __name__ == "__main__":
    main()