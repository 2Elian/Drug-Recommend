import argparse, json, sys
from typing import Dict, List, Tuple, Set
import json

def load_gt_jsonl(path: str, id_field: str, label_field: str, lower: bool) -> Dict[str, Set[str]]:
    gt = {}
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            if not line.strip(): continue
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
            if lower: items = [x.lower() for x in items]
            gt[_id] = set(items)
    return gt

def load_pred(path: str, id_field: str, label_field: str, lower: bool) -> Dict[str, List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    pred = {}
    if isinstance(data, dict):
        # 形如 {ID: [..], ...}
        for _id, lst in data.items():
            items = [str(x).strip() for x in (lst or []) if str(x).strip()!=""]
            if lower: items = [x.lower() for x in items]
            pred[str(_id)] = items
    elif isinstance(data, list):
        # 形如 [{"ID":..,"prediction":[..]}, ...]
        for i, obj in enumerate(data, 1):
            if id_field not in obj:
                raise KeyError(f"[PRED] 第{i}个样本缺少字段: {id_field}")
            if label_field not in obj:
                raise KeyError(f"[PRED] 第{i}个样本缺少字段: {label_field}")
            _id = str(obj[id_field])
            lst = obj[label_field] or []
            if isinstance(lst, str):
                lst = [lst]
            if not isinstance(lst, list):
                raise TypeError(f"[PRED] 第{i}个样本的 {label_field} 不是list/str")
            items = [str(x).strip() for x in lst if str(x).strip()!=""]
            if lower: items = [x.lower() for x in items]
            pred[_id] = items
    else:
        raise TypeError("[PRED] 不支持的JSON结构：应为对象或数组")

    return pred

def jaccard(p: Set[str], y: Set[str]) -> float:
    if not p and not y: return 1.0
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
    if P + R == 0: return 0.0
    return 2 * P * R / (P + R)

def evaluate(gt: Dict[str, Set[str]], pred: Dict[str, List[str]]) -> Tuple[float,float,float,float,float]:
    ids = list(gt.keys())
    miss = [i for i in ids if i not in pred]
    extra = [i for i in pred.keys() if i not in gt]
    if miss:
        print(f"[警告] 预测缺少 {len(miss)} 个ID（将按空预测处理）示例: {miss[:5]}", file=sys.stderr)
    if extra:
        print(f"[提示] 预测包含 {len(extra)} 个不在GT中的ID（将忽略）示例: {extra[:5]}", file=sys.stderr)

    J_list, P_list, R_list, F1_list = [], [], [], []
    for _id, y in gt.items():
        p = set(pred.get(_id, []))  # 缺失按空预测
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

def compute(pred_file, gt_file):
    gt_id_field="就诊标识"
    gt_label_field="出院带药列表"
    pred_id_field="ID"
    pred_label_field="prediction"
    lower = False
    gt = load_gt_jsonl(gt_file, gt_id_field, gt_label_field, lower)
    pred = load_pred(pred_file, pred_id_field, pred_label_field, lower)
    AVG_J, AVG_P, AVG_R, AVG_F1, SCORE = evaluate(gt, pred)
    print(f"AVG_Jaccard: {AVG_J:.6f}")
    print(f"AVG_Precision: {AVG_P:.6f}")
    print(f"AVG_Recall: {AVG_R:.6f}")
    print(f"AVG_F1: {AVG_F1:.6f}")
    print(f"Final Score = 0.5 * (Jaccard + F1) = {SCORE:.6f}")
