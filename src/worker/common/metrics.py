import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
import torch

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    if isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu()
    else:
        logits = torch.tensor(logits)

    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu()
    else:
        labels = torch.tensor(labels)

    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).int()
    labels = labels.int()

    intersection = (labels & preds).sum(dim=1)
    union = (labels | preds).sum(dim=1)
    jaccard = torch.where(union != 0, intersection / union, torch.zeros_like(intersection, dtype=torch.float))
    avg_jaccard = jaccard.mean().item()
    true_positives = labels.sum(dim=1)
    pred_positives = preds.sum(dim=1)
    precision = torch.where(pred_positives != 0, intersection / pred_positives, torch.zeros_like(intersection, dtype=torch.float))
    recall = torch.where(true_positives != 0, intersection / true_positives, torch.zeros_like(intersection, dtype=torch.float))
    f1 = torch.where((precision + recall) != 0, 2 * precision * recall / (precision + recall), torch.zeros_like(precision))
    avg_f1 = f1.mean().item()

    final_score = 0.5 * (avg_f1 + avg_jaccard)

    return {
        "avg_f1": avg_f1,
        "avg_jaccard": avg_jaccard,
        "final_score": final_score
    }

def compute_metrics_fast(predictions, labels, thresholds=[0.3, 0.5, 0.7]):
    metrics = {}
    labels = labels.astype(int)
    for threshold in thresholds:
        preds = (predictions > threshold).astype(int)
        intersection = np.sum(labels & preds, axis=1)
        union = np.sum(labels | preds, axis=1)
        true_positives = np.sum(labels, axis=1)
        pred_positives = np.sum(preds, axis=1)
        jaccard = np.divide(intersection, union, out=np.zeros_like(intersection, dtype=float), where=union!=0)
        precision = np.divide(intersection, pred_positives, out=np.zeros_like(intersection, dtype=float), where=pred_positives!=0)
        recall = np.divide(intersection, true_positives, out=np.zeros_like(intersection, dtype=float), where=true_positives!=0)
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
        
        metrics.update({
            f"f1_micro_th{threshold}": f1_score(labels, preds, average="micro", zero_division=0),
            f"precision_micro_th{threshold}": precision_score(labels, preds, average="micro", zero_division=0),
            f"recall_micro_th{threshold}": recall_score(labels, preds, average="micro", zero_division=0),
        })

    preds_05 = (predictions > 0.5).astype(int)
    metrics.update({
        "avg_labels_per_sample": np.mean(np.sum(labels, axis=1)),
        "avg_preds_per_sample": np.mean(np.sum(preds_05, axis=1)),
        "exact_match_accuracy": np.mean(np.all(preds_05 == labels, axis=1)),
        "total_samples": len(labels)
    })
    
    return metrics


def compute_simple_metrics(predictions, labels, threshold=0.5):
    """只计算F1和Jaccard指标"""
    labels = labels.astype(int)
    preds = (predictions > threshold).astype(int)

    intersection = np.sum(labels & preds, axis=1)
    union = np.sum(labels | preds, axis=1)
    jaccard = np.divide(intersection, union, out=np.zeros_like(intersection, dtype=float), where=union!=0)
    avg_jaccard = np.mean(jaccard)

    true_positives = np.sum(labels, axis=1)
    pred_positives = np.sum(preds, axis=1)
    precision = np.divide(intersection, pred_positives, out=np.zeros_like(intersection, dtype=float), where=pred_positives!=0)
    recall = np.divide(intersection, true_positives, out=np.zeros_like(intersection, dtype=float), where=true_positives!=0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(precision), where=(precision + recall)!=0)
    avg_f1 = np.mean(f1)
    
    return avg_f1, avg_jaccard