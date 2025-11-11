import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    probs = 1 / (1 + np.exp(-predictions))
    metrics = {}
    thresholds = [0.3, 0.5, 0.7]
    preds_05 = (probs > 0.5).astype(int)
    
    for threshold in thresholds:
        preds = (probs > threshold).astype(int)
        jaccard_scores = []
        precision_scores = []
        recall_scores = [] 
        f1_scores = []
        max_samples = min(1000, len(labels))
        
        for i in range(max_samples):
            y_true = labels[i]
            y_pred = preds[i]
            
            true_set = set(np.where(y_true == 1)[0])
            pred_set = set(np.where(y_pred == 1)[0])
            
            # Jaccard
            intersection = len(true_set & pred_set)
            union = len(true_set | pred_set)
            jaccard = intersection / union if union > 0 else 0.0
            jaccard_scores.append(jaccard)
            
            # Precision
            precision = intersection / len(pred_set) if len(pred_set) > 0 else 0.0
            precision_scores.append(precision)
            
            # Recall
            recall = intersection / len(true_set) if len(true_set) > 0 else 0.0
            recall_scores.append(recall)
            
            # F1
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            f1_scores.append(f1)
        
        avg_jaccard = np.mean(jaccard_scores)
        avg_precision = np.mean(precision_scores)
        avg_recall = np.mean(recall_scores)
        avg_f1 = np.mean(f1_scores)
        score = 0.5 * (avg_jaccard + avg_f1)

        metrics.update({
            f"jaccard_th{threshold}": avg_jaccard,
            f"precision_avg_th{threshold}": avg_precision,
            f"recall_avg_th{threshold}": avg_recall,
            f"f1_avg_th{threshold}": avg_f1,
            f"score_th{threshold}": score,
        })
    
    if len(labels) > 0:
        metrics.update({
            "f1_micro": f1_score(labels[:max_samples], preds_05[:max_samples], average="micro", zero_division=0),
            "precision_micro": precision_score(labels[:max_samples], preds_05[:max_samples], average="micro", zero_division=0),
            "recall_micro": recall_score(labels[:max_samples], preds_05[:max_samples], average="micro", zero_division=0),
        })
    
    metrics["avg_labels_per_sample"] = labels.sum(axis=1).mean() if len(labels) > 0 else 0
    metrics["avg_preds_per_sample"] = preds_05.sum(axis=1).mean() if len(labels) > 0 else 0
    
    exact_match = np.all(preds_05 == labels, axis=1)
    metrics["exact_match_accuracy"] = exact_match.mean() if len(labels) > 0 else 0

    del predictions, labels, probs, preds_05
    import gc
    gc.collect()
    
    return metrics

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