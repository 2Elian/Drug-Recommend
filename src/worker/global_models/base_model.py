import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import AutoModel, AutoConfig, PreTrainedModel


class DrugBaseModel(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

    def compute_metrics(self, predictions, labels, thresholds=[0.3, 0.4, 0.5]):
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
        
        return metrics


@dataclass
class DrugRecommendOutput(SequenceClassifierOutput):
    metrics: Optional[Dict[str, Any]] = None


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.75, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        # logits: [B, num_classes]
        probs = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = probs * targets + (1 - probs) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)
        
        if self.alpha is not None:
            alpha_factor = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_factor * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss