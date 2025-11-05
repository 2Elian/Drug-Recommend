import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from transformers.modeling_outputs import SequenceClassifierOutput

@dataclass
class DrugRecommendOutput(SequenceClassifierOutput):
    probs: Optional[torch.FloatTensor] = None
    predictions: Optional[torch.BoolTensor] = None
    threshold: Optional[float] = None
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