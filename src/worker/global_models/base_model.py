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
    # TODO: 基于Xtuner-v1集成 TP SP 和 loss相关操作

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

@dataclass
class DrugBertOutput:
    cls_loss: Optional[torch.FloatTensor] = None
    ndf_loss: Optional[torch.FloatTensor] = None
    cmm_loss: Optional[torch.FloatTensor] = None
    mlm_loss: Optional[torch.FloatTensor] = None

@dataclass
class DrugBertEvalOutput:
    logits_mode_1: Optional[torch.FloatTensor] = None
    logits_mode_2: Optional[torch.FloatTensor] = None
    logits_mode_fusion: Optional[torch.FloatTensor] = None


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
    
def reduce_loss(loss, reduction):
    if reduction == 'none':
        return loss
    elif reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        raise ValueError(f"Unknown reduction: {reduction}")

def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    if weight is not None:
        loss = loss * weight
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        if reduction == 'mean':
            loss = loss.sum() / avg_factor
        elif reduction == 'none':
            pass
        else:
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss

class ResampleLoss(nn.Module):
    def __init__(self,
                 use_sigmoid=True, partial=False,
                 loss_weight=1.0, reduction='mean',
                 reweight_func=None,  # None, 'inv', 'sqrt_inv', 'rebalance', 'CB'
                 weight_norm=None, # None, 'by_instance', 'by_batch'
                 focal=None,
                 map_param=None,
                 CB_loss=None,
                 logit_reg=None,
                 class_freq=None,
                 train_num=None,
                 eps=1e-8):
        super().__init__()
        self.use_sigmoid = use_sigmoid
        self.partial = partial
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.reweight_func = reweight_func
        self.weight_norm = weight_norm

        # focal defaults
        focal = focal or {'focal': True, 'alpha': 0.5, 'gamma': 2.0}
        self.focal = focal.get('focal', True)
        self.alpha = float(focal.get('alpha', 0.5))
        self.gamma = float(focal.get('gamma', 2.0))

        map_param = map_param or {'alpha': 10.0, 'beta': 0.2, 'gamma': 0.1}
        self.map_alpha = float(map_param.get('alpha', 10.0))
        self.map_beta = float(map_param.get('beta', 0.2))
        self.map_gamma = float(map_param.get('gamma', 0.1))

        CB_loss = CB_loss or {'CB_beta': 0.9, 'CB_mode': 'average_w'}
        self.CB_beta = float(CB_loss.get('CB_beta', 0.9))
        self.CB_mode = CB_loss.get('CB_mode', 'average_w')

        logit_reg = logit_reg or {}
        self.logit_reg = bool(logit_reg)
        self.neg_scale = float(logit_reg.get('neg_scale', 1.0))
        self.init_bias_coef = float(logit_reg.get('init_bias', 0.0))

        # class frequency: keep as numpy or tensor but do NOT cuda() in init
        if class_freq is None:
            self.class_freq = None
            self.num_classes = None
        else:
            cf = np.asarray(class_freq, dtype=np.float64)
            cf = np.maximum(cf, eps)   # avoid zeros
            self.class_freq = torch.tensor(cf, dtype=torch.float32)  # device-agnostic
            self.num_classes = self.class_freq.numel()

        self.train_num = float(train_num) if train_num is not None else None
        self.eps = eps

        # precompute inverses safely if available
        if self.class_freq is not None and self.train_num is not None:
            # clamp to avoid extremely large ratios
            cf_safe = self.class_freq.clone()
            self.freq_inv = (1.0 / cf_safe)
            self.propotion_inv = (self.train_num / cf_safe).float()
            # init_bias computed lazily on device in forward for device-safety
        else:
            self.freq_inv = None
            self.propotion_inv = None

    def forward(self, cls_score, label, weight=None, avg_factor=None, reduction_override=None):
        """
        cls_score: logits tensor shape [B, C]
        label: same shape or broadcastable (B, C), values {0,1} (float or long)
        """
        reduction = reduction_override if reduction_override is not None else self.reduction
        device = cls_score.device
        # prepare labels
        if not torch.is_tensor(label):
            label = torch.tensor(label, device=device, dtype=torch.float32)
        else:
            label = label.to(device).float()

        # ensure shape [B, C]
        if label.dim() == 1 and cls_score.dim() == 2:
            # assume label was single-index? but we expect multilabel; keep as is
            label = label.unsqueeze(1).to(device)

        # compute weight from reweight function (on device)
        sample_weight = None
        if self.reweight_func is not None:
            sample_weight = self.reweight_functions(label, device=device)

        # apply logit regularization (on a copy)
        logits = cls_score
        logits, sample_weight = self.logit_reg_functions(label, logits, sample_weight)

        # compute base elementwise loss (binary cross entropy with logits)
        # use reduction='none' so we can apply focal / weighting
        # stable bce with logits:
        bce = F.binary_cross_entropy_with_logits(logits, label, reduction='none')

        # focal modulation
        if self.focal:
            # compute pt = p for positive, 1-p for negative
            prob = torch.sigmoid(logits)
            pt = prob * label + (1 - prob) * (1 - label)  # shape [B,C]
            # avoid pt = 0
            pt = torch.clamp(pt, min=self.eps, max=1.0)
            alpha_t = label * self.alpha + (1.0 - label) * (1.0 - self.alpha)
            focal_factor = alpha_t * ((1.0 - pt) ** self.gamma)
            loss = focal_factor * bce
        else:
            loss = bce

        # apply reweight if available (sample_weight shape should match [B,C] or be broadcastable)
        if sample_weight is not None:
            # ensure same device
            sample_weight = sample_weight.to(device)
        # combine user-supplied 'weight' too
        if weight is not None:
            # move to device and broadcast
            weight = weight.to(device)
            if sample_weight is None:
                sample_weight = weight
            else:
                sample_weight = sample_weight * weight

        loss = weight_reduce_loss(loss, weight=sample_weight, reduction=reduction, avg_factor=avg_factor)
        loss = self.loss_weight * loss
        return loss

    # ---------------- helper functions ----------------
    def logit_reg_functions(self, labels, logits, weight=None):
        """
        Apply init_bias and neg_scale in a numerically stable & broadcast-friendly way.
        We compute init_bias on the same device as logits.
        """
        device = logits.device
        out_logits = logits
        out_weight = weight
        if not self.logit_reg:
            return out_logits, out_weight

        # compute init_bias if coefficient provided and class_freq/train_num are available
        if self.init_bias_coef != 0.0 and (self.class_freq is not None and self.train_num is not None):
            cf = self.class_freq.to(device)
            # ratio = train_num / freq
            ratio = torch.clamp(self.train_num / (cf + self.eps), min=1.0 + 1e-6)
            init_bias = - torch.log(ratio - 1.0) * self.init_bias_coef
            # broadcast to [1, C] if necessary
            init_bias = init_bias.view(1, -1)
            out_logits = out_logits + init_bias

        # negative scaling: multiply logits for negative examples by neg_scale
        if self.neg_scale is not None and self.neg_scale != 1.0:
            neg_scale = float(self.neg_scale)
            # use where to only scale negatives
            out_logits = out_logits * torch.where(labels == 1.0, torch.ones_like(labels, device=out_logits.device), torch.tensor(neg_scale, device=out_logits.device))
            if out_weight is not None:
                out_weight = out_weight * torch.where(labels == 1.0, torch.ones_like(labels, device=out_logits.device), 1.0 / neg_scale)
        return out_logits, out_weight

    def reweight_functions(self, gt_labels, device=None):
        """Dispatch reweight function and return weight [B, C] or None."""
        if self.reweight_func is None:
            return None
        if device is None:
            device = gt_labels.device

        if self.reweight_func in ('inv', 'sqrt_inv'):
            return self.RW_weight(gt_labels, device=device)
        elif self.reweight_func == 'rebalance':
            return self.rebalance_weight(gt_labels, device=device)
        elif self.reweight_func == 'CB':
            return self.CB_weight(gt_labels, device=device)
        else:
            return None

    def RW_weight(self, gt_labels, device=None, by_class=True):
        """inverse or sqrt inverse weighting; returns shape [B, C]"""
        device = device or gt_labels.device
        if self.propotion_inv is None:
            return None
        w = self.propotion_inv.to(device)
        if 'sqrt' in (self.reweight_func or ''):
            w = torch.sqrt(w)
        # w is [C], expand to [B, C]
        w = w.view(1, -1).expand(gt_labels.shape[0], -1)
        if not by_class:
            # per-instance normalization
            sum_ = (w * gt_labels).sum(dim=1, keepdim=True) + self.eps
            w = sum_ / (gt_labels.sum(dim=1, keepdim=True) + self.eps)
        return w

    def rebalance_weight(self, gt_labels, device=None):
        """
        Implementation of 'rebalance' mapping:
        pos_weight = freq_inv / repeat_rate, then map via sigmoid(map_beta*(pos_weight - map_gamma)) + map_alpha
        """
        device = device or gt_labels.device
        if self.freq_inv is None:
            return None
        freq_inv = self.freq_inv.to(device).view(1, -1)  # [1, C]
        repeat_rate = (gt_labels.float() * freq_inv).sum(dim=1, keepdim=True) + self.eps  # [B,1]
        pos_weight = freq_inv / repeat_rate  # [B, C]
        mapped = torch.sigmoid(self.map_beta * (pos_weight - self.map_gamma)) + self.map_alpha
        return mapped

    def CB_weight(self, gt_labels, device=None):
        device = device or gt_labels.device
        if self.class_freq is None:
            return None
        cf = self.class_freq.to(device).view(1, -1)  # [1, C]
        beta = float(self.CB_beta)
        mode = self.CB_mode
        if mode == 'by_class':
            w = (1 - beta) / (1 - torch.pow(beta, cf + self.eps))
            return w.expand(gt_labels.shape[0], -1)
        elif mode == 'average_n':
            avg_n = (gt_labels * cf).sum(dim=1, keepdim=True) / (gt_labels.sum(dim=1, keepdim=True) + self.eps)
            w = (1 - beta) / (1 - torch.pow(beta, avg_n + self.eps))
            return w
        elif mode == 'average_w':
            w_ = (1 - beta) / (1 - torch.pow(beta, cf + self.eps))  # [1, C]
            w = (gt_labels * w_).sum(dim=1, keepdim=True) / (gt_labels.sum(dim=1, keepdim=True) + self.eps)
            # expand to [B, C] (per-instance scalar)
            return w.expand(gt_labels.shape[0], cf.shape[1])
        elif mode == 'min_n':
            # handle zero labels
            big = 1e6
            masked = gt_labels * cf + (1 - gt_labels) * big
            min_n, _ = masked.min(dim=1, keepdim=True)
            w = (1 - beta) / (1 - torch.pow(beta, min_n + self.eps))
            return w.expand(gt_labels.shape[0], cf.shape[1])
        else:
            raise ValueError("Unknown CB mode: " + str(mode))