import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig, PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput
from typing import Optional, List

from .base_model import FocalLoss, DrugRecommendOutput, DrugRecommendEvalOutput, DrugBaseModel
from src.utils.log import get_logger

class BaselineModel(DrugBaseModel):
    def __init__(
        self,
        args,
        model_name_or_path: str,
        d_type: str = "bfloat16",
        num_labels: int = 600,
        use_focal_loss: bool = True,
        focal_alpha: float = 0.75,
        focal_gamma: float = 2.0,
        pos_weight: Optional[torch.Tensor] = None,
        use_metrics: bool = False,
        is_train: bool = False
    ):
        self.drug_model_logger = get_logger(name="DrugBaselineModel")
        self.trust_remote_code = True
        glm_config = AutoConfig.from_pretrained(
            model_name_or_path,
            trust_remote_code=self.trust_remote_code
        )
        glm_config.num_labels = num_labels
        super().__init__(
            config=glm_config,
            model_name=model_name_or_path,
            lora_r=args.lora_rank,                      
            lora_alpha=args.lora_alpha,
            dtype=d_type,
            lora_dropout=args.lora_dropout,
            train_mode=args.train_mode
        )
        self.args = args
        self.ignore_index = self.args.ignore_index
        self.r1 = self.args.r1
        self.r2 = self.args.r2
        if args.lm_loss:
            self.lm_loss_fct = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        else:
            if hasattr(self.backbone, 'output_layer'):
                if isinstance(self.backbone.lm_head, torch.nn.Module):
                    self.backbone.output_layer.to('cpu')
                del self.backbone.output_layer
                if hasattr(self.backbone, 'output_layer'):
                    delattr(self.backbone, 'output_layer')
        self.num_labels = num_labels
        self.use_metrics = use_metrics
        self.drug_classifier  = nn.Linear(self.hidden_size, self.num_labels)
        self.dropout = nn.Dropout(0.1)
        self.use_focal_loss = use_focal_loss
        if use_focal_loss:
            self.loss_fn = FocalLoss(gamma=focal_gamma, alpha=focal_alpha)
        else:
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        lm_labels=None,
        **kwargs
    ):
        if input_ids is not None:
            input_ids = input_ids.to(self.backbone.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.backbone.device)
        if labels is not None:
            labels = labels.to(self.backbone.device)
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True, return_dict=True
        )
        lm_loss = None
        if self.args.lm_loss:
            assert lm_labels is not None, "lm_loss is not None, but found lm_labels is None"
            lm_logits = outputs.logits
            lm_loss = self.lm_loss_fct(lm_logits.view(-1, lm_logits.size(-1)).float(), lm_labels.view(-1))
        # multi-cls for durg 
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            hidden_states = outputs.hidden_states[-1]  # [B, L, H]
        elif hasattr(outputs, 'last_hidden_state'):
            hidden_states = outputs.last_hidden_state
        else:
            raise ValueError
        # take the last non-pad token
        pad_token_id = getattr(self.backbone.config, "pad_token_id", None)
        if pad_token_id is not None: # 151329 is pad_token_id for glm4-chat
            non_pad_mask = (input_ids != pad_token_id).int()
            token_idx = torch.arange(input_ids.shape[-1], device=input_ids.device)
            last_token = (token_idx * non_pad_mask).argmax(-1)
        else:
            raise ValueError
        pooled_output = hidden_states[torch.arange(hidden_states.size(0)), last_token]
        
        pooled_output = self.dropout(pooled_output)
        logits = self.drug_classifier(pooled_output)  # [batch_size, num_labels]
        all_loss = None
        cls_loss = None
        if labels is not None:
            if self.use_focal_loss:
                cls_loss = self.loss_fn(logits, labels.float()) # FocalLoss
            else:
                cls_loss = self.loss_fn(logits, labels.float()) # bce loss
        if self.args.lm_loss: all_loss += self.r1 * lm_loss
        if labels is not None: all_loss += self.r2 * cls_loss

        return DrugRecommendOutput(
            loss=all_loss,
            lm_loss=lm_loss,
            cls_loss=cls_loss
        )
    
    @torch.no_grad()
    def drug_eval(self, input_ids=None, attention_mask=None, labels=None):
        if input_ids is not None:
            input_ids = input_ids.to(self.backbone.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.backbone.device)
        if labels is not None:
            labels = labels.to(self.backbone.device)
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True, return_dict=True
        )
        # multi-cls for durg 
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            hidden_states = outputs.hidden_states[-1]  # [B, L, H]
        elif hasattr(outputs, 'last_hidden_state'):
            hidden_states = outputs.last_hidden_state
        else:
            raise ValueError
        # take the last non-pad token
        pad_token_id = getattr(self.backbone.config, "pad_token_id", None)
        if pad_token_id is not None: # 151329 is pad_token_id for glm4-chat
            non_pad_mask = (input_ids != pad_token_id).int()
            token_idx = torch.arange(input_ids.shape[-1], device=input_ids.device)
            last_token = (token_idx * non_pad_mask).argmax(-1)
        else:
            raise ValueError
        pooled_output = hidden_states[torch.arange(hidden_states.size(0)), last_token]
        
        pooled_output = self.dropout(pooled_output)
        logits = self.drug_classifier(pooled_output)  # [batch_size, num_labels]
        cls_loss=None
        if labels is not None:
            if self.use_focal_loss:
                cls_loss = self.loss_fn(logits, labels.float()) # FocalLoss
            else:
                cls_loss = self.loss_fn(logits, labels.float()) # bce loss
        return DrugRecommendEvalOutput(
            cls_loss=cls_loss,
            logits=logits
        )
    def print_trainable_parameters(self):
        total_params = 0
        trainable_params = 0
        for name, param in self.named_parameters():
            num_params = param.numel()
            total_params += num_params
            if param.requires_grad:
                trainable_params += num_params
                self.drug_model_logger.info(f"[Trainable] {name}: {num_params} params")
            else:
                self.drug_model_logger.info(f"[Frozen]    {name}: {num_params} params")
        
        self.drug_model_logger.info("="*100)
        self.drug_model_logger.info(f"Total parameters: {total_params}")
        self.drug_model_logger.info(f"Trainable parameters: {trainable_params}")
        self.drug_model_logger.info(f"Trainable ratio: {100 * trainable_params / total_params:.2f}%")
        self.drug_model_logger.info("="*100)