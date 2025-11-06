import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig, PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput
from typing import Optional, List
from .base_model import FocalLoss, DrugRecommendOutput, DrugBaseModel, ResampleLoss

class Q2Model(DrugBaseModel):
    # 
    def __init__(
        self,
        model_name_or_path: str,
        d_type: str = "bfloat16",
        num_labels: int = 600,   
        use_metrics: bool = False,
        is_train: bool = False,
        class_freq: Optional[np.ndarray] = None, 
        train_num: Optional[int] = None
    ):
        self.num_labels = num_labels
        self.trust_remote_code: bool = True
        self.d_type = d_type
        self.use_metrics = use_metrics
        self.config = AutoConfig.from_pretrained(
            model_name_or_path,
            trust_remote_code=self.trust_remote_code
        )
        self.config.num_labels = num_labels
        super().__init__(self.config)
        self.model = AutoModel.from_pretrained(
            model_name_or_path, 
            config=self.config,
            trust_remote_code=self.trust_remote_code,
            dtype=self.d_type,
            attn_implementation="flash_attention_2" if is_train else "eager" # TODO: The input parameters are attn_implementation, flash_attention_2 for training, and eager for inference.
        )
        self.drug_classifier  = nn.Linear(self.config.hidden_size, num_labels)
        self.dropout = nn.Dropout(0.1)
        self.loss_fn = ResampleLoss(reweight_func='rebalance', loss_weight=1.0,
                             focal=dict(focal=True, alpha=0.5, gamma=2),
                             logit_reg=dict(init_bias=0.05, neg_scale=2.0),
                             map_param=dict(alpha=0.1, beta=10.0, gamma=0.9), 
                             class_freq=class_freq, train_num=train_num)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        task_ids=None,
        **kwargs
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if input_ids is not None:
            input_ids = input_ids.to(self.model.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.model.device)
        if inputs_embeds is not None:
            inputs_embeds = inputs_embeds.to(self.model.device)
        if input_ids is not None:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=return_dict,
                output_hidden_states=False,
                output_attentions=False, 
            )
        elif inputs_embeds is not None:
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=return_dict,
                output_hidden_states=False, 
                output_attentions=False, 
            )
        else:
            raise ValueError("Either input_ids or inputs_embeds must be provided")
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            hidden_states = outputs.hidden_states[-1]  # [B, L, H]
        elif hasattr(outputs, 'last_hidden_state'):
            hidden_states = outputs.last_hidden_state
        else:
            hidden_states = outputs[0] if isinstance(outputs, tuple) else outputs
        if hasattr(hidden_states, 'hidden_states'):
            hidden_states = hidden_states.hidden_states[-1] if hidden_states.hidden_states else hidden_states
        # take the last non-pad token
        if input_ids is not None:
            pad_token_id = getattr(self.model.config, "pad_token_id", None)
            if pad_token_id is not None:
                non_pad_mask = (input_ids != pad_token_id).int()
                token_idx = torch.arange(input_ids.shape[-1], device=input_ids.device)
                last_token = (token_idx * non_pad_mask).argmax(-1)
            else:
                last_token = torch.full((input_ids.size(0),), input_ids.shape[1] - 1, device=input_ids.device)
            pooled_output = hidden_states[torch.arange(hidden_states.size(0)), last_token]
        else:
            raise
        
        pooled_output = self.dropout(pooled_output)
        logits = self.drug_classifier(pooled_output)  # [batch_size, num_labels]
        
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels.float()) # DB Loss
    
        metrics = None
        if self.use_metrics:
            probs = torch.sigmoid(logits)
            probs_np = probs.detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy()
            metrics = self.compute_metrics(probs_np, labels_np)

        return DrugRecommendOutput(
            loss=loss,
            logits=logits,
            metrics=metrics
        )
