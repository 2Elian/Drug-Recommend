# 生成式推荐算法
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig, PreTrainedModel
from typing import Optional, List, Dict, Any
from .base_model import DrugRecommendOutput, DrugRecommendEvalOutput, DrugCausalLMBaseModel
from src.utils.log import get_logger

class GenerRecommendBaselineModel(DrugCausalLMBaseModel):
    def __init__(
            self,
            model_name_or_path: str,
            num_labels: Optional[int] = None,
            dtype: str = "bfloat16",
            trust_remote_code: bool = True,
            eos_id: int = None,
            **kwargs
        ):
            config = AutoConfig.from_pretrained(
                model_name_or_path,
                trust_remote_code=trust_remote_code
            )
            if num_labels is not None:
                config.num_labels = num_labels
            super().__init__(
                config=config,
                model_name_or_path=model_name_or_path,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
                **kwargs
            )
            if num_labels is not None:
                self.auxiliary_classifier = nn.Linear(self.hidden_size, num_labels)
            self.num_labels = num_labels
            self.eos_id = eos_id

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        aux_labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, Any]:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels, # AutoModelForCausalLM takes labels as input and automatically calculates the LM Loss.
            output_hidden_states=True if self.num_labels is not None else False,
            **kwargs
        )
        lm_loss = outputs.loss
        total_loss = lm_loss
        aux_loss = None
        if self.num_labels is not None and aux_labels is not None:
            # TODO 参考Llama的序列分类逻辑
            last_hidden_state = outputs.hidden_states[-1] 
            batch_size, seq_len, _ = last_hidden_state.shape
            sequence_lengths = attention_mask.sum(dim=1) - 1
            gather_indices = sequence_lengths.unsqueeze(-1).unsqueeze(-1).expand(batch_size, 1, self.hidden_size)
            pooled_output = torch.gather(last_hidden_state, 1, gather_indices).squeeze(1)
            aux_logits = self.auxiliary_classifier(pooled_output)
            aux_loss_fct = nn.CrossEntropyLoss()
            aux_loss = aux_loss_fct(aux_logits, aux_labels)
            total_loss = total_loss + aux_loss
        return DrugRecommendOutput(
             all_loss=total_loss,
             lm_loss=lm_loss,
             aux_loss=aux_loss,
        )

    @torch.no_grad()
    def drug_eval(self, input_ids=None, attention_mask=None, **kwargs):
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
        
        MAX_DRUG_TOKENS = 138 
        generated_ids = self.backbone.generate(
            **inputs,
            max_new_tokens=MAX_DRUG_TOKENS,
            eos_token_id=self.eos_id,
            do_sample=False,
            num_beams=1,
            use_cache=False,
        )
        prompt_length = input_ids.size(1)
        generated_tokens = generated_ids[:, prompt_length:]
        return DrugRecommendEvalOutput(
            gen_token_id=generated_tokens,
            loss=None
        )