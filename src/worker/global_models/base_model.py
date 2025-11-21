import torch
import torch.nn as nn
import numpy as np
import bitsandbytes as bnb
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from transformers import AutoModelForCausalLM, AutoConfig, PreTrainedModel, PretrainedConfig
from src.utils.log import get_logger
class DrugCausalLMBaseModel(PreTrainedModel):
    config_class = PretrainedConfig
    def __init__(
        self,
        config: PretrainedConfig,
        model_name_or_path: str,
        dtype: str = "bfloat16",
        **kwargs
    ):
        super().__init__(config=config)
        self.config = config
        self.model_name_or_path = model_name_or_path
        torch_dtype = getattr(torch, dtype, torch.float32)
        self.backbone = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            config=config,
            dtype=torch_dtype,
            trust_remote_code=kwargs.pop("trust_remote_code", False),
            **kwargs
        )
        # has_lm_head = hasattr(self.backbone, 'lm_head')
        # has_output_layer = hasattr(self.backbone, 'output_layer')
        # if not (has_lm_head or has_output_layer):
        #     raise AttributeError("Backbone model must have an 'lm_head' or 'output_layer' for Causal LM tasks.")
        self.hidden_size = self.backbone.config.hidden_size

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Subclass must implement the forward method.")
    
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if hasattr(self.backbone, "gradient_checkpointing_enable"):
            print("INFO: Enabling gradient checkpointing on the backbone model.")
            self.backbone.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
            )
        else:
            raise ValueError(
                f"Backbone model {self.backbone.__class__.__name__} does not support gradient checkpointing."
            )


@dataclass
class DrugRecommendOutput:
    all_loss: Optional[torch.FloatTensor] = None
    lm_loss: Optional[torch.FloatTensor] = None
    aux_loss: Optional[torch.FloatTensor] = None

@dataclass
class DrugRecommendEvalOutput:
    loss: Optional[torch.FloatTensor] = None
    gen_token_id: Optional[torch.FloatTensor] = None
