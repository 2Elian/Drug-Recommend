# 生成式推荐算法
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig, PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput
from typing import Optional, List
from .base_model import FocalLoss, DrugRecommendOutput, DrugLmBaseModel, ResampleLoss, DrugRecommendEvalOutput, MultiLabelCircleLoss
from src.utils.log import get_logger

class GenerRecommendBaselineModel(DrugLmBaseModel):
    def __init__(
        self,
        args,
        tokenizer,
        model_name_or_path: str,
        d_type: str = "bfloat16",
        **kwargs
    ):
        self.drug_model_logger = get_logger(name="DrugGenerCommendBaselineModel")
        self.trust_remote_code = True
        
        glm_config = AutoConfig.from_pretrained(
            model_name_or_path,
            trust_remote_code=True
        )
        self.tokenizer = tokenizer
        super().__init__(
            config=glm_config,
            tokenizer=self.tokenizer,
            model_name=model_name_or_path,
            lora_r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            dtype=d_type,
            lora_dropout=args.lora_dropout,
            train_mode=args.train_mode
        )
        
        self.args = args
        self.ignore_index = args.ignore_index
        self.lm_loss_fct = nn.CrossEntropyLoss(ignore_index=self.ignore_index)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        lm_logits = outputs.logits   # [B, L, Vocab]
        V = lm_logits.size(-1)
        print("lm_logits.shape:", lm_logits.shape)          # [B, L, V]
        print("labels.shape:", labels.shape, "dtype:", labels.dtype)
        print("labels device:", labels.device)

        # 报最大/最小值
        labels_cpu = labels.detach().cpu()
        print("labels.max():", labels_cpu.max().item(), "labels.min():", labels_cpu.min().item())
        print("tokenizer.vocab_size:", len(self.tokenizer) if hasattr(self, "tokenizer") else None)
        print("model vocab (lm_logits.size(-1)):", V)
        lm_loss = None
        if labels is not None:
            lm_loss = self.lm_loss_fct(
                lm_logits.view(-1, lm_logits.size(-1)).float(),
                labels.view(-1)
            )

        return DrugRecommendOutput(
            all_loss=lm_loss,
            lm_loss=lm_loss,
            cls_loss=None
        )

    @torch.no_grad()
    def drug_eval(self, input_ids=None, attention_mask=None, **kwargs):
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
        
        MAX_DRUG_TOKENS = 21 
        generated_ids = self.backbone.generate(
            **inputs,
            max_new_tokens=MAX_DRUG_TOKENS,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            do_sample=False,
            num_beams=1,
        )
        prompt_length = input_ids.size(1)
        generated_tokens = generated_ids[:, prompt_length:]
        return DrugRecommendEvalOutput(
            logits=generated_tokens,
            cls_loss=None
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