#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/11/1 23:09
# @Author  : lizimo@nuist.edu.cn
# @File    : hf_tokenizer.py
# @Description:
from dataclasses import dataclass
from typing import List

from transformers import AutoTokenizer

from src.worker.common.base_tokenizer import BaseTokenizer


@dataclass
class HFTokenizer(BaseTokenizer):
    def __post_init__(self):
        self.enc = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)

    def encode(self, text: str) -> List[int]:
        return self.enc.encode(text, add_special_tokens=False)

    def decode(self, token_ids: List[int]) -> str:
        return self.enc.decode(token_ids, skip_special_tokens=True)
