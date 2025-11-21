from os.path import join
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
)
from transformers import DataCollatorWithPadding, DefaultDataCollator
from src.utils.log import get_logger
from src.worker.common.data_process import load_drug_vocab, drug_classification_map_fn_optimized, drug_classification_map_fn_optimized_eval
from src.worker.common.common_utils import save_config
from src.configs.train_config import configuration_parameter
from src.worker.global_models.baseline import GenerRecommendBaselineModel
from src.trainer.drug_trainer import get_trainer

# 生成式推荐算法
def fit():
    args = configuration_parameter()
    logger = get_logger(name='GenerRecommendFit')
    _, _, all_drugs = load_drug_vocab(args.drug_file)
    num_drugs = len(all_drugs)
    model_path = args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    tokenizer.padding_side = 'left'
    GLOBAL_TOKENIZER = tokenizer
    special_drug_tokens = [f"<DRUG_{d}>" for d in all_drugs]
    tokenizer.add_special_tokens({"additional_special_tokens": special_drug_tokens})
    exra = {
        "num_drug": num_drugs
    }
    save_config(args, exra)
    eos_id = tokenizer.eos_token_id
    data = pd.read_json('/data/lzm/DrugRecommend/src/data/CDrugRed-A-v1/CDrugRed_train.jsonl', lines=True)
    # val_data = pd.read_json(args.eval_file, lines=True)
    logger.info(f"Loaded {len(data)} training examples")
    train_ds = Dataset.from_pandas(data)
    # val_ds = Dataset.from_pandas(val_data)
    train_dataset = train_ds.map(
        drug_classification_map_fn_optimized,
        fn_kwargs={
            "tokenizer": tokenizer, 
            "max_seq_length": args.max_seq_length,
        },
        remove_columns=train_ds.column_names
    )
    # eval_dataset = val_ds.map(
    #     drug_classification_map_fn_optimized_eval,
    #     fn_kwargs={
    #         "tokenizer": tokenizer, 
    #         "max_seq_length": args.max_seq_length,
    #     },
    #     remove_columns=val_ds.column_names
    # )
    """
    关于transformers的DataCollator的小tips
    DataCollatorWithPadding用于所有需要将长度不一的序列 Padding 到相同长度的任务 --> 仅处理 input_ids, attention_mask 等输入字段。 不会处理 labels 字段。
    DataCollatorForSeq2Seq通用 Collator --> 同时处理输入和标签。会将 labels 填充到批次中最长序列的长度。
    DataCollatorForLanguageModeling用于MLM/CLM处理输入和标签。通常对 labels 进行特殊的 shift 或掩码操作。
    DefaultDataCollator默认的Collator 需要自己写填充和截断逻辑
    """
    data_collator = DefaultDataCollator(return_tensors="pt")
    if args.bf16:
        dtype = "bfloat16"
    else:
        dtype = "float16"
    logger.info("[Start] Model Initialization")
    model = GenerRecommendBaselineModel(
        model_name_or_path=model_path,
        dtype=dtype,
        eos_id=eos_id,
    )
    trainer = get_trainer(args, train_dataset, data_collator, model)
    logger.info("Start Training")
    trainer.train()
    
    final_save_path = join(args.output_dir)
    trainer.save_model(final_save_path)
    logger.info(f"Model saved to {final_save_path}")

if __name__ == "__main__":
    fit()