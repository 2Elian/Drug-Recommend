from os.path import join
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
)
from transformers import DataCollatorWithPadding
from src.utils.log import get_logger
from src.worker.common.data_process import load_drug_vocab, drug_classification_map_fn_optimized
from src.worker.common.common_utils import save_config
from src.configs.train_config import configuration_parameter
from src.worker.global_models.Q3_models import GenerRecommendBaselineModel
from src.trainer.trainer_util import get_trainer

# 生成式推荐算法
def fit():
    args = configuration_parameter()
    logger = get_logger(name='GenerRecommendDrugTrainer')
    drug_to_idx, _, all_drugs = load_drug_vocab(args.drug_file)
    is_gener = args.is_gener
    num_drugs = len(all_drugs)
    model_path = args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    special_drug_tokens = [f"<DRUG_{d}>" for d in all_drugs]
    tokenizer.add_special_tokens({"additional_special_tokens": special_drug_tokens})
    exra = {
        "num_drug": num_drugs
    }
    save_config(args, exra)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    data = pd.read_json('/data/lzm/DrugRecommend/src/data/CDrugRed-A-v1/CDrugRed_train.jsonl', lines=True)
    val_data = pd.read_json(args.eval_file, lines=True)
    # val_data = pd.read_json(args.eval_file, lines=True)
    logger.info(f"Loaded {len(data)} training examples")
    train_ds = Dataset.from_pandas(data)
    val_ds = Dataset.from_pandas(val_data)
    train_dataset = train_ds.map(
        drug_classification_map_fn_optimized,
        fn_kwargs={
            "tokenizer": tokenizer, 
            "max_seq_length": args.max_seq_length,
            "special_drug_tokens": special_drug_tokens,
            "is_train": True
        },
        remove_columns=train_ds.column_names
    )
    eval_dataset = val_ds.map(
        drug_classification_map_fn_optimized,
        fn_kwargs={
            "tokenizer": tokenizer, 
            "max_seq_length": args.max_seq_length,
            "special_drug_tokens": special_drug_tokens,
            "is_train": True
        },
        remove_columns=val_ds.column_names
    )
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer, 
        # padding='longest',  # 动态填充到批次中最长序列, 因为在map处理的时候 已经截断了 所以训练的显存会相对稳定
        padding=True, 
        return_tensors="pt"
    )
    if args.bf16:
        dtype = "bfloat16"
    else:
        dtype = "float16"
    logger.info("[Start] Model Initialization")
    model = GenerRecommendBaselineModel(
        args=args,
        tokenizer=tokenizer,
        model_name_or_path=model_path,
        d_type=dtype,
    )
    model.print_trainable_parameters()
    logger.info("[Start] Training")
    trainer = get_trainer(args, train_dataset, eval_dataset, data_collator, model, is_gener)
    trainer.train()
    
    final_save_path = join(args.output_dir)
    trainer.save_model(final_save_path)
    logger.info(f"Model saved to {final_save_path}")

if __name__ == "__main__":
    fit()