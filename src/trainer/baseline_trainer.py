from os.path import join
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
)
from transformers import DataCollatorWithPadding
from src.utils.log import get_logger
from src.worker.common.data_process import process_data_for_drug_prediction, load_drug_vocab
from src.worker.common.common_utils import save_config
from src.configs.train_config import configuration_parameter
from src.worker.global_models.baseline import BaselineModel
from src.trainer.trainer_util import get_trainer

def fit():
    args = configuration_parameter()
    logger = get_logger(name=args.task_name)
    logger.info("[Preparation] Loading drug vocabulary")
    drug_to_idx, _, all_drugs = load_drug_vocab(args.drug_file)
    num_drugs = len(all_drugs)
    logger.info(f"Total drugs: {num_drugs}")
    logger.info(f"First 10 drugs: {all_drugs[:10]}")
    logger.info("[Preparation] Loading tokenizer")
    model_path = args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    exra = {
        "num_drug": num_drugs
    }
    save_config(args, exra)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    
    logger.info("[Preparation] Processing Data")
    data = pd.read_json(args.train_file, lines=True)
    # val_data = pd.read_json(args.eval_file, lines=True)
    logger.info(f"Loaded {len(data)} training examples")
    train_ds = Dataset.from_pandas(data)
    # val_ds = Dataset.from_pandas(val_data)
    train_dataset = train_ds.map(
        process_data_for_drug_prediction,
        fn_kwargs={
            "tokenizer": tokenizer, 
            "max_seq_length": args.max_seq_length,
            "drug_to_idx": drug_to_idx,
            "num_drugs": num_drugs
        },
        remove_columns=train_ds.column_names
    )
    
    logger.info(f"Processed dataset: {train_dataset}")
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer, 
        # padding='longest',  # 动态填充到批次中最长序列, 因为在map处理的时候 已经截断了 所以训练的显存会相对稳定
        padding=True, 
        return_tensors="pt"
    )
    
    logger.info("[Start] Model Initialization")
    model = BaselineModel(
        model_name_or_path=model_path,
        num_labels=num_drugs,
        use_focal_loss=args.use_focal_loss,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        use_metrics=args.use_metrics,
        is_train=args.is_train,
    )
    
    logger.info("[Start] Training")
    trainer = get_trainer(args, train_dataset, data_collator, model, logger)
    trainer.train()
    
    final_save_path = join(args.output_dir)
    trainer.save_model(final_save_path)
    logger.info(f"Model saved to {final_save_path}")

if __name__ == "__main__":
    fit()