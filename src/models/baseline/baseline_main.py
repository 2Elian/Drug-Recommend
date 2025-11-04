from os.path import join
import pandas as pd
import json
from datasets import Dataset
from transformers import (
    DataCollatorForSeq2Seq,
    AutoTokenizer,
    AutoConfig
)
from transformers import DataCollatorWithPadding
from src.models.baseline.utils import get_logger, process_data_for_drug_prediction, get_trainer, load_drug_vocab, compute_metrics, save_config
from src.models.baseline.config import configuration_parameter
from src.models.baseline.model import GenericForMultiLabelClassification

def main():
    args = configuration_parameter()
    logger = get_logger(name=args.task_name)
    logger.info("[Preparation] Loading drug vocabulary")
    drug_to_idx, idx_to_drug, all_drugs = load_drug_vocab(args.drug_file)
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
    # val_dataset = val_ds.map(
    #     process_data_for_drug_prediction,
    #     fn_kwargs={
    #         "tokenizer": tokenizer, 
    #         "max_seq_length": args.max_seq_length,
    #         "drug_to_idx": drug_to_idx,
    #         "num_drugs": num_drugs
    #     },
    #     remove_columns=val_ds.column_names
    # )
    
    logger.info(f"Processed dataset: {train_dataset}")
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer, 
        padding=True, 
        return_tensors="pt"
    )
    
    logger.info("[Start] Model Initialization")
    model = GenericForMultiLabelClassification(
        model_name_or_path=model_path,
        num_labels=num_drugs,
        use_focal_loss=args.use_focal_loss,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
    )
    
    logger.info("[Start] Training")
    trainer = get_trainer(args, train_dataset, data_collator, model, logger, compute_metrics)
    trainer.train()
    
    final_save_path = join(args.output_dir)
    trainer.save_model(final_save_path)
    logger.info(f"Model saved to {final_save_path}")
    
    # if args.load_best_model_at_end and val_dataset is not None:
    #     best_model_path = join(args.output_dir, "best_model")
    #     trainer.save_model(best_model_path)
    #     logger.info(f"Best model saved to {best_model_path}")
    
    # vocab_save_path = join(args.output_dir, "drug_vocab.json")
    # with open(vocab_save_path, 'w', encoding='utf-8') as f:
    #     json.dump({
    #         "drug_to_idx": drug_to_idx,
    #         "idx_to_drug": idx_to_drug,
    #         "all_drugs": all_drugs
    #     }, f, ensure_ascii=False, indent=2)
    # logger.info(f"Drug vocabulary saved to {vocab_save_path}")

    # if val_dataset is not None:
    #     logger.info("[Final Evaluation]")
    #     eval_results = trainer.evaluate()
    #     logger.info(f"finally evaluation result: {eval_results}")

if __name__ == "__main__":
    main()

