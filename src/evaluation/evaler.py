import json
import torch
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from os.path import join
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import AutoTokenizer
import argparse
from datasets import Dataset
from transformers import DataCollatorWithPadding

from src.utils.log import get_logger
from src.worker.common.data_process import process_data_for_drug_prediction, load_drug_vocab
from src.worker.common.metrics import compute_metrics_fast
from src.worker.common.common_utils import setup_device

def load_model_and_tokenizer(model_path, device, logger, pre_drug_path, base_model_path=None):
    from peft import PeftModel, PeftConfig
    from src.worker.baseline.model import GenericForMultiLabelClassification
    config = PeftConfig.from_pretrained(model_path)
    base_model_path = base_model_path or config.base_model_name_or_path
    drug_to_idx, idx_to_drug, all_drugs = load_drug_vocab(pre_drug_path)
    num_drugs = len(all_drugs)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    base_model = GenericForMultiLabelClassification(
        model_name_or_path=base_model_path,
        num_labels=num_drugs,
        use_focal_loss=True,
        focal_alpha=0.75,
        focal_gamma=2.0,
    )
    model = PeftModel.from_pretrained(base_model, model_path)
    logger.info("model struct:")
    logger.info(model)
    logger.info("\n LoRA layer :")
    try:
        from peft import get_peft_model_state_dict
        lora_state = get_peft_model_state_dict(model)
        print(f"There are a total of {len(lora_state)} LoRA layer parameters. Examples of the first 10 parameter names:")
        for i, k in enumerate(list(lora_state.keys())[:10]):
            print(f"  {i+1}. {k}")
    except Exception as e:
        logger.warning(e)
    logger.info("\nBase model internal structure:")
    if hasattr(model, "base_model"):
        try:
            logger.info(model.base_model)
        except:
            logger.info(model.base_model.model)
    model.to(device)
    model.eval()
    logger.info(f"The LoRA model has been successfully loaded and moved to...{device}")
    return model, tokenizer, drug_to_idx, idx_to_drug, num_drugs

def evaluate_test_set_optimized(model, tokenizer, test_file, drug_to_idx, 
                                idx_to_drug, is_submit, save_path, num_drugs, device, 
                                max_seq_length=512, batch_size=64, threshold=0.5):
    
    test_data = pd.read_json(test_file, lines=True)
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id if getattr(tokenizer, "eos_token_id", None) is not None else 0
    test_ds = Dataset.from_pandas(test_data)
    test_dataset = test_ds.map(
        process_data_for_drug_prediction,
        fn_kwargs={
            "tokenizer": tokenizer, 
            "max_seq_length": max_seq_length,
            "drug_to_idx": drug_to_idx,
            "num_drugs": num_drugs
        },
        remove_columns=test_ds.column_names
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=DataCollatorWithPadding(tokenizer=tokenizer, padding=True, return_tensors="pt") # notic: 
    )
    all_labels, all_probs = [], []
    model.eval()

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="evaling"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            if is_submit is False:
                labels = batch.pop("labels").cpu().numpy()
                all_labels.append(labels)
            del batch
    if is_submit is False:
        all_labels = np.vstack(all_labels)
    all_probs = np.vstack(all_probs)
    preds = (all_probs > threshold).astype(int)
    
    # Map predicted indices back to drug names
    predictions = []
    if "就诊标识" in test_data.columns:
        id_field = "就诊标识"
    else:
        raise

    for idx, row in enumerate(test_data.itertuples(index=False)):
        pred_indices = np.where(preds[idx] == 1)[0]
        pred_drugs = [idx_to_drug[i] for i in pred_indices]
        predictions.append({
            "ID": getattr(row, id_field),
            "prediction": pred_drugs
        })

    # Save JSON if required
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    if is_submit is False:
        metrics_res = compute_metrics_fast(all_probs, all_labels)
        directory = os.path.dirname(save_path)
        results_file = os.path.join(directory, "metrics.json")
        with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(metrics_res, f, indent=2, ensure_ascii=False)
    return predictions

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, default="/data1/nuist_llm/TrainLLM/ModelCkpt/glm/glm4-8b-chat", help="base model model path")
    parser.add_argument("--model_path", type=str, default="/data/lzm/DrugRecommend/src/models/baseline/output/drug_prediction_deepspeed", help="lora model path")
    parser.add_argument("--test_file", type=str, default="/data/lzm/DrugRecommend/src/models/baseline/dataset/eval.jsonl", help="test file path")
    parser.add_argument("--save_path", type=str, default="/data/lzm/DrugRecommend/src/data/CDrugRed-B-v1/predict_resullt.json", help="save result json file path")
    parser.add_argument("--pre_drug_path", type=str, default="/data/lzm/DrugRecommend/src/data/pre_drug.json", help="pre_drug_path")
    parser.add_argument("--is_submit", action="store_true",
                        help="whether it is a test set for the B chart")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (can be set larger for GPUs)")
    parser.add_argument("--gpu_id", type=int, default=2, help="gpu id")
    parser.add_argument("--threshold", type=float, default=0.5, help="probability threshold")
    parser.add_argument("--max_seq_length", type=int, default=2000, help=" ")

    args = parser.parse_args()
    logger = get_logger(name="Evaler")
    device = setup_device(logger=logger, gpu_id=args.gpu_id)
    logger.info("Loading the model and tokenizer...")
    model, tokenizer, drug_to_idx, idx_to_drug, num_drugs = load_model_and_tokenizer(
        model_path=args.model_path, device=device, logger=logger, pre_drug_path=args.pre_drug_path, base_model_path=args.base_model_path
    )
    
    logger.info("Start evaluating the test set...")
    predictions = evaluate_test_set_optimized(
        model=model,
        tokenizer=tokenizer,
        test_file=args.test_file,
        drug_to_idx=drug_to_idx,
        idx_to_drug=idx_to_drug,
        is_submit=args.is_submit,
        save_path=args.save_path,
        num_drugs=num_drugs,
        device=device,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size
    )
    return predictions
if __name__ == "__main__":
    predictions = main()