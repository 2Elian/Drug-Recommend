import json
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from os.path import join
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import AutoTokenizer
import argparse
from datasets import Dataset
from transformers import DataCollatorWithPadding

from src.worker.baseline.utils import process_data_for_drug_prediction, load_drug_vocab, compute_metrics, compute_metrics_fast

def setup_device(gpu_id=2):
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
        print(f"use GPU: {torch.cuda.get_device_name(gpu_id)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3:.1f} GB")
    else:
        device = torch.device("cpu")
        print("use CPU")
    return device


def load_model_and_tokenizer(model_path, device, base_model_path=None):
    from peft import PeftModel, PeftConfig
    from src.worker.baseline.model import GenericForMultiLabelClassification
    config = PeftConfig.from_pretrained(model_path)
    base_model_path = base_model_path or config.base_model_name_or_path

    drug_to_idx, idx_to_drug, all_drugs = load_drug_vocab("/data/lzm/DrugRecommend/src/models/baseline/dataset/pre_drug.json")
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
    print("model struct:")
    print(model)

    print("\n LoRA layer :")
    try:
        from peft import get_peft_model_state_dict
        lora_state = get_peft_model_state_dict(model)
        print(f"There are a total of {len(lora_state)} LoRA layer parameters. Examples of the first 10 parameter names:")
        for i, k in enumerate(list(lora_state.keys())[:10]):
            print(f"  {i+1}. {k}")
    except Exception as e:
        print("Unable to retrieve LoRA parameters:", e)
    print("\nBase model internal structure:")
    if hasattr(model, "base_model"):
        try:
            print(model.base_model)
        except:
            print(model.base_model.model)
    model.to(device)
    model.eval()

    print(f"The LoRA model has been successfully loaded and moved to...{device}")
    return model, tokenizer, drug_to_idx, idx_to_drug, num_drugs

def evaluate_test_set_optimized(model, tokenizer, test_file, drug_to_idx, num_drugs, device, max_seq_length=512, batch_size=64):
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
            labels = batch.pop("labels").cpu().numpy()
            outputs = model(**batch)
            logits = outputs.logits

            probs = torch.sigmoid(logits).cpu().numpy()

            all_labels.append(labels)
            all_probs.append(probs)
            del batch

    all_labels = np.vstack(all_labels)
    all_probs = np.vstack(all_probs)
    
    # clean GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # eval_pred = (all_probs, all_labels)
    # compute metrics
    # metrics = compute_metrics(eval_pred)
    metrics = compute_metrics_fast(all_probs, all_labels)
    
    return metrics, all_probs, all_labels

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/data/lzm/DrugRecommend/src/models/baseline/output/drug_prediction_deepspeed", help="lora model path")
    parser.add_argument("--test_file", type=str, default="/data/lzm/DrugRecommend/src/models/baseline/dataset/eval.jsonl", help="test file path")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (can be set larger for GPUs)")
    parser.add_argument("--max_seq_length", type=int, default=2000, help=" ")
    parser.add_argument("--optimized", action="store_true", help=" ")
    
    args = parser.parse_args()
    device = setup_device()
    
    print("Loading the model and tokenizer...")
    model, tokenizer, drug_to_idx, idx_to_drug, num_drugs = load_model_and_tokenizer(
        args.model_path, device, "/data1/nuist_llm/TrainLLM/ModelCkpt/glm/glm4-8b-chat"
    )
    
    print("Start evaluating the test set...")
    if args.optimized:
        metrics, predictions, labels = evaluate_test_set_optimized(
            model=model,
            tokenizer=tokenizer,
            test_file=args.test_file,
            drug_to_idx=drug_to_idx,
            num_drugs=num_drugs,
            device=device,
            max_seq_length=args.max_seq_length,
            batch_size=args.batch_size
        )
    
    print("\n" + "="*50)
    print("Test set evaluation results:")
    print("="*50)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    results_file = join(args.model_path, "test_evaluation_results.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()