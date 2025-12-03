# TODO
import os
import yaml
import json
import pandas as pd
from tqdm import tqdm
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import AutoTokenizer
from transformers import DefaultDataCollator
from peft import LoraConfig, PeftModel, PeftConfig

from src.worker.common.data_process import drug_classification_map_fn_optimized_eval
from src.worker.global_models.baseline import GenerRecommendBaselineModel

class MyCollator:
    def __init__(self, tokenizer):
        self.base = DefaultDataCollator(return_tensors="pt")

    def __call__(self, batch):
        ids = [item["ID"] for item in batch]
        for item in batch:
            del item["ID"]

        padded = self.base(batch)
        padded["ID"] = ids
        return padded
    
class DrugEvaler():
    def __init__(self, config_path, lora_model_path, test_file_path, save_path, save_file, **kwargs):
        self.config_path = config_path
        self.save_file = save_file
        self._load_config()
        print(self.config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.config.model_name_or_path,
            trust_remote_code=True
        )
        self.tokenizer.padding_side = 'left'
        self.eos_id = self.tokenizer.eos_token_id

        self.model = GenerRecommendBaselineModel(
            model_name_or_path=self.config.model_name_or_path,
            dtype='bfloat16',
            eos_id=self.eos_id
        )
        self.low_cpu_mem_usage = kwargs.get('low_cpu_mem_usage', True)
        self.lora_model = PeftModel.from_pretrained(
            model=self.model,
            model_id=lora_model_path,
            low_cpu_mem_usage=self.low_cpu_mem_usage
        )
        try:
            self.lora_model.to(self.device)
        except Exception as e:
            print("Warning: moving lora_model to device failed:", e)

        self.lora_model.eval()
        torch.cuda.empty_cache()
        self.test_file_path = test_file_path
        self.save_path = save_path

        self.collator = MyCollator(self.tokenizer)

    def _lora_test_dataloader(self):
        data = pd.read_json(self.test_file_path, lines=True)
        test_ds = Dataset.from_pandas(data)
        test_dataset = test_ds.map(
            drug_classification_map_fn_optimized_eval,
            fn_kwargs={
                "tokenizer": self.tokenizer,
                "max_seq_length": self.config.max_seq_length,
            },
            remove_columns=test_ds.column_names
        )
        data_collator = self.collator
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=getattr(self.config, "eval_batch_size", 2),
            collate_fn=data_collator,
            shuffle=False,
            num_workers=0,
        )
        return test_dataloader

    def _load_config(self):
        with open(self.config_path, 'r', encoding='utf-8') as file:
            config_dict = yaml.safe_load(file)
            self.config = SimpleNamespace(**config_dict)

    @torch.no_grad()
    def eval(self):
        test_dataloader = self._lora_test_dataloader()
        all_results = []
        debug_run_one_batch = False
        for batch_idx, batch in enumerate(tqdm(test_dataloader, desc="DrugEvaluating")):
            if batch_idx % 10 == 0:
                print(f"Starting batch {batch_idx}/{len(test_dataloader)}")
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            # labels = batch.get('labels', None)
            save_id = batch.get('ID')
            print('id')
            print(save_id)
            outputs = self.lora_model.drug_eval(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            generated_tokens = outputs.gen_token_id
            predicted_texts = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            pred_drug_sets = self._extract_drugs(predicted_texts, self._get_valid_drug_set(), True)
            print('pred')
            print(pred_drug_sets)
            for i, (save_id_item, drug_set) in enumerate(zip(save_id, pred_drug_sets)):
                if drug_set is None:
                    pred_drug_list = []
                elif isinstance(drug_set, set):
                    pred_drug_list = [str(drug).strip() for drug in drug_set if drug is not None]
                elif isinstance(drug_set, (list, tuple)):
                    pred_drug_list = [str(drug).strip() for drug in drug_set if drug is not None]
                else:
                    raise ValueError(f"Error: 未知的药物集合类型: {type(drug_set)}")
                result = {
                    "ID": str(save_id_item),
                    "prediction": pred_drug_list
                }
                all_results.append(result)

            if debug_run_one_batch:
                print("DEBUG: finished one batch run, exiting (debug mode).")
                break

        out_file = self._save_results_to_json(all_results)
        print("Saved results to:", out_file)

    def _get_valid_drug_set(self):
        with open('/data/lzm/DrugRecommend/src/data/CDrugRed-A-v1/候选药物列表.json', 'r', encoding='utf-8') as f:
            original_drug_list = json.load(f)
        valid_drug_set = set()
        for drug in original_drug_list:
            standardized_drug_name = drug.replace('，', '').replace('、', '').replace(' ', '')
            if standardized_drug_name:
                valid_drug_set.add(standardized_drug_name)
        return valid_drug_set

    def _extract_drugs(self, text_list, valid_drugs_set: set = None, filter: bool = False):
        drug_sets = []      
        for text in text_list:
            if not text:
                drug_sets.append(set())
                continue
            normalized_text = text.replace('，', ',').replace('、', ',')
            no_space_text = normalized_text.replace(' ', '')
            drug_items = [item for item in no_space_text.split(',') if item]
            current_drug_set = set(drug_items)
            if filter:
                current_drug_set = current_drug_set.intersection(valid_drugs_set)
            drug_sets.append(current_drug_set)
                
        return drug_sets
    
    def _save_results_to_json(self, results):
        os.makedirs(self.save_path, exist_ok=True)
        output_file = os.path.join(self.save_path, self.save_file)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        return output_file
    
if __name__ == '__main__':
    drug_evaler = DrugEvaler(
        '/data/lzm/DrugRecommend/resource/output/checkpoint_save/baseline/config.yaml',
        '/data/lzm/DrugRecommend/resource/output/checkpoint_save/baseline/checkpoint-2200',
        '/data/lzm/DrugRecommend/src/data/error.jsonl',
        '/data/lzm/DrugRecommend/resource/output/submit',
        'drug_predictions1203'
    )
    drug_evaler.eval()