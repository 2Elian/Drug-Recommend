import logging
import sys
import os
import yaml
from logging.handlers import RotatingFileHandler
import bitsandbytes as bnb
import torch.nn as nn
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    TrainingArguments,
    Trainer,
)
import json
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from swanlab.integration.transformers import SwanLabCallback


def find_all_linear_names(model, train_mode, logger):
    assert train_mode in ['lora', 'qlora']
    cls = bnb.nn.Linear4bit if train_mode == 'qlora' else nn.Linear
    lora_module_names = set()
    # param names of glm4
    target_patterns = [
        'query_key_value',
        'dense',
        'dense_h_to_4h',
        'dense_4h_to_h',
        # 'word_embeddings', # optional
        # 'input_layernorm', # optional
        # 'final_layernorm', # optional
    ]
    
    for name, module in model.named_modules():
        if isinstance(module, cls):
            if any(pattern in name for pattern in target_patterns):
                names = name.split('.')
                lora_module_names.add(names[-1])

    if 'lm_head' in lora_module_names:
        logger.warning("find lm_head layer in lora_module_names, we will remove it")
        lora_module_names.remove('lm_head')
    if 'output_layer' in lora_module_names:
        logger.warning("find output_layer layer in lora_module_names, we will remove it")
        lora_module_names.remove('output_layer')
    
    lora_module_names = list(lora_module_names)
    logger.info(f'GLM LoRA target module names: {lora_module_names}')
    return lora_module_names

def setup_distributed(args, logger):
    if args.distributed:
        if args.local_rank == -1:
            raise ValueError("The local_rank was not initialized correctly. Please ensure that the parameter is passed through a distributed startup script, such as torchrun.")

        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(args.local_rank)
        logger.info(f"Distributed training is enabled, Local rank: {args.local_rank}")
    else:
        logger.warning("Distributed training is not enabled; it is in single-threaded mode.")

def process_data(data: dict, tokenizer, max_seq_length):
    text = data["text"]
    labels = data["labels"]
    encoding = tokenizer(
        text,
        truncation=True,
        padding=False,
        max_length=max_seq_length,
        return_tensors=None,
    )
    return {
        "input_ids": encoding["input_ids"],
        "attention_mask": encoding["attention_mask"],
        "labels": labels
    }

def get_trainer(args, train_dataset, data_collator, model, logger, compute_metrics):
    # setup_distributed(args, logger)
    
    # if args.distributed:
    #     model.to(args.local_rank)
    #     model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    model = model.to(device)
    logger.info(f"Model moved to device: {device}")
    if args.use_lora:
        logger.info("using lora to train your model")
        # target_modules = find_all_linear_names(
        #     model.module if isinstance(model, DDP) else model, 
        #     args.train_mode, 
        #     logger
        # )
        target_modules = find_all_linear_names(model, args.train_mode, logger)
        config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            target_modules=target_modules,
            task_type=TaskType.SEQ_CLS,
            inference_mode=False
        )
        # model = get_peft_model(model.module if isinstance(model, DDP) else model, config)
        model = get_peft_model(model, config)
    
    use_bfloat16 = torch.cuda.is_bf16_supported()

    fsdp_config = None
    fsdp_strategy = None
    if args.use_fsdp:
        fsdp_strategy = args.fsdp_sharding_strategy
        fsdp_config = {
            "sharding_strategy": args.fsdp_sharding_strategy,
            "min_num_params": args.fsdp_min_num_params,
            "cpu_offload": args.fsdp_offload_params,
            "backward_prefetch": args.fsdp_backward_prefetch,
            "forward_prefetch": args.fsdp_forward_prefetch,
            "use_orig_params": True,
        }
        logger.info(f"use fsdp config: {fsdp_config}")

        train_args = TrainingArguments(
            output_dir=args.output_dir,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            logging_steps=args.logging_steps,
            num_train_epochs=args.num_train_epochs,
            save_steps=args.save_steps,
            learning_rate=args.learning_rate,
            save_on_each_node=True,
            gradient_checkpointing=args.gradient_checkpointing,
            report_to=args.report_to,
            seed=args.seed,
            optim=args.optim,
            local_rank=args.local_rank,
            ddp_find_unused_parameters=False,
            fp16=args.fp16,
            bf16=not args.fp16 and use_bfloat16,
            remove_unused_columns=False,
            dataloader_num_workers=args.dataloader_num_workers,
            dataloader_pin_memory=args.dataloader_pin_memory,
            # eval_strategy=args.evaluation_strategy if eval_dataset is not None else "no",
            # eval_steps=args.eval_steps if eval_dataset is not None else None,
            # load_best_model_at_end=args.load_best_model_at_end if eval_dataset is not None else False,
            # metric_for_best_model=args.metric_for_best_model if eval_dataset is not None else None,
            # greater_is_better=args.greater_is_better,
            fsdp=fsdp_strategy,
            fsdp_config=fsdp_config,
            save_total_limit=args.save_total_limit,
            warmup_steps=args.warmup_steps,
            lr_scheduler_type=args.lr_scheduler_type,
            weight_decay=args.weight_decay,
            max_grad_norm=args.max_grad_norm,
        )
    else:
        train_args = TrainingArguments(
            output_dir=args.output_dir,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            logging_steps=args.logging_steps,
            num_train_epochs=args.num_train_epochs,
            save_steps=args.save_steps,
            learning_rate=args.learning_rate,
            save_on_each_node=True,
            gradient_checkpointing=args.gradient_checkpointing,
            report_to=args.report_to,
            seed=args.seed,
            optim=args.optim,
            local_rank=args.local_rank,
            ddp_find_unused_parameters=False,
            fp16=args.fp16,
            bf16=not args.fp16 and use_bfloat16,
            per_device_eval_batch_size=2,
            eval_accumulation_steps=4,
            remove_unused_columns=False,
            dataloader_num_workers=args.dataloader_num_workers,
            dataloader_pin_memory=args.dataloader_pin_memory,
            # eval_strategy=args.evaluation_strategy if eval_dataset is not None else "no",
            # eval_steps=args.eval_steps if eval_dataset is not None else None,
            # load_best_model_at_end=args.load_best_model_at_end if eval_dataset is not None else False,
            # metric_for_best_model=args.metric_for_best_model if eval_dataset is not None else None,
            # greater_is_better=args.greater_is_better,
            deepspeed=args.deepspeed_config if args.use_deepspeed else None,
            save_total_limit=args.save_total_limit,
            warmup_steps=args.warmup_steps,
            lr_scheduler_type=args.lr_scheduler_type,
            weight_decay=args.weight_decay,
            max_grad_norm=args.max_grad_norm,
        )

    logger.info("Model structure:")
    if hasattr(model, 'print_trainable_parameters'):
        model.print_trainable_parameters()

    swanlab_config = {
        "lora_rank": args.lora_rank if args.use_lora else "none",
        "lora_alpha": args.lora_alpha if args.use_lora else "none",
        "lora_dropout": args.lora_dropout if args.use_lora else "none",
        "dataset": args.task_dataset_name,
        "num_drugs": model.num_labels,
        "use_focal_loss": args.use_focal_loss,
        "learning_rate": args.learning_rate,
        # "with_validation": eval_dataset is not None,
    }
    
    swanlab_callback = SwanLabCallback(
        project=args.task_name,
        experiment_name=args.task_name,
        description=args.task_des,
        workspace=None,
        config=swanlab_config,
    )
    
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        # eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[swanlab_callback],
        # compute_metrics=compute_metrics,
    )
    
    return trainer

def get_logger(name: str = "app_logger",
               level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if logger.hasHandlers():
        return logger
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    return logger

def load_drug_vocab(drug_file_path: str):
    with open(drug_file_path, 'r', encoding='utf-8') as f:
        drugs = json.load(f)
    drug_to_idx = {drug: idx for idx, drug in enumerate(drugs)}
    """
    example
    # {
    #     "左甲状腺素钠片": 0,
    #     "氨氯地平片": 1, 
    #     "阿卡波糖": 2,
    #     "瑞格列奈": 3,
    #     ...
    # }
    """
    idx_to_drug = {idx: drug for drug, idx in drug_to_idx.items()}
    """
    example
    # {
    #     0: "左甲状腺素钠片",
    #     1: "氨氯地平片",
    #     2: "阿卡波糖", 
    #     3: "瑞格列奈",
    #     ...
    # }
    """
    return drug_to_idx, idx_to_drug, drugs

def create_multihot_labels(drug_list: list, drug_to_idx: dict, num_drugs: int):
    labels = torch.zeros(num_drugs, dtype=torch.float) # [0, 0, 0, 0, ...]
    for drug in drug_list:
        if drug in drug_to_idx:
            labels[drug_to_idx[drug]] = 1.0
    return labels # example [1, 1, 0, 0, 1, 0, ...]

def build_medical_prompt(data: dict) -> str:
    sections = []
    basic_info = []
    gender = data.get("性别")
    if gender is not None:
        basic_info.append(f"性别：{gender}")
    birth_date = data.get("出生日期")
    if birth_date is not None:
        basic_info.append(f"出生日期：{birth_date}")
    bmi = data.get("BMI")
    if bmi is not None:
        bmi_status = "肥胖" if bmi >= 28 else "超重" if bmi >= 24 else "正常"
        basic_info.append(f"BMI：{bmi}（{bmi_status}）")
    ethnicity = data.get("民族")
    if ethnicity is not None:
        basic_info.append(f"民族：{ethnicity}")
    if basic_info:
        sections.append("【患者基本信息】\n" + "，".join(basic_info))
    chief_complaint = data.get("主诉")
    if chief_complaint is not None and chief_complaint.strip():
        sections.append("【主诉】\n" + chief_complaint.strip())
    present_illness = data.get("现病史")
    if present_illness is not None and present_illness.strip():
        sections.append("【现病史】\n" + present_illness.strip())
    admission_status = data.get("入院情况")
    if admission_status is not None and admission_status.strip():
        sections.append("【入院情况】\n" + admission_status.strip())
    process_desc = data.get("诊疗过程描述")
    if process_desc is not None and process_desc.strip():
        clean_process = process_desc.strip()
        sections.append("【诊疗过程】\n" + clean_process)
    
    past_history = data.get("既往史")
    if past_history is not None and past_history.strip():
        sections.append("【既往史】\n" + past_history.strip())
    diagnoses = data.get("出院诊断")
    if diagnoses is not None and isinstance(diagnoses, list):
        valid_diagnoses = []
        for diagnosis in diagnoses:
            if diagnosis is not None:
                if isinstance(diagnosis, str) and diagnosis.strip():
                    valid_diagnoses.append(diagnosis.strip())
                elif isinstance(diagnosis, (int, float)):
                    valid_diagnoses.append(str(diagnosis))
        
        if valid_diagnoses:
            sections.append("【出院诊断】\n" + "，".join(valid_diagnoses))
    if sections:
        prompt = "\n\n".join(sections)
    else:
        raise
    
    instruction = "根据上述患者医疗信息，预测出院时需要带哪些药物："
    final_text = prompt + "\n\n" + instruction
    
    return final_text


def process_data_for_drug_prediction(data: dict, tokenizer, max_seq_length: int, drug_to_idx: dict, num_drugs: int):
    input_text = build_medical_prompt(data)
    # tokenize input_text
    encoding = tokenizer(
        input_text,
        truncation=True,
        padding=False, # subsequently, it will be uniformly filled in the collator
        max_length=max_seq_length,
        return_tensors=None, # None will rerurn {'input_ids': [101, 123, 456, 102], 'attention_mask': [1, 1, 1, 1]}. If not None, it will return {'input_ids': tensor([[101, 123, 456, 102]]), 'attention_mask': tensor([[1, 1, 1, 1]])}
        add_special_tokens=True, # Add [CLS] at the beginning and [SEP] at the end
    )
    
    drug_list = data.get("出院带药列表", [])
    labels = create_multihot_labels(drug_list, drug_to_idx, num_drugs)
    
    return {
        "input_ids": encoding["input_ids"],
        "attention_mask": encoding["attention_mask"],
        "labels": labels.tolist(),
    }

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    probs = 1 / (1 + np.exp(-predictions))
    metrics = {}
    thresholds = [0.3, 0.4, 0.5]
    preds_05 = (probs > 0.5).astype(int)
    
    for threshold in thresholds:
        preds = (probs > threshold).astype(int)
        jaccard_scores = []
        precision_scores = []
        recall_scores = [] 
        f1_scores = []
        max_samples = min(1000, len(labels))
        
        for i in range(max_samples):
            y_true = labels[i]
            y_pred = preds[i]
            
            true_set = set(np.where(y_true == 1)[0])
            pred_set = set(np.where(y_pred == 1)[0])
            
            # Jaccard
            intersection = len(true_set & pred_set)
            union = len(true_set | pred_set)
            jaccard = intersection / union if union > 0 else 0.0
            jaccard_scores.append(jaccard)
            
            # Precision
            precision = intersection / len(pred_set) if len(pred_set) > 0 else 0.0
            precision_scores.append(precision)
            
            # Recall
            recall = intersection / len(true_set) if len(true_set) > 0 else 0.0
            recall_scores.append(recall)
            
            # F1
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            f1_scores.append(f1)
        
        avg_jaccard = np.mean(jaccard_scores)
        avg_precision = np.mean(precision_scores)
        avg_recall = np.mean(recall_scores)
        avg_f1 = np.mean(f1_scores)
        score = 0.5 * (avg_jaccard + avg_f1)

        metrics.update({
            f"jaccard_th{threshold}": avg_jaccard,
            f"precision_avg_th{threshold}": avg_precision,
            f"recall_avg_th{threshold}": avg_recall,
            f"f1_avg_th{threshold}": avg_f1,
            f"score_th{threshold}": score,
        })
    
    if len(labels) > 0:
        metrics.update({
            "f1_micro": f1_score(labels[:max_samples], preds_05[:max_samples], average="micro", zero_division=0),
            "precision_micro": precision_score(labels[:max_samples], preds_05[:max_samples], average="micro", zero_division=0),
            "recall_micro": recall_score(labels[:max_samples], preds_05[:max_samples], average="micro", zero_division=0),
        })
    
    metrics["avg_labels_per_sample"] = labels.sum(axis=1).mean() if len(labels) > 0 else 0
    metrics["avg_preds_per_sample"] = preds_05.sum(axis=1).mean() if len(labels) > 0 else 0
    
    exact_match = np.all(preds_05 == labels, axis=1)
    metrics["exact_match_accuracy"] = exact_match.mean() if len(labels) > 0 else 0

    del predictions, labels, probs, preds_05
    import gc
    gc.collect()
    
    return metrics


def save_config(args, extra_vars=None, save_path=None):
    cfg = vars(args).copy()
    if extra_vars:
        cfg.update(extra_vars)
    if save_path is None:
        save_path = os.path.join(cfg.get("output_dir", "./output"), "config.yaml")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    return save_path

def compute_metrics_fast(predictions, labels, thresholds=[0.3, 0.4, 0.5]):
    metrics = {}
    labels = labels.astype(int)
    for threshold in thresholds:
        preds = (predictions > threshold).astype(int)
        intersection = np.sum(labels & preds, axis=1)
        union = np.sum(labels | preds, axis=1)
        true_positives = np.sum(labels, axis=1)
        pred_positives = np.sum(preds, axis=1)
        jaccard = np.divide(intersection, union, out=np.zeros_like(intersection, dtype=float), where=union!=0)
        precision = np.divide(intersection, pred_positives, out=np.zeros_like(intersection, dtype=float), where=pred_positives!=0)
        recall = np.divide(intersection, true_positives, out=np.zeros_like(intersection, dtype=float), where=true_positives!=0)
        f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(precision), where=(precision + recall)!=0)
        
        avg_jaccard = np.mean(jaccard)
        avg_precision = np.mean(precision)
        avg_recall = np.mean(recall)
        avg_f1 = np.mean(f1)
        score = 0.5 * (avg_jaccard + avg_f1)

        metrics.update({
            f"jaccard_th{threshold}": avg_jaccard,
            f"precision_avg_th{threshold}": avg_precision,
            f"recall_avg_th{threshold}": avg_recall,
            f"f1_avg_th{threshold}": avg_f1,
            f"score_th{threshold}": score,
        })
        
        metrics.update({
            f"f1_micro_th{threshold}": f1_score(labels, preds, average="micro", zero_division=0),
            f"precision_micro_th{threshold}": precision_score(labels, preds, average="micro", zero_division=0),
            f"recall_micro_th{threshold}": recall_score(labels, preds, average="micro", zero_division=0),
        })

    preds_05 = (predictions > 0.5).astype(int)
    metrics.update({
        "avg_labels_per_sample": np.mean(np.sum(labels, axis=1)),
        "avg_preds_per_sample": np.mean(np.sum(preds_05, axis=1)),
        "exact_match_accuracy": np.mean(np.all(preds_05 == labels, axis=1)),
        "total_samples": len(labels)
    })
    
    return metrics
