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
                module_name = name.split('.')[-1]
                lora_module_names.add(module_name)
                logger.debug(f"Candidate LoRA module -> {name} -> {module_name}")
    if 'lm_head' in lora_module_names:
        logger.warning("find lm_head layer in lora_module_names, we will remove it")
        lora_module_names.remove('lm_head')
    if 'output_layer' in lora_module_names:
        logger.warning("find output_layer layer in lora_module_names, we will remove it")
        lora_module_names.remove('output_layer')
    
    lora_module_names = list(lora_module_names)
    logger.info(f'GLM LoRA target module names: {lora_module_names}')
    return lora_module_names


def get_trainer(args, train_dataset, data_collator, model, logger):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    logger.info(f"Model moved to device: {device}")
    if args.use_lora:
        logger.info("using lora to train your model")
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
        model = get_peft_model(model, config)
    logger.info("🔍 LoRA包装前的可训练参数:")
    trainable_before = [name for name, param in model.named_parameters() if param.requires_grad]
    for name in trainable_before:
        logger.info(f"  - {name}")
    model = model.to(device)
    for name, param in model.named_parameters():
        if "drug_adapter" in name or "drug_classifier" in name:
            param.requires_grad = True
    logger.info("🔍 LoRA包装后的可训练参数:")
    trainable_before = [name for name, param in model.named_parameters() if param.requires_grad]
    for name in trainable_before:
        logger.info(f"  - {name}")
    use_bfloat16 = torch.cuda.is_bf16_supported()
    return 1
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
    }
    
    swanlab_callback = SwanLabCallback(
        project=args.task_name,
        experiment_name=args.task_name,
        description=args.task_des,
        workspace=None,
        config=swanlab_config,
    )
    # TODO @2Elian: record metrics of train processing.
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
