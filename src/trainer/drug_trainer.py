"""
**memory estimate**
    python -c 'from transformers import AutoModel; \
    from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live; \
    model = AutoModel.from_pretrained("/data1/nuist_llm/TrainLLM/ModelCkpt/glm/glm4-8b-chat"); \
    estimate_zero3_model_states_mem_needs_all_live(model, num_gpus_per_node=2, num_nodes=1)'
this file, its purpose is obtrain a trainer for drug task
"""
import bitsandbytes as bnb
import yaml
import torch
import torch.nn as nn
from transformers import (
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model
from swanlab.integration.transformers import SwanLabCallback
from src.worker.common.metrics import genercommend_compute_metrics, GLOBAL_TOKENIZER


class DrugGenerRecommendTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = outputs.all_loss
        # step = self.state.global_step
        # should_log = (
        #     step > 0 
        #     and self.args.logging_strategy == "steps"
        #     and (step % self.args.logging_steps == 0)
        # )

        # if should_log:
        #     if hasattr(outputs, "lm_loss") and outputs.lm_loss is not None:
        #         self.log({"lm_loss": outputs.lm_loss.detach().item()})
        #     if hasattr(outputs, "aux_loss") and outputs.aux_loss is not None:
        #         self.log({"aux_loss": outputs.aux_loss.detach().item()})

        return (loss, outputs) if return_outputs else loss

    # def prediction_step(
    #     self,
    #     model,
    #     inputs,
    #     prediction_loss_only: bool,
    #     ignore_keys=None,
    # ):
    #     has_labels = "labels" in inputs 
    #     labels = inputs["labels"].clone() if has_labels else None
    #     inputs = self._prepare_inputs(inputs)
    #     with torch.no_grad():
    #         outputs = model.drug_eval(**inputs) 
    #         loss = None
    #         if has_labels and not prediction_loss_only:
    #             loss_outputs = model(**inputs)
    #             loss = loss_outputs.all_loss
    #     generated_tokens = outputs.gen_token_id 
    #     return (loss, generated_tokens, labels)

def load_yaml_config(config_path):
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        print(f"Error loading YAML config from {config_path}: {e}")
        raise e

def find_all_target_module(model, train_mode):
        assert train_mode in ['lora', 'qlora']
        cls = bnb.nn.Linear4bit if train_mode == 'qlora' else nn.Linear
        lora_module_names = set()
        target_patterns = [
            'query_key_value',
            'dense',
            'dense_h_to_4h',
            'dense_4h_to_h',
        ]
        for name, module in model.named_modules():
            if isinstance(module, cls):
                if any(pattern in name for pattern in target_patterns):
                    module_name = name.split('.')[-1]
                    lora_module_names.add(module_name)
                    # self.logger.info(f"Candidate LoRA module -> {name} -> {module_name}")
        if 'lm_head' in lora_module_names:
            print("find lm_head layer in lora_module_names, we will remove it")
            lora_module_names.remove('lm_head')
        if 'output_layer' in lora_module_names:
            print("find output_layer layer in lora_module_names, we will remove it")
            lora_module_names.remove('output_layer')
        
        lora_module_names = list(lora_module_names)
        print(f'GLM LoRA target module names: {lora_module_names}')
        return lora_module_names

def get_trainer(args, train_dataset, data_collator, model):
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=find_all_target_module(model, train_mode='lora'),
        bias="none", # 'all' or 'lora_only'
        use_rslora=True # while it is True，will use Rank-Stabilized LoRA，该算法会将适配器缩放因子设置为lora_alpha/math.sqrt(r)，因为实践证明这样效果更好。否则，它将使用原始默认值lora_alpha/r
        # modules_to_save=["word_embeddings", "output_layer"] # 这是一个选项，可以让embedding层和输出层也参与训练
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    train_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        # eval_strategy=args.evaluation_strategy,
        # eval_steps=args.eval_steps,
        # load_best_model_at_end=True,
        # metric_for_best_model="final_score",
        # greater_is_better=True,
        num_train_epochs=args.num_train_epochs,
        save_steps=args.save_steps,
        learning_rate=args.learning_rate,
        gradient_checkpointing=args.gradient_checkpointing,
        report_to=args.report_to,
        seed=args.seed,
        optim=args.optim,
        local_rank=args.local_rank,
        ddp_find_unused_parameters=False,
        fp16=args.fp16,
        bf16=args.bf16,
        remove_unused_columns=False,
        dataloader_num_workers=args.dataloader_num_workers,
        # dataloader_pin_memory=args.dataloader_pin_memory,
        save_total_limit=args.save_total_limit,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
    )

    swanlab_config = {
        "dataset": args.task_dataset_name,
        "learning_rate": args.learning_rate,
    }

    swanlab_callback = SwanLabCallback(
        project=args.task_name,
        experiment_name=args.task_name,
        description=args.task_des,
        workspace=None,
        config=swanlab_config,
    )

    trainer = DrugGenerRecommendTrainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        # eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[swanlab_callback],
        # compute_metrics=genercommend_compute_metrics,
    )
    return trainer
