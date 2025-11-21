import yaml
import torch
from transformers import (
    TrainingArguments,
    Trainer,
)
from swanlab.integration.transformers import SwanLabCallback

from src.worker.common.metrics import compute_metrics, genercommend_compute_metrics

"""
**memory estimate**
    python -c 'from transformers import AutoModel; \
    from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live; \
    model = AutoModel.from_pretrained("/data1/nuist_llm/TrainLLM/ModelCkpt/glm/glm4-8b-chat"); \
    estimate_zero3_model_states_mem_needs_all_live(model, num_gpus_per_node=2, num_nodes=1)'
"""

def load_yaml_config(config_path):
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        print(f"Error loading YAML config from {config_path}: {e}")
        raise e

def get_trainer(args, train_dataset, eval_dataset, data_collator, model, is_gener):
    """
    evaluation_strategy="steps",   # 或 "epoch"
    eval_steps=500,                # 每500步验证一次
    load_best_model_at_end=True,   # 保存最优模型
    metric_for_best_model="final_score",
    greater_is_better=True,
"""
    from src.worker.common.metrics import GLOBAL_TOKENIZER
    if hasattr(model, 'tokenizer') and model.tokenizer is not None:
        GLOBAL_TOKENIZER = model.tokenizer
    else:
        raise AttributeError
    # use_bfloat16 = torch.cuda.is_bf16_supported()
    train_args = TrainingArguments(
        # save path and batch_size
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,

        # train log step and eval step
        logging_steps=args.logging_steps,
        eval_strategy=args.evaluation_strategy,
        eval_steps=args.eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="final_score",
        greater_is_better=True,
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
        dataloader_pin_memory=args.dataloader_pin_memory,
        save_total_limit=args.save_total_limit,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
    )

    swanlab_config = {
        "lora_rank": args.lora_rank if args.use_lora else "none",
        "lora_alpha": args.lora_alpha if args.use_lora else "none",
        "lora_dropout": args.lora_dropout if args.use_lora else "none",
        "dataset": args.task_dataset_name,
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

    trainer = DrugGenerRecommendTrainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[swanlab_callback],
        compute_metrics=genercommend_compute_metrics,
    ) if is_gener else DrugTrainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[swanlab_callback],
        compute_metrics=compute_metrics,
    )
    
    return trainer

class DrugTrainer(Trainer):
    # TODO 自定义计算模型loss + 训练过程评估展示到swanlab
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = outputs.all_loss
        step = self.state.global_step
        should_log = (
            step > 0 
            and self.args.logging_strategy == "steps"
            and (step % self.args.logging_steps == 0)
        )

        if should_log:
            if hasattr(outputs, "cls_loss") and outputs.cls_loss is not None:
                self.log({"cls_loss": outputs.cls_loss.detach().item()})
            if hasattr(outputs, "lm_loss") and outputs.lm_loss is not None:
                self.log({"lm_loss": outputs.lm_loss.detach().item()})

        return (loss, outputs) if return_outputs else loss

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only: bool,
        ignore_keys=None,
    ):
        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            outputs = model.drug_eval(**inputs)
            loss = outputs.cls_loss if has_labels else None
        logits = outputs.logits if hasattr(outputs, "logits") else None
        labels = inputs["labels"] if has_labels else None
        if prediction_loss_only:
            return (loss, None, None)
        return (loss, logits, labels)
    
class DrugGenerRecommendTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = outputs.all_loss
        step = self.state.global_step
        should_log = (
            step > 0 
            and self.args.logging_strategy == "steps"
            and (step % self.args.logging_steps == 0)
        )

        if should_log:
            if hasattr(outputs, "cls_loss") and outputs.cls_loss is not None:
                self.log({"cls_loss": outputs.cls_loss.detach().item()})
            if hasattr(outputs, "lm_loss") and outputs.lm_loss is not None:
                self.log({"lm_loss": outputs.lm_loss.detach().item()})

        return (loss, outputs) if return_outputs else loss

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only: bool,
        ignore_keys=None,
    ):
        has_labels = "labels" in inputs
        labels = inputs["labels"].clone() if has_labels else None
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            outputs = model.drug_eval(**inputs) 
            loss = None
            if has_labels and not prediction_loss_only:
                loss_outputs = model(**inputs)
                loss = loss_outputs.all_loss
        generated_tokens = outputs.logits 
        return (loss, generated_tokens, labels)