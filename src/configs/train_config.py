import argparse
import os
def configuration_parameter(yaml_config=None):
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for Drug model")
    parser.add_argument("--model_name_or_path", type=str, default="./model",
                        help="your glm-8b-chat path")
    parser.add_argument("--output_dir", type=str,
                        default="2048-16-32-epoch-2",
                        help="Directory to save the fine-tuned model and checkpoints")

    # train param
    parser.add_argument("--train_file", type=str, default="./data/single_datas.jsonl",
                        help="Path to the training data file in JSONL format")
    parser.add_argument("--task_name", type=str, default="drug-glm4-9b-lora",
                        help="your task name, example: drug-glm4-9b-lora")
    parser.add_argument("--task_dataset_name", type=str, default="drug-train-80precent",
                        help="your task dataset name, example: drug-train-80precent")
    parser.add_argument("--task_des", type=str, default="not des",
                        help="your task des, example: my drug recommend system")
    parser.add_argument("--train_type", type=str, default="A",
                        help="none")
    parser.add_argument("--num_train_epochs", type=int, default=2,
                        help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4,
                        help="Batch size per device during training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16,
                        help="Number of updates steps to accumulate before performing a backward/update pass")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="Learning rate for the optimizer")
    parser.add_argument("--max_seq_length", type=int, default=2048,
                        help="Maximum sequence length for the input")
    parser.add_argument("--logging_steps", type=int, default=1,
                        help="Number of steps between logging metrics")
    parser.add_argument("--save_steps", type=int, default=200,
                        help="Number of steps between saving checkpoints")
    parser.add_argument("--save_total_limit", type=int, default=1,
                        help="Maximum number of checkpoints to keep")
    # 用哪种策略?
    parser.add_argument("--lr_scheduler_type", type=str, default="constant_with_warmup",
                        help="Type of learning rate scheduler")
    parser.add_argument("--warmup_steps", type=int, default=100,
                        help="Number of warmup steps for learning rate scheduler, set it be 3-5 precent of train num")
    # lora
    parser.add_argument("--lora_rank", type=int, default=64,
                        help="Rank of LoRA matrices")
    parser.add_argument("--lora_alpha", type=int, default=16,
                        help="Alpha parameter for LoRA")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="Dropout rate for LoRA")

    # distributed
    parser.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", -1)),
                        help="Local rank for distributed training")
    parser.add_argument("--distributed", action="store_true",
                        help="Enable distributed training")

    # other param
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Enable gradient checkpointing to save memory")
    parser.add_argument("--optim", type=str, default="adamw_torch",
                        help="Optimizer to use during training")
    parser.add_argument("--train_mode", type=str, default="lora",
                        help="lora or qlora")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--fp16", action="store_true",
                        help="Use mixed precision (FP16) training")
    parser.add_argument("--bf16", action="store_true",
                        help="Use mixed precision (bFP16) training")
    parser.add_argument("--report_to", type=str, default=None,
                        help="Reporting tool for logging (e.g., tensorboard)")
    parser.add_argument("--dataloader_num_workers", type=int, default=8,
                        help="Number of workers for data loading")
    parser.add_argument("--save_strategy", type=str, default="steps",
                        help="Strategy for saving checkpoints ('steps', 'epoch')")
    parser.add_argument("--weight_decay", type=float, default=0,
                        help="Weight decay for the optimizer")
    parser.add_argument("--max_grad_norm", type=float, default=1,
                        help="Maximum gradient norm for clipping")
    parser.add_argument("--remove_unused_columns", type=bool, default=True,
                        help="Remove unused columns from the dataset")
    
    parser.add_argument("--drug_file", type=str, required=False,
                        help="Path to pre_drug.json file containing drug vocabulary")
    parser.add_argument("--use_focal_loss", action="store_true",
                        help="Use focal loss for imbalanced multi-label classification")
    parser.add_argument("--focal_alpha", type=float, default=0.75,
                        help="Alpha parameter for focal loss")
    parser.add_argument("--focal_gamma", type=float, default=2.0,
                        help="Gamma parameter for focal loss")
    parser.add_argument("--num_neg_samples", type=int, default=50,
                        help="Number of negative samples for training (if using negative sampling)")
    parser.add_argument("--model_type", type=str, default="glm", choices=["bert", "glm", "llama"],
                        help="Type of base model")
    parser.add_argument("--use_lora", action="store_true",
                        help="Use LoRA for parameter-efficient fine-tuning")
    parser.add_argument("--eval_file", type=str, default=None,
                        help="Path to evaluation data file")
    parser.add_argument("--eval_steps", type=int, default=500,
                        help="Number of steps between evaluations")
    parser.add_argument("--eval_ratio", type=float, default=0.2,
                        help="Ratio of data to use for evaluation (if no eval_file provided)")
    parser.add_argument("--evaluation_strategy", type=str, default="steps", 
                        choices=["no", "steps", "epoch"],
                        help="Evaluation strategy")
    parser.add_argument("--metric_for_best_model", type=str, default="best_score",
                        help="Metric for selecting the best model")
    parser.add_argument("--load_best_model_at_end", action="store_true",
                        help="Load the best model at the end of training")
    parser.add_argument("--greater_is_better", type=bool, default=True,
                        help="Whether higher metric values are better")
    parser.add_argument("--dataloader_pin_memory", action="store_true", default=False,
                        help="Pin memory for data loader (improves performance)")
    parser.add_argument("--dataloader_drop_last", action="store_true", default=True,
                        help="Drop last incomplete batch in distributed training")
    
    # fsdp param
    parser.add_argument("--use_fsdp", action="store_true", 
                        help="Use Fully Sharded Data Parallel")
    parser.add_argument("--fsdp_sharding_strategy", type=str, default="FULL_SHARD",
                        choices=["FULL_SHARD", "SHARD_GRAD_OP", "NO_SHARD"],
                        help="FSDP sharding strategy")
    parser.add_argument("--fsdp_offload_params", action="store_true",
                        help="Offload parameters to CPU")
    parser.add_argument("--fsdp_min_num_params", type=int, default=1e8,
                        help="Minimum number of parameters for FSDP auto wrap")
    parser.add_argument("--fsdp_backward_prefetch", type=str, default="BACKWARD_PRE",
                        choices=["BACKWARD_PRE", "BACKWARD_POST", "NONE"],
                        help="FSDP backward prefetch strategy")
    parser.add_argument("--fsdp_forward_prefetch", action="store_true",
                        help="Enable FSDP forward prefetch")
    
    # deepspeed param
    parser.add_argument("--use_deepspeed", action="store_true",
                        help="Use DeepSpeed")
    parser.add_argument("--deepspeed_config", type=str, default=None,
                        help="Path to DeepSpeed config file")
    
    parser.add_argument("--use_metrics", action="store_true", 
                        help=" ")
    parser.add_argument("--is_train", action="store_true", 
                        help=" ")
    parser.add_argument("--sp_size", type=int, default=2,
                        help="sp")
    parser.add_argument("--pack_to_max_length", action="store_true", 
                        help=" ")
    parser.add_argument("--use_varlen_attn", action="store_true", 
                        help=" ")
    
    parser.add_argument("--ignore_index", type=int, default=-100,
                        help="")
    parser.add_argument("--r1", type=float, default=0.6,
                        help="")
    parser.add_argument("--r2", type=float, default=2.5,
                        help="")
    parser.add_argument("--lm_loss", action="store_true", 
                        help=" ")


    args = parser.parse_args()

    if yaml_config:
        for key, value in yaml_config.items():
            setattr(args, key, value)

    return args
