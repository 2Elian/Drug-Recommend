# Thanks to xtuner 0.1.0
import json
import torch
from datasets import load_dataset
from mmengine.dataset import DefaultSampler
from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook,
                            LoggerHook, ParamSchedulerHook)
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from mmengine.registry import MAP_FUNC
from torch.optim import AdamW
from functools import partial
from transformers import AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from src.worker.xtuner.dataset import process_hf_dataset
from src.worker.xtuner.dataset.collate_fns import default_collate_fn
from src.worker.xtuner.dataset.map_fns import openai_map_fn, template_map_fn_factory
from src.worker.xtuner.engine.hooks import (DatasetInfoHook, HFCheckpointHook,
                                            ThroughputHook,
                                            VarlenAttnArgsToMessageHubHook)
from src.worker.xtuner.engine.runner import TrainLoop
from src.worker.xtuner.model import SupervisedFinetune
from src.worker.xtuner.model.transformers_models.qwen2 import Qwen2ForCausalLM
from src.worker.xtuner.parallel.sequence import SequenceParallelSampler
from src.worker.xtuner.utils import PROMPT_TEMPLATE
from mmengine.visualization import Visualizer, TensorboardVisBackend
from src.worker.global_models.baseline import BaselineModel
from src.worker.common.data_process import drug_classification_map_fn

data_files = ["/data/lzm/DrugRecommend/src/worker/dataset/train.jsonl"]
pretrained_model_name_or_path = "/data1/nuist_llm/TrainLLM/ModelCkpt/glm/glm4-8b-chat"
project='drug-recommend',
experiment_name='xtuner-train'
max_length = 4096
max_position_embeddings = 4096 
pack_to_max_length = False
use_varlen_attn = False 
prompt_template = PROMPT_TEMPLATE.chatglm3 # TODO 
# parallel
sequence_parallel_size = 2 

# Scheduler & Optimizer
batch_size = 2              # per_device
accumulative_counts = 8     # bs = 1 GPU * batch_size_per_device * accumulative_counts
accumulative_counts = sequence_parallel_size * accumulative_counts

dataloader_num_workers = 8
max_epochs = 2
optim_type = AdamW
lr = 2e-6 # 1e-4 1e-5 1e-6 1e-7
betas = (0.9, 0.999)
weight_decay = 0.01
max_norm = 1  # grad clip 
warmup_ratio = 0.03

# Save
save_steps = 200
save_total_limit = 3  # Maximum checkpoints to keep (-1 means unlimited)
log_step=50

# LoRA param
use_lora = True
lora_rank = 8
lora_alpha = 16
lora_dropout = 0.1

# drug-param
use_focal_loss = True
fp16 = False
is_train = True
use_metrics = False
num_drugs = 651 

tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=pretrained_model_name_or_path,
    trust_remote_code=True, use_fast=True,
    padding_side='right')

target_modules = ['query_key_value', 'dense_4h_to_h', 'dense', 'dense_h_to_4h']

lora_config = dict(
    type=LoraConfig,
    r=lora_rank,
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    bias="none",
    target_modules=target_modules,
    task_type=TaskType.SEQ_CLS,
    inference_mode=False
)

model = dict(
    type=SupervisedFinetune,
    use_varlen_attn=use_varlen_attn,
    max_position_embeddings=max_position_embeddings,
    lora=lora_config,
    llm=dict(
        type=BaselineModel,
        model_name_or_path=pretrained_model_name_or_path,
        d_type="float16" if fp16 else "bfloat16",
        use_focal_loss=use_focal_loss,
        num_labels=num_drugs,
        use_metrics=use_metrics,
        is_train=is_train
    )
)

sampler = SequenceParallelSampler if sequence_parallel_size > 1 else DefaultSampler

train_dataset = dict(
    type=process_hf_dataset,
    dataset=dict(type=load_dataset, path='json', data_files=data_files, cache_dir="/data1/nuist_llm/TrainLLM/dataCache/datasets"),
    tokenizer=tokenizer,
    max_length=max_length,
    dataset_map_fn=drug_classification_map_fn,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    remove_unused_columns=True,
    shuffle_before_pack=True,
    pack_to_max_length=pack_to_max_length,
    use_varlen_attn=use_varlen_attn)

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    dataset=train_dataset,
    sampler=dict(type=sampler, shuffle=True),
    collate_fn=dict(type=default_collate_fn, use_varlen_attn=use_varlen_attn))

# optimizer
optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(
        type=optim_type, lr=lr, betas=betas, weight_decay=weight_decay),
    clip_grad=dict(max_norm=max_norm, error_if_nonfinite=False), 
    accumulative_counts=accumulative_counts,
    loss_scale='dynamic', 
    dtype='bfloat16')

# learning policy
param_scheduler = [
    dict(
        type=LinearLR, # warmup 
        start_factor=1e-5,
        by_epoch=True,
        begin=0,
        end=warmup_ratio * max_epochs,
        convert_to_iter_based=True),
    dict(
        type=CosineAnnealingLR,
        eta_min=0.0,
        by_epoch=True,
        begin=warmup_ratio * max_epochs,
        end=max_epochs,
        convert_to_iter_based=True)
]

# train, val, test setting
train_cfg = dict(type=TrainLoop, max_epochs=max_epochs)


custom_hooks = [
    dict(type=DatasetInfoHook, tokenizer=tokenizer),
    dict(type=ThroughputHook),
    dict(type=HFCheckpointHook)
]


if use_varlen_attn:
    custom_hooks += [dict(type=VarlenAttnArgsToMessageHubHook)]

# configure default hooks
default_hooks = dict(
    # record the time of every iteration.
    timer=dict(type=IterTimerHook),
    # print log every 10 iterations.
    logger=dict(type=LoggerHook, log_metric_by_epoch=False, interval=log_step),
    # enable the parameter scheduler.
    param_scheduler=dict(type=ParamSchedulerHook),
    # save checkpoint per `save_steps`.
    checkpoint=dict(
        type=CheckpointHook,
        by_epoch=False,
        interval=save_steps,
        max_keep_ckpts=save_total_limit),
    # set sampler seed in distributed evrionment.
    sampler_seed=dict(type=DistSamplerSeedHook),
)

# configure environment
env_cfg = dict(
    # whether to enable cudnn benchmark
    cudnn_benchmark=False,
    # set multi process parameters
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    # set distributed parameters
    dist_cfg=dict(backend='nccl'),
)

# set visualizer
visualizer = dict(type=Visualizer, vis_backends=[dict(type=TensorboardVisBackend)])

# visualizer = dict(
#     type='Visualizer',
#     vis_backends=[
#         dict(
#             type='SwanLabVisBackend',
#             save_dir='/data1/nuist_llm/TrainLLM/SFT-elian/Qwen2.5-Xtuner/logs',
#             project='mmengine-project',
#             experiment_name='experiment-1',
#             init_kwargs=dict(
#                 workspace='qwen-2.5-math',
#                 mode='cloud',
                
#             )
#         )
#     ]
# )

# set log level
log_level = 'INFO'

# load from which checkpoint
load_from = " "

# whether to resume training from the loaded checkpoint
resume = False

# Defaults to use random seed and disable `deterministic`
randomness = dict(seed=None, deterministic=False)

# set log processor
log_processor = dict(by_epoch=False)