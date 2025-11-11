import argparse
import os
import yaml
import numpy as np
import pandas as pd
import random
import time
import datetime
import json
from pathlib import Path
from datasets import Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader

from src.worker.global_models.base_encoder import EncoderModelBaseBert
import src.worker.tool.bert_utils as utils
from src.worker.tool.bert_utils import cosine_lr_schedule, cos_with_warmup_lr_scheduler, compute_class_frequency,process_data_for_drug_prediction
from src.worker.common.drug_bert_dataset import create_dataset, create_sampler, create_loader
from src.worker.common.metrics import compute_simple_metrics

def train(model, data_loader, optimizer, epoch):
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.8f}'))
    metric_logger.add_meter('cls_loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('ndf_loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('cmm_loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))

    header = 'Train Epoch: [{}]'.format(epoch)#
    print_freq = 10

    for i, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        mode_1 = batch["mode_1"]
        mode_2 = batch["mode_2"] 
        labels = batch["labels"]
        drug_output = model(mode_1, mode_2, labels)

        cls_loss = drug_output.cls_loss * 20.
        ndf_loss = drug_output.ndf_loss * 1.
        cmm_loss = drug_output.cmm_loss * 1.
        loss = cls_loss + ndf_loss + cmm_loss
        # loss = cls_loss  + cmm_loss
        # loss = cls_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_logger.update(cls_loss=cls_loss.item())
        metric_logger.update(ndf_loss=ndf_loss.item())
        metric_logger.update(cmm_loss=cmm_loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]['lr'])
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.8f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}

@torch.inference_mode()
def evaluation(model, data_loader, fusion_weights=None):
    model.eval()
    text_bs=256
    if fusion_weights is None:
        fusion_weights = {'mode_1': 0.3, 'mode_2': 0.3, 'fusion': 0.4}

    dataset = data_loader.dataset
    mode_1s = getattr(dataset, "mode_1_data")
    mode_2s = getattr(dataset, "mode_2_data")
    labels_all = getattr(dataset, "labels")

    num_text = len(mode_1s)
    all_probs = {k: [] for k in ['mode_1', 'mode_2', 'fusion', 'ensemble']}
    all_labels = []

    for i in range(0, num_text, text_bs):
        batch_mode1 = mode_1s[i: i + text_bs]
        batch_mode2 = mode_2s[i: i + text_bs]
        batch_labels = labels_all[i: i + text_bs]
        if isinstance(batch_labels, torch.Tensor):
            pass
        elif isinstance(batch_labels, np.ndarray):
            batch_labels = torch.from_numpy(batch_labels).float()
        elif isinstance(batch_labels, list):
            # supports list of lists / list of np.ndarray / list of tensors
            try:
                batch_labels = torch.tensor(batch_labels, dtype=torch.float32)
            except Exception:
                batch_labels = torch.stack([
                    x if isinstance(x, torch.Tensor)
                    else torch.tensor(np.array(x), dtype=torch.float32)
                    for x in batch_labels
                ], dim=0)
        else:
            raise TypeError(f"Unexpected batch_labels type: {type(batch_labels)}")

        outputs = model.eval_func(batch_mode1, batch_mode2)

        # logits -> probs
        p1 = torch.sigmoid(outputs.logits_mode_1)
        p2 = torch.sigmoid(outputs.logits_mode_2)
        pf = torch.sigmoid(outputs.logits_mode_fusion)

        pe = (fusion_weights['mode_1'] * p1 +
              fusion_weights['mode_2'] * p2 +
              fusion_weights['fusion'] * pf)

        for k, p in zip(['mode_1', 'mode_2', 'fusion', 'ensemble'], [p1, p2, pf, pe]):
            all_probs[k].append(p.detach())

        all_labels.append(batch_labels.detach())

    all_labels = torch.cat(all_labels, dim=0)
    for k in all_probs:
        all_probs[k] = torch.cat(all_probs[k], dim=0)

    results = {}
    for k, preds in all_probs.items():
        f1, jaccard = torch_metrics_gpu(preds, all_labels)
        results[f"{k}_f1"] = f1
        results[f"{k}_jaccard"] = jaccard
        results[f"{k}_score"] = (f1 + jaccard) * 0.5

    print("\n" + "=" * 100)
    print("Fast GPU Evaluation:")
    print("=" * 100)
    for modality in ['mode_1', 'mode_2', 'fusion', 'ensemble']:
        print(f"{modality:10} | F1: {results[f'{modality}_f1']:.4f} | "
              f"Jaccard: {results[f'{modality}_jaccard']:.4f} | "
              f"Score: {results[f'{modality}_score']:.4f}")

    return results["ensemble_score"]

def torch_metrics_gpu(preds, labels, threshold=0.5, eps=1e-8):
    """GPU上计算micro F1和Jaccard"""
    preds = preds.float()
    labels = labels.float()
    device = preds.device
    labels = labels.to(device)
    preds = (preds > threshold).float()
    labels = labels.float()

    tp = (preds * labels).sum(dim=0)
    fp = (preds * (1 - labels)).sum(dim=0)
    fn = ((1 - preds) * labels).sum(dim=0)

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = (2 * precision * recall / (precision + recall + eps)).mean().item()
    jaccard = (tp / (tp + fp + fn + eps)).mean().item()
    return f1, jaccard

def main(args):
    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset ####
    print("Creating drug dataset")
    train_dataset, val_dataset = create_dataset(args.train_path, args.val_path, args.drug_file)
    num_classes = len(train_dataset.drugs)

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        samplers = create_sampler([train_dataset], [True], num_tasks, global_rank) + [None]
    else:
        samplers = [None, None]

    train_loader, val_loader = create_loader([train_dataset, val_dataset], samplers,
                                                          batch_size=[args.batch_size_train] + [
                                                              args.batch_size_test],
                                                          num_workers=[4, 4],
                                                          is_trains=[True, False],
                                                          collate_fns=[None, None])
    all_data = pd.read_json(args.all_data_file, lines=True)
    train_ds = Dataset.from_pandas(all_data)
    all_dataset = train_ds.map(
        process_data_for_drug_prediction,
        fn_kwargs={
            "drug_to_idx": train_dataset.drug_to_idx,
            "num_drugs": num_classes
        },
        remove_columns=train_ds.column_names
    )
    class_freq, train_num = compute_class_frequency(all_dataset, num_classes)
    #### Model ####
    print("Creating model")
    model = EncoderModelBaseBert(bert_path=args.bert_path, num_drug=num_classes, hidden_size=args.hidden_size, max_token=args.max_token,
                                 task=args.task, cls_loss_type=args.cls_loss_type, class_freq=class_freq, train_num=train_num)

    model = model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module


    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = args.init_lr
        weight_decay = args.weight_decay

        if 'mlm_head' in key:
            lr = args.init_lr * 5

        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    optimizer = torch.optim.AdamW(params=params, lr=args.init_lr, weight_decay=args.weight_decay)

    best = 0
    epoch_eval = args.epoch_eval

    print("Start training")
    start_time = time.time()

    for epoch in range(0, args.max_epoch):
        if not args.evaluate:
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)

            cosine_lr_schedule(optimizer, epoch, args.max_epoch, args.init_lr, args.min_lr)

            train_stats = train(model, train_loader, optimizer, epoch)
        if epoch % epoch_eval != 0:
            continue

        if utils.is_main_process():
            val_result = evaluation(model_without_ddp, val_loader)
            if val_result > best:
                save_obj = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                }
                if not args.evaluate:
                    torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best.pth'))
                best = val_result
        dist.barrier()
        torch.cuda.empty_cache()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='output/Retrieval_Person')
    parser.add_argument('--evaluate', action='store_true') # 保持不动即可
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--max_epoch', default=0, type=int)
    parser.add_argument('--batch_size_train', default=128, type=int)
    parser.add_argument('--batch_size_test', default=128, type=int)
    parser.add_argument('--init_lr', default=0.0001, type=float)
    parser.add_argument('--epoch_eval',default=1,type=int)
    parser.add_argument('--bert_path', default="/data1/nuist_llm/TrainLLM/ModelCkpt/bert/base_chinese", type=str, help='')

    parser.add_argument('--train_path', default="", type=str, help='jsonl train_path')
    parser.add_argument('--val_path', default="", type=str, help='jsonl train_path')
    parser.add_argument('--drug_file', default="", type=str, help='json train_path')
    parser.add_argument('--all_data_file', default="", type=str, help='jsonl train_path')
    parser.add_argument('--hidden_size',default=1,type=int)
    parser.add_argument('--max_token',default=1,type=int)
    parser.add_argument('--cls_loss_type', default="", type=str, help='cls_loss_type train_path')

    parser.add_argument('--weight_decay', default=0.1, type=float, help='')
    parser.add_argument('--min_lr', default=0.1, type=float, help='')
    parser.add_argument('--task', nargs='*', type=str, default=['local'], help='task name')
    args = parser.parse_args()

    main(args)