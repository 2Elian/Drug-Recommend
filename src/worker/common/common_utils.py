import os
import yaml
import torch

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

def setup_device(logger, gpu_id: int = 2):
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
        logger.info(f"use GPU: {torch.cuda.get_device_name(gpu_id)}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3:.1f} GB")
    else:
        device = torch.device("cpu")
        logger.warning("use CPU")
    return device