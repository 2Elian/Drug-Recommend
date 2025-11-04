from peft import PeftModel
from transformers import AutoModelForCausalLM
import torch
import os
from modelscope import AutoTokenizer
import shutil
from .model import GenericForMultiLabelClassification

def copy_files_not_in_B(A_path, B_path):
    if not os.path.exists(A_path):
        raise FileNotFoundError(f"The directory {A_path} does not exist.")
    if not os.path.exists(B_path):
        os.makedirs(B_path)
    files_in_A = os.listdir(A_path)
    files_in_A = set([file for file in files_in_A if not (".bin" in file or "safetensors" in file)])
    # List all files in directory B
    files_in_B = set(os.listdir(B_path))
    files_to_copy = files_in_A - files_in_B
    for file in files_to_copy:
        src_path = os.path.join(A_path, file)
        dst_path = os.path.join(B_path, file)
        if os.path.isdir(src_path):
            shutil.copytree(src_path, dst_path)
        else:
            shutil.copy2(src_path, dst_path)

def merge_lora_to_base_model():
    from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
    model_name_or_path = '/data1/nuist_llm/TrainLLM/ModelCkpt/glm/glm4-8b-chat'
    adapter_name_or_path = '/data/lzm/DrugRecommend/src/models/baseline/output/drug_prediction_deepspeed'
    save_path = '/data/lzm/DrugRecommend/src/models/baseline/output/drug_prediction_deepspeed/merge'
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    zero3_checkpoint_path = adapter_name_or_path
    fp32_model_path = os.path.join(save_path, 'fp32_model.pth')
    state_dict = get_fp32_state_dict_from_zero_checkpoint(zero3_checkpoint_path)
    torch.save(state_dict, fp32_model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    
    model = GenericForMultiLabelClassification(
        model_name_or_path=model_name_or_path,
        num_labels=651
    )
    model.load_state_dict(state_dict, strict=False)
    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path, safe_serialization=False)
    copy_files_not_in_B(model_name_or_path, save_path)

if __name__ == '__main__':
    merge_lora_to_base_model()