import json
import torch
from typing import Dict, Any, List
from transformers import AutoTokenizer
from src.utils.helper import calculate_age, get_bmi_description
from src.utils.templates.sft_prompt import BASELINE_PROMPT

def load_drug_vocab(drug_file_path: str):
    with open(drug_file_path, 'r', encoding='utf-8') as f:
        drugs = json.load(f)
    drug_to_idx = {drug: idx for idx, drug in enumerate(drugs)}
    idx_to_drug = {idx: drug for drug, idx in drug_to_idx.items()}
    return drug_to_idx, idx_to_drug, drugs

def create_multihot_labels(drug_list: list, drug_to_idx: dict, num_drugs: int):
    labels = torch.zeros(num_drugs, dtype=torch.float) # [0, 0, 0, 0, ...]
    for drug in drug_list:
        if drug in drug_to_idx:
            labels[drug_to_idx[drug]] = 1.0
    return labels # example [1, 1, 0, 0, 1, 0, ...]

def build_medical_lm_prompt(data: Dict[str, Any], default: str = '未知') -> str:
    gender = data.get("性别", default)
    birth_date = data.get("出生日期")
    vis_time = data.get("就诊时间")
    bmi_t = data.get("BMI")
    age = default
    if birth_date and vis_time:
        age = calculate_age(birth_date, vis_time)
    bmi = default
    bmi_des = default
    if bmi_t is not None:
        try:
            bmi_value = float(bmi_t)
            bmi = f"{bmi_value:.2f}"
            bmi_des = get_bmi_description(bmi_value, default)
        except (ValueError, TypeError):
            pass
    text_fields = {
        "主诉": "complaint",
        "现病史": "history_now",
        "既往史": "history_past",
        "入院情况": "admission",
        "诊疗过程描述": "process",
        "出院诊断": "diagnosis",
    }
    cleaned_data = {}
    for key_cn, key_en in text_fields.items():
        value = data.get(key_cn)
        if value and isinstance(value, str):
            cleaned_data[key_en] = value.strip()
        elif value and isinstance(value, list) and len(value) > 0:
            cleaned_data[key_en] = ",".join(map(str, value)).strip()
        else:
            cleaned_data[key_en] = default
    _, _, drug_str = load_drug_vocab('/data/lzm/DrugRecommend/src/data/pre_drug.json')
    input_prompt = BASELINE_PROMPT['ZH']['TEMPLATE2'].format(
        sex=gender, age=age, bmi=bmi, bmi_des=bmi_des, process=cleaned_data["process"], admission=cleaned_data["admission"], complaint=cleaned_data["complaint"],
        history_now=cleaned_data["history_now"], history_past=cleaned_data["history_past"], diagnosis=cleaned_data["diagnosis"]# , drug_str=drug_str
    )
    return input_prompt

def drug_classification_map_fn_optimized_eval(data: dict, tokenizer: AutoTokenizer, max_seq_length: int)-> Dict[str, List[int]]:
    """
    Optimized mapping function for training and inference (evaluation/prediction).
    
    Args:
        data (dict): The input sample dictionary (e.g., medical record).
        tokenizer: The pre-trained tokenizer.
        max_seq_length (int): Maximum sequence length for truncation.
        special_drug_tokens (list): List of special drug token strings (e.g., ['<DRUG_X>']).
        is_train (bool): If True, prepares 'labels'; otherwise, omits 'labels'.
    """
    tokenizer.truncation_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    prompt_text = build_medical_lm_prompt(data)
    encoded_input = tokenizer(
        prompt_text, 
        max_length=max_seq_length, 
        add_special_tokens=False,
        truncation=False, 
        padding=False,
    )
    drug_list = data.get("出院带药列表", [])
    if not drug_list:
        labels = None
    else:
        drug_text = ",".join(drug_list)
        encoded_output = tokenizer(
        drug_text, 
        max_length=max_seq_length, 
        add_special_tokens=False,
        truncation=False, 
        padding=False,
        )
        labels = encoded_output["input_ids"]
    input_ids = (
            encoded_input["input_ids"]
    )
    attention_mask = encoded_input["attention_mask"]
    if len(input_ids) > max_seq_length:
        input_ids = input_ids[-max_seq_length:]
        attention_mask = attention_mask[-max_seq_length:]
    padding_len = max_seq_length - len(input_ids)
    if padding_len > 0:
        pad_ids = [tokenizer.pad_token_id] * padding_len
        pad_mask = [0] * padding_len
        is_left_padding = (tokenizer.pad_token != tokenizer.eos_token)
        if is_left_padding:
            input_ids = pad_ids + input_ids
            attention_mask = pad_mask + attention_mask
        else:
            input_ids += pad_ids
            attention_mask += pad_mask
    # 处理labels
    if labels is not None:
        labels_padding_len = max_seq_length - len(labels)
        labels = [-100] * labels_padding_len + labels
        assert len(input_ids) == max_seq_length, "The sequence length is not equal to max_seq_length"
        assert len(attention_mask) == max_seq_length, "The length of the attention mask is not equal to max_seq_length."
        # assert len(labels) == max_seq_length, "The length of the abels array is not equal to max_seq_length."
        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "ID": data['就诊标识']
        }
    else:
        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "ID": data['就诊标识']
        }       
        
    return result

def drug_classification_map_fn_optimized(data: dict, tokenizer: AutoTokenizer, max_seq_length: int)-> Dict[str, List[int]]:
    """
    Optimized mapping function for training and inference (evaluation/prediction).
    
    Args:
        data (dict): The input sample dictionary (e.g., medical record).
        tokenizer: The pre-trained tokenizer.
        max_seq_length (int): Maximum sequence length for truncation.
        special_drug_tokens (list): List of special drug token strings (e.g., ['<DRUG_X>']).
        is_train (bool): If True, prepares 'labels'; otherwise, omits 'labels'.
    """
    tokenizer.truncation_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    prompt_text = build_medical_lm_prompt(data)
    drug_list = data.get("出院带药列表", [])
    if not drug_list:
        drug_text = ""
    else:
        drug_text = "<|assistant|>" + ",".join(drug_list)
    encoded_input = tokenizer(
        prompt_text, 
        max_length=max_seq_length, 
        add_special_tokens=False,
        truncation=True, 
        padding=False,
    )
    encoded_output = tokenizer(
        drug_text, 
        max_length=max_seq_length, 
        add_special_tokens=False,
        truncation=True, 
        padding=False,
    )
    input_ids = (
            encoded_input["input_ids"] + encoded_output["input_ids"] + [tokenizer.eos_token_id]
    )
    attention_mask = encoded_input["attention_mask"] + encoded_output["attention_mask"] + [1]
    labels = ([-100] * len(encoded_input["input_ids"]) + encoded_output["input_ids"] + [tokenizer.eos_token_id]
                   )
    
    if len(input_ids) > max_seq_length:
        input_ids = input_ids[-max_seq_length:]
        attention_mask = attention_mask[-max_seq_length:]
        labels = labels[-max_seq_length:]
    padding_len = max_seq_length - len(input_ids)
    if padding_len > 0:
        pad_ids = [tokenizer.pad_token_id] * padding_len
        pad_mask = [0] * padding_len
        pad_labels = [-100] * padding_len 
        is_left_padding = (tokenizer.pad_token != tokenizer.eos_token)
        if is_left_padding:
            input_ids = pad_ids + input_ids
            attention_mask = pad_mask + attention_mask
            labels = pad_labels + labels
        else:
            input_ids += pad_ids
            attention_mask += pad_mask
            labels += pad_labels
    assert len(input_ids) == max_seq_length, "The sequence length is not equal to max_seq_length"
    assert len(attention_mask) == max_seq_length, "The length of the attention mask is not equal to max_seq_length."
    assert len(labels) == max_seq_length, "The length of the abels array is not equal to max_seq_length."
    result = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }
        
    return result

def get_labels_num(labels: List[int]) -> int:
    filtered_list = [x for x in labels if x != -100]
    return len(filtered_list)

if __name__ == '__main__':
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(
        '/data1/nuist_llm/TrainLLM/ModelCkpt/glm/glm4-8b-chat', 
        trust_remote_code=True
    )
    jsonl_file = '/data/lzm/DrugRecommend/src/data/CDrugRed-A-v1/CDrugRed_train.jsonl'
    index = 0
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line_number, line in enumerate(f, 1):
            stripped_line = line.strip()
            if not stripped_line:
                continue
            data = json.loads(stripped_line)
            result = drug_classification_map_fn_optimized(data, tokenizer, 2000)
            print(tokenizer.decode(result["input_ids"], skip_special_tokens=False))
            # labels = result["labels"]
            # len_labels = get_labels_num(labels)
        #     if len_labels>index:
        #         index=len_labels
        # print(index)
            