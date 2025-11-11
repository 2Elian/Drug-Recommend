import json
import torch

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
        basic_info.append(f"<性别>: {gender}")
    birth_date = data.get("出生日期")
    if birth_date is not None:
        basic_info.append(f"<出生日期>: {birth_date}")
    bmi = data.get("BMI")
    if bmi is not None:
        bmi_status = "肥胖" if bmi >= 28 else "超重" if bmi >= 24 else "正常"
        basic_info.append(f"<BMI>: {bmi}（{bmi_status}）")
    ethnicity = data.get("民族")
    if ethnicity is not None:
        basic_info.append(f"<民族>: {ethnicity}")
    if basic_info:
        sections.append("【患者基本信息】\n" + ":".join(basic_info) + "\n 【患者临床信息】")
    chief_complaint = data.get("主诉")
    if chief_complaint is not None and chief_complaint.strip():
        sections.append("<主诉>\n" + chief_complaint.strip())
    present_illness = data.get("现病史")
    if present_illness is not None and present_illness.strip():
        sections.append("<现病史>\n" + present_illness.strip())
    admission_status = data.get("入院情况")
    if admission_status is not None and admission_status.strip():
        sections.append("<入院情况>\n" + admission_status.strip())
    process_desc = data.get("诊疗过程描述")
    if process_desc is not None and process_desc.strip():
        clean_process = process_desc.strip()
        sections.append("<诊疗过程>\n" + clean_process)
    
    past_history = data.get("既往史")
    if past_history is not None and past_history.strip():
        sections.append("<既往史>\n" + past_history.strip())
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
            sections.append("<|diagnosis_start|><出院诊断>\n" + ":".join(valid_diagnoses)+ "<|diagnosis_end|>")
    if sections:
        prompt = "\n\n".join(sections)
    else:
        raise
    
    instruction = """
                你是一名专业的临床医生，你的任务是针对患者的基础信息与临床信息给出"出院诊断"。
"""
    final_text =  instruction + "\n\n" + prompt
    
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

def drug_classification_map_fn(example, tokenizer, max_seq_length, num_drugs):
    """xtuner format"""
    input_text = build_medical_prompt(example)
    # Tokenize
    encoding = tokenizer(
        input_text,
        truncation=True,
        padding=False,
        max_length=max_seq_length,
        return_tensors=None,
        add_special_tokens=True,
    )
    drug_to_idx, idx_to_drug, drugs = load_drug_vocab(drug_file_path="/data/lzm/DrugRecommend/src/data/pre_drug.json")
    drug_list = example.get("出院带药列表", [])
    labels = create_multihot_labels(drug_list, drug_to_idx, num_drugs)
    return {
        "input_ids": encoding["input_ids"],
        "attention_mask": encoding["attention_mask"],
        "labels": labels.tolist(),  # 多分类标签
        "length": len(encoding["input_ids"])
    }