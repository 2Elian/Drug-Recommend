from datetime import datetime
import pandas as pd
from transformers import AutoTokenizer
from collections import Counter
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from src.worker.tool.tokenizer import Tokenizer


def calculate_age(birth_date: str, visit_date: str) -> int:
    birth = datetime.strptime(birth_date, "%Y-%m")
    visit = datetime.strptime(visit_date, "%Y-%m")
    age = visit.year - birth.year
    if (visit.month, visit.day if hasattr(visit, "day") else 1) < (birth.month, 1):
        age -= 1
    return age

def build_medical_prompt(data: dict) -> str:
    sections = []
    basic_info = []
    gender = data.get("性别")
    if gender is not None:
        basic_info.append(f"性别：{gender}")
    birth_date = data.get("出生日期")
    if birth_date is not None:
        basic_info.append(f"出生日期：{birth_date}")
    bmi = data.get("BMI")
    if bmi is not None:
        bmi_status = "肥胖" if bmi >= 28 else "超重" if bmi >= 24 else "正常"
        basic_info.append(f"BMI：{bmi}（{bmi_status}）")
    ethnicity = data.get("民族")
    if ethnicity is not None:
        basic_info.append(f"民族：{ethnicity}")
    if basic_info:
        sections.append("【患者基本信息】\n" + "，".join(basic_info))
    chief_complaint = data.get("主诉")
    if chief_complaint is not None and chief_complaint.strip():
        sections.append("【主诉】\n" + chief_complaint.strip())
    present_illness = data.get("现病史")
    if present_illness is not None and present_illness.strip():
        sections.append("【现病史】\n" + present_illness.strip())
    admission_status = data.get("入院情况")
    if admission_status is not None and admission_status.strip():
        sections.append("【入院情况】\n" + admission_status.strip())
    process_desc = data.get("诊疗过程描述")
    if process_desc is not None and process_desc.strip():
        clean_process = process_desc.strip()
        sections.append("【诊疗过程】\n" + clean_process)
    
    past_history = data.get("既往史")
    if past_history is not None and past_history.strip():
        sections.append("【既往史】\n" + past_history.strip())
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
            sections.append("【出院诊断】\n" + "，".join(valid_diagnoses))
    if sections:
        prompt = "\n\n".join(sections)
    else:
        raise
    
    instruction = "根据上述患者医疗信息，预测出院时需要带哪些药物："
    final_text = prompt + "\n\n" + instruction
    
    return final_text

def token_count(tokenizer, data):
    fin_text = build_medical_prompt(data)
    return tokenizer.count_tokens(fin_text)
