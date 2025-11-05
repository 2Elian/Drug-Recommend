#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/11/1 21:50
# @Author  : lizimo@nuist.edu.cn
# @File    : datatypes.py
# @Description:
import math
from dataclasses import dataclass, field
from typing import List, Union, Dict, Any

@dataclass
class DataType:
    """
    {
    "患者序号": 2,
    "就诊标识": "2-1",
    "性别": "女",
    "出生日期": "1940-12",
    "民族": "汉族",
    "BMI": 27.3,
    "就诊时间": "2015-03",
    "诊疗过程描述": "门诊查尿酮体：+，白细胞：250/ul、镜下：7-9/Hp。入院后查：尿常规:白细胞+++/ul、尿白细胞62.25/HP↑,口服补液后复查尿常规白细胞及酮体（-）。馒头餐试验结果回报：糖化血红蛋白...... ",
    "入院情况": "患者以\"烦渴、多饮、多尿5年，尿痛伴血糖控制不佳2个月。\"为主诉入院重要查体：T36.6℃，P76次/分，R22次/分，BP160/80mmHg......",
    "现病史": "患者5年前无明显诱因出现烦渴、多饮、多尿症状，遂于某医院就诊，测空腹血糖16.7mmol/l,予患者二甲双胍、瑞格列奈、阿卡波糖片控制血糖...... ",
    "既往史": "否认冠心病病史，否认有肝炎、结核、疟疾等传染病史，否认食物、药物过敏史，否认外伤、手术史，否认输血史，预防接种史不详。",
    "主诉": "烦渴、多饮、多尿5年，尿痛伴血糖控制不佳2个月。",
    "出院诊断": ["2型糖尿病", "糖尿病酮症", "泌尿系感染", "糖尿病大血管病变",...]
    "出院带药列表": [ "阿卡波糖", "瑞格列奈", "瑞舒伐他汀", "替米沙坦", "氨氯地平片", "碳酸钙片" ]
    }

    """
    id: int  # 患者序号或唯一标识
    patient_id: str  # 就诊标识
    sex: str  # 性别
    birth_date: str  # 出生日期
    ethnicity: str  # 民族
    bmi: float  # BMI
    visit_date: str  # 就诊时间
    diagnosis_process: str  # 诊疗过程描述
    admission_info: str  # 入院情况
    current_history: str  # 现病史
    past_history: str  # 既往史
    chief_complaint: str  # 主诉
    discharge_diagnosis: List[str] = field(default_factory=list)  # 出院诊断

@dataclass
class OutputFormat:
    id: int  # 患者序号或唯一标识
    patient_id: str  # 就诊标识
    sex: str  # 性别
    birth_date: str  # 出生日期
    ethnicity: str  # 民族
    bmi: float  # BMI
    visit_date: str  # 就诊时间
    diagnosis_process: str  # 诊疗过程描述
    admission_info: str  # 入院情况
    current_history: str  # 现病史
    past_history: str  # 既往史
    chief_complaint: str  # 主诉
    discharge_diagnosis: List[str] = field(default_factory=list)  # 出院诊断
    predict_list: List[str] = field(default_factory=list)  # 出院带药列表

@dataclass
class Token:
    text: str
    prob: float
    top_candidates: List = field(default_factory=list)
    ppl: Union[float, None] = field(default=None)

    @property
    def logprob(self) -> float:
        return math.log(self.prob)