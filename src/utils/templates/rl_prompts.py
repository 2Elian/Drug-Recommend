RECOMMEND_MODEL_TEMP: str = """
你是一名专业的临床药师，负责根据患者病历信息分析出院携带药物方案。你必须严格遵循预设药物列表，确保所有推荐药物都在许可范围内。
-预设药物列表-
{pre_drug_list}

-步骤-
1. 依据患者性别、年龄、BMI、诊疗过程描述、入院情况、现病史、既往史、主诉以及出院诊断来推荐患者出院所携带的药物信息
2. 基于步骤1识别的药物信息，将每个推荐的药物格式化为("drug"{tuple_delimiter}"你推荐的药物")
3. 以中文返回步骤2中识别出的所有推荐药物的输出列表。使用**{record_delimiter}**作为列表分隔符。
4. 完成后，输出{completion_delimiter}

-真实数据-
【基本信息】
- 性别：{sex}
- 出生日期：{birth_date}
- 民族：{ethnicity}
- BMI：{bmi}
- 就诊时间：{visit_date}
【临床信息】
- 诊疗过程描述：{diagnosis_process}
- 入院情况：{admission_info}
- 现病史：{current_history}
- 既往史：{past_history}
- 主诉：{chief_complaint}
- 出院诊断：{discharge_diagnosis}
"""

POST_VERIFICATION_PROMPT: str = """
你是一名医学博士生，你正在进行医学测试。你将获得患者的临床状况以及该患者出院携带的候选药物。
你的任务是判断该候选药物对患者是否有效且安全。你只能输出"YES"或"NO"，YES代表当前药物对患者安全，NO代表当前药物对患者不安全。

## 当前推荐的候选药物
{init_drug_recommend}

## 该药物详细的描述
{drug_detail}

## 患者临床状况
- 性别：{sex}
- 年龄：{years_old}
- 民族：{ethnicity}
- BMI：{bmi}
- 既往史：{past_history}
- 出院诊断：{discharge_diagnosis}

## 输出要求
**你只能输出“YES”或“NO”，无需解释。
"""


DRUG_RECOMMEND: dict = {
    "PROMPT": {
        "RL_TRAIN_PROMPT": RECOMMEND_MODEL_TEMP,
        "POST_PROCESS_PROMPT": POST_VERIFICATION_PROMPT
    },
    "FORMAT": {
        "tuple_delimiter": "<|>",
        "record_delimiter": "##",
        "completion_delimiter": "<|COMPLETE|>",
    },
}
