
BASELINE_TEMP_ZH: str = """
<|user|>你作为三甲医院主任医师，需根据患者完整病历和药物列表，严格遵循临床规范生成出院带药方案。病历如下：
【患者核心信息】
    - 基础体征：{sex}/{age}岁（注意年龄相关剂量调整）
    - 生理指标： BMI{bmi},{bmi_des}
【临床诊疗关键点】
    - 诊疗过程描述：{process}
    - 入院情况：{admission}
    - 主诉：{complaint}
    - 现病史：{history_now}
    - 既往史：{history_past}（关注慢性病用药史）
    - 出院诊断：{diagnosis}（核心依据）
【待选药物库】{drug_str}
【输出规范】
请从给定的药物列表中推荐该患者的 “出院带药列表”,
请仅输出以下格式："药物A,药物B,药物C,……"
所推荐药物必须在待选药物库中。严格按照格式回答，不要输出任何解释或额外内容。
"""

BASELINE_TEMP_EN: str = """
You are a chief physician at a top-tier hospital. Based on the patient’s complete medical record and medication list, you must
strictly follow clinical guidelines to generate a discharge medication plan. The medical record is as follows:
[Patient Core Information]
    - Basic vitals: {sex}/{age} years old (pay attention to age-related dose adjustments)
    - Physiological indicators: Height {height} cm, Weight {weight} kg, BMI {bmi}
[Key Clinical Points]
    - Description of treatment process: {process}
    - Admission condition: {admission}
    - Chief complaint: {complaint}
    - Present illness history: {history_now}
    - Past medical history: {history_past} (pay attention to chronic medication history)
    - Discharge diagnosis: {diagnosis} (core basis)
[Medication Library for Selection] {drug_str}
[Output Specification]
Please recommend this patient’s “discharge medication list” from the given medication options.
Only output in the following format:
“DrugA, DrugB, DrugC, ..."
"""


COT_TEMP_ZH: str = """
分析+给出推荐药物
"""

COT_TEMP_EN: str = """
分析+给出推荐药物
"""


BASELINE_PROMPT: dict = {
    "ZH": {
        "BASE_TEMPLATE": BASELINE_TEMP_ZH,
        "COT_TEMPLATE": COT_TEMP_ZH,
    },
    "EN": {
        "BASE_TEMPLATE": BASELINE_TEMP_EN,
        "COT_TEMPLATE": COT_TEMP_EN,
    }
}