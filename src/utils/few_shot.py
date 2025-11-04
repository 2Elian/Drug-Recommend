#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/11/1 21:46
# @Author  : lizimo@nuist.edu.cn
# @File    : few_shot.py.py
# @Description:
import json
import re
import sys
import asyncio
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from dataclasses import asdict
from typing import Dict, List, Tuple

from src.models.bases.datatypes import DataType, OutputFormat
from src.utils.templates.llm_prompt_baseline import BASELINE_PROMPT, BASELINE_PROMPT_3
from src.models.bases.base_llm_client import BaseLLMClient
from src.models.llm.openai_client import OpenAIClient
from src.models.llm.tokenizer import Tokenizer
from src.utils.helper import calculate_age


class Generator:
    def __init__(
        self,
        llm_client: BaseLLMClient = None,
        tokenizer_instance: Tokenizer = None,
        max_loop: int = 3,
        pre_drug_list: List[str] = None,
        pre_drug_mapping: List[Dict[str, str]] = None
    ) -> None:
        self.llm_client = llm_client
        self.tokenizer_instance = tokenizer_instance
        self.max_loop = max_loop
        self.pre_drug_list = pre_drug_list or []
        self.pre_drug_mapping = pre_drug_mapping or []
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger("DrugRecommend-Generator")
        self.logger.setLevel(logging.INFO)
    async def generate(self, data_type: DataType) -> OutputFormat:
        id = data_type.id
        patient_id = data_type.patient_id
        sex = data_type.sex
        birth_date = data_type.birth_date
        ethnicity = data_type.ethnicity
        bmi = data_type.bmi
        visit_date = data_type.visit_date
        diagnosis_process = data_type.diagnosis_process
        admission_info = data_type.admission_info
        current_history = data_type.current_history
        past_history = data_type.past_history
        chief_complaint = data_type.chief_complaint
        discharge_diagnosis = data_type.discharge_diagnosis


        # step 1: structure prompt
        hint_prompt = BASELINE_PROMPT["Chinese"]["TEMPLATE"].format(
            **BASELINE_PROMPT["FORMAT"], pre_drug_list=self.pre_drug_list,
            sex=sex, birth_date=birth_date, ethnicity=ethnicity, bmi=bmi,
            visit_date=visit_date, diagnosis_process=diagnosis_process,
            admission_info=admission_info, current_history=current_history,
            past_history=past_history, chief_complaint=chief_complaint, discharge_diagnosis=discharge_diagnosis
        )
        self.logger.info(f"init_prompt: token number is {self.tokenizer_instance.count_tokens(hint_prompt)}")

        # step 2: initial glean
        final_result = await self.llm_client.generate_answer(hint_prompt)
        # self.logger.info("Init Drug Result: %s", final_result)

        # step3: iterative refinement
        history = pack_history_conversations(hint_prompt, final_result)
        for loop_idx in range(self.max_loop):
            if_loop_result = await self.llm_client.generate_answer(
                text=BASELINE_PROMPT["Chinese"]["IF_LOOP"], history=history
            )
            if_loop_result = if_loop_result.strip().strip('"').strip("'").lower()
            if if_loop_result != "yes": # if output is no, it represents no entity leaked
                break
            glean_result = await self.llm_client.generate_answer(
                text=BASELINE_PROMPT["Chinese"]["CONTINUE"], history=history
            )
            print("Loop %s glean: %s", loop_idx + 1, glean_result)

            history += pack_history_conversations(
                BASELINE_PROMPT["Chinese"]["CONTINUE"], glean_result
            )
            final_result += glean_result
        # step 4: parse the result
        init_drug_recommend = parse_result(final_result)
        self.logger.info(f"init_drug_recommend: {init_drug_recommend}")
        # step5:
        drug_dict = {item["drug"]: item["des"] for item in self.pre_drug_mapping}
        drug_detail = [
                            {"drug": drug, "des": drug_dict.get(drug, "[无描述]")}
                            for drug in init_drug_recommend
                        ]
        finally_check_prompt = BASELINE_PROMPT["FIN_PROMPT"].format(
            **BASELINE_PROMPT["FORMAT"], init_drug_recommend=init_drug_recommend, drug_detail=drug_detail, 
            sex=sex, years_old=calculate_age(birth_date, visit_date), ethnicity=ethnicity, bmi=bmi,
            past_history=past_history, discharge_diagnosis=discharge_diagnosis)
        check_result = await self.llm_client.generate_answer(finally_check_prompt)
        # step6: parse finally result to OutputFormat
        predict_list = parse_result(check_result)
        self.logger.info(f"predict_list: {predict_list}")
        predict = OutputFormat(id=id, patient_id=patient_id, sex=sex, birth_date=birth_date, ethnicity=ethnicity,
                               bmi=bmi, visit_date=visit_date, diagnosis_process=diagnosis_process, admission_info=admission_info,
                               current_history=current_history, past_history=past_history, chief_complaint=chief_complaint, discharge_diagnosis=discharge_diagnosis,
                               predict_list=predict_list)
        return predict


def pack_history_conversations(*args: str):
    roles = ["user", "assistant"]
    return [
        {"role": roles[i % 2], "content": content} for i, content in enumerate(args)
    ]

def parse_result(result_text: str):
    result_text = result_text.strip()
    result_text = result_text.replace("<|COMPLETE|>", "").strip()
    pattern = r'\("drug"\s*<\|>\s*"([^"]+)"\)'
    drugs = re.findall(pattern, result_text)
    drugs = [d.strip() for d in drugs if d.strip()]
    drugs = list(dict.fromkeys(drugs))

    return drugs

FIELD_MAP = {
    "id": "患者序号",
    "patient_id": "就诊标识",
    "sex": "性别",
    "birth_date": "出生日期",
    "ethnicity": "民族",
    "bmi": "BMI",
    "visit_date": "就诊时间",
    "diagnosis_process": "诊疗过程描述",
    "admission_info": "入院情况",
    "current_history": "现病史",
    "past_history": "既往史",
    "chief_complaint": "主诉",
    "discharge_diagnosis": "出院诊断",
    "predict_list": "出院带药列表"
}

async def main():
    tokenizer_instance: Tokenizer = Tokenizer(
        model_name="/data1/nuist_llm/TrainLLM/ModelCkpt/glm/glm4-8b-chat"
    )

    synthesizer_llm_client: OpenAIClient = OpenAIClient(
        model_name="/data1/nuist_llm/TrainLLM/ModelCkpt/glm/glm4-8b-chat",
        api_key="dummy",
        base_url="http://172.16.107.15:23333/v1",
        tokenizer=tokenizer_instance,
    )
    with open("/data/lzm/DrugRecommend/src/data/pre_drug_mapping.json", "r", encoding="utf-8") as g:
        pre_drug_mapping = json.load(g)
    with open("/data/lzm/DrugRecommend/src/data/pre_drug.json", "r", encoding="utf-8") as p:
        pre_drug_list = json.load(p)
    generator_instance: Generator = Generator(
        llm_client=synthesizer_llm_client,
        tokenizer_instance=tokenizer_instance, 
        max_loop=3,
        pre_drug_list=pre_drug_list,
        pre_drug_mapping=pre_drug_mapping
    )
    output_file = r"/data/lzm/DrugRecommend/src/data/CDrugRed-A-v1/CDrugRed_predict-val.jsonl"
    data_path = r"/data/lzm/DrugRecommend/src/data/CDrugRed-A-v1/CDrugRed_val_split_train_18present.jsonl"
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                data_type = DataType(
                    id=data["患者序号"],
                    patient_id=data["就诊标识"],
                    sex=data["性别"],
                    birth_date=data["出生日期"],
                    ethnicity=data["民族"],
                    bmi=data["BMI"],
                    visit_date=data["就诊时间"],
                    diagnosis_process=data["诊疗过程描述"],
                    admission_info=data["入院情况"],
                    current_history=data["现病史"],
                    past_history=data["既往史"],
                    chief_complaint=data["主诉"],
                    discharge_diagnosis=data["出院诊断"],
                )
                result: OutputFormat = await generator_instance.generate(data_type)
                result_dict = asdict(result)
                result_dict_zh = {FIELD_MAP[k]: v for k, v in result_dict.items()}
                with open(output_file, "a", encoding="utf-8") as out_f:
                    out_f.write(json.dumps(result_dict_zh, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    asyncio.run(main())