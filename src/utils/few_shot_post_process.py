import json
import asyncio
import re
import os
from tqdm.asyncio import tqdm
from src.utils.templates.llm_prompt_baseline import BASELINE_PROMPT
from src.worker.tool.openai_client import OpenAIClient
from src.worker.tool.tokenizer import Tokenizer

SAVE_INTERVAL = 1  # 每处理多少个患者就立即保存一次
SAVE_PATH = "/data/lzm/DrugRecommend/resource/output/submit/glm4_submit.json"

async def verify_single_drug(llm_cli, sem, patient, cur_drug, drug_dict):
    async with sem:
        des = drug_dict.get(cur_drug, "[无药物描述]")
        hint_prompt = BASELINE_PROMPT["POST_V_PROMPT"].format(
            **BASELINE_PROMPT["FORMAT"],
            init_drug_recommend=cur_drug,
            drug_detail=des,
            sex=patient["性别"],
            birth_date=patient["出生日期"],
            ethnicity=patient["民族"],
            bmi=patient["BMI"],
            visit_date=patient["就诊时间"],
            diagnosis_process=patient["诊疗过程描述"],
            admission_info=patient["入院情况"],
            current_history=patient["现病史"],
            past_history=patient["既往史"],
            chief_complaint=patient["主诉"],
            discharge_diagnosis=patient["出院诊断"],
        )

        try:
            result = await llm_cli.generate_answer(hint_prompt)
            result = result.strip().strip('"').strip("'").lower()
        except Exception as e:
            return cur_drug, f"error: {e}"
        return cur_drug, result


async def process_single_patient(llm_cli, sem, patient, pre_drug_set_data, drug_dict):
    patient_id = patient["就诊标识"]
    drug_list = patient["出院带药列表"]

    # step1: 过滤药物
    step1_drug_list = [d for d in drug_list if d in pre_drug_set_data]
    if not step1_drug_list:
        return {"ID": patient_id, "prediction": []}

    filtered = step1_drug_list.copy()
    tasks = [verify_single_drug(llm_cli, sem, patient, d, drug_dict) for d in step1_drug_list]

    for coro in asyncio.as_completed(tasks):
        cur_drug, ans = await coro
        if not ans:
            continue
        match = re.search(r'\b(yes|no)\b', ans)
        if match:
            ans = match.group(1)
        else:
            continue

        if ans == "no" and cur_drug in filtered:
            filtered.remove(cur_drug)

    return {"ID": patient_id, "prediction": filtered}


async def post_process(llm_cli, data_file, pre_drug_set, pre_drug_des, save_json_path, max_concurrent=10):
    sem = asyncio.Semaphore(max_concurrent)

    with open(data_file, "r", encoding="utf-8") as f:
        init_data = json.load(f)
    with open(pre_drug_set, "r", encoding="utf-8") as f:
        pre_drug_set_data = json.load(f)
    with open(pre_drug_des, "r", encoding="utf-8") as f:
        drug_dict = {item["drug"]: item["des"] for item in json.load(f)}

    # -------------------
    # 支持断点续跑
    # -------------------
    completed = {}
    if os.path.exists(save_json_path):
        with open(save_json_path, "r", encoding="utf-8") as f:
            try:
                for record in json.load(f):
                    completed[record["ID"]] = record
            except Exception:
                completed = {}
        print(f"🔁 检测到已完成 {len(completed)} 条记录，将跳过重复患者。")

    results = list(completed.values())
    pending_patients = [p for p in init_data if p["就诊标识"] not in completed]

    # -------------------
    # 主循环
    # -------------------
    count = 0
    for coro in tqdm(asyncio.as_completed([
        process_single_patient(llm_cli, sem, patient, pre_drug_set_data, drug_dict)
        for patient in pending_patients
    ]), total=len(pending_patients), desc="药物验证中", ncols=100):
        result = await coro
        results.append(result)
        count += 1

        # 实时保存
        if count % SAVE_INTERVAL == 0:
            with open(save_json_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            tqdm.write(f"💾 已保存 {len(results)} 条结果至 {save_json_path}")

    # 最终保存
    with open(save_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"✅ 完成，总计 {len(results)} 位患者。结果已保存至：{save_json_path}")


if __name__ == "__main__":
    tokenizer_instance = Tokenizer(
        model_name="/data1/nuist_llm/TrainLLM/ModelCkpt/glm/glm4-8b-chat"
    )

    synthesizer_llm_client = OpenAIClient(
        model_name="glm4-9b",
        api_key="NuistMathAutoModelForCausalLM",
        base_url="http://172.16.107.15:23333/v1",
        tokenizer=tokenizer_instance,
    )

    asyncio.run(post_process(
        synthesizer_llm_client,
        "/data/lzm/DrugRecommend/resource/output/submit/few_shot_glm4.json",
        "/data/lzm/DrugRecommend/src/worker/dataset/pre_drug.json",
        "/data/lzm/DrugRecommend/src/data/pre_drug_mapping.json",
        SAVE_PATH,
        max_concurrent=20  # ⚡ 提高并发
    ))
