import requests
import trafilatura
import json
from typing import List, Dict, Any
from tqdm import tqdm

def fetch_web_content(url: str) -> str:
    """访问网页并提取正文"""
    try:
        response = requests.get(url, timeout=10, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        })
        downloaded = trafilatura.fetch_url(url)
        text = trafilatura.extract(downloaded)
        return text or ""
    except Exception as e:
        print(f"❌ 抓取 {url} 失败: {e}")
        return ""


def search_and_fetch(api_key: str, drugs: List[str]) -> List[Dict[str, Any]]:
    import http.client
    conn = http.client.HTTPSConnection("google.serper.dev")
    return_des = []
    headers = {'X-API-KEY': api_key, 'Content-Type': 'application/json'}

    for drug in tqdm(drugs, desc="抓取药物信息"):
        try:
            payload = json.dumps({"q": drug, "gl": "cn", "hl": "zh-cn"})
            conn.request("POST", "/search", payload, headers)
            res = conn.getresponse()
            data = res.read()
            search_results = json.loads(data.decode("utf-8"))
        except Exception as e:
            print(f"❌ {drug} 搜索失败: {e}")
            return_des.append({"drug": drug, "des": "搜索失败"})
            continue

        max_count = 2
        success_count = 0
        des = ""
        for item in search_results.get("organic", []):
            if success_count >= max_count:
                break

            link = item.get("link")
            if not link or ".pdf" in link.lower():
                continue

            snippet = item.get("snippet", "")
            content = fetch_web_content(link)
            if not content:
                content = "[网页内容抓取失败]"
            des += snippet + content[:1000]
            success_count += 1

        if not des:  # 如果一条有效内容都没抓到
            des = "[抓取失败或无有效内容]"

        return_des.append({
            "drug": drug,
            "des": des
        })
    return return_des


if __name__ == "__main__":
    json_file = "/data/lzm/DrugRecommend/src/data/pre_drug.json" 
    with open(json_file, "r", encoding="utf-8") as f: 
        drugs = json.load(f)
    api_key = "db9c7d421abe2ed2122ebf34bfdbdefd28e5d169"
    results = search_and_fetch(api_key, drugs)

    output_file = "/data/lzm/DrugRecommend/src/data/pre_drug_mapping.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
