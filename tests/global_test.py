import json

def jsonl_to_json(jsonl_path: str, json_path: str):
    """
    将 JSONL 文件转换为 JSON 文件

    Args:
        jsonl_path: 输入 JSONL 文件路径
        json_path: 输出 JSON 文件路径
    """
    data_list = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:  # 忽略空行
                try:
                    data = json.loads(line)
                    data_list.append(data)
                except json.JSONDecodeError as e:
                    print(f"解析出错：{line}\n错误信息：{e}")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data_list, f, ensure_ascii=False, indent=4)

    print(f"已成功将 {jsonl_path} 转换为 {json_path}，共 {len(data_list)} 条数据。")


if __name__ == "__main__":
    # 示例用法
    jsonl_to_json(r"G:\github仓库管理\Drug-Recommend\src\data\CDrugRed-A-v1\CDrugRed_val_split_train_18present.jsonl", "./output.json")
