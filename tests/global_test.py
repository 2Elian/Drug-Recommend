import json

with open(r"G:\项目成果打包\出院用药推荐算法研究\src\data\CDrugRed-A-v1\候选药物列表.json", "r", encoding="utf-8") as f:
    data = json.load(f)
    print(len(data))