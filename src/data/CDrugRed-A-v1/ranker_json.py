# thanks to deepseek-v3.1
import json
from typing import List, Dict, Any

def sort_medical_records(jsonl_file: str, output_file: str) -> None:
    """
    对医疗记录按照就诊标识进行排序
    
    Args:
        jsonl_file: 输入的jsonl文件路径
        output_file: 输出的jsonl文件路径
    """
    
    def extract_sort_key(record: Dict[str, Any]) -> tuple:
        """
        从记录中提取排序键值
        
        Args:
            record: 单条医疗记录
            
        Returns:
            排序键值元组 (n, k)
        """
        visit_id = record.get("就诊标识", "0-0")
        try:
            n, k = map(int, visit_id.split('-'))
            return (n, k)
        except (ValueError, AttributeError):
            # 如果格式不正确，返回(0,0)放在最前面
            return (0, 0)
    records = []
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                record = json.loads(line.strip())
                records.append(record)
    sorted_records = sorted(records, key=extract_sort_key)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for record in sorted_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"排序完成！共处理 {len(sorted_records)} 条记录")
    print(f"输出文件: {output_file}")


# 使用示例
if __name__ == "__main__":
    input_file = "/data/lzm/DrugRecommend/src/data/CDrugRed-A-v1/CDrugRed_predict-val.jsonl"  # 你的输入文件
    output_file = "/data/lzm/DrugRecommend/src/data/CDrugRed-A-v1/jsonlsorted_medical_records.jsonl"  # 输出文件
    
    sort_medical_records(input_file, output_file)
