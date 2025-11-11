test_text = "这是一个很长的测试文本，用来验证tokenizer的截断行为，看看是从左边开始截断还是从右边开始截断"
from src.worker.tool.bert_tokenizer import init_tokenizer
tokenizer = init_tokenizer()
# 左截断
tokenizer.truncation_side = "left"
result_left = tokenizer(
    test_text, 
    truncation=True, 
    max_length=10,
    return_tensors="pt"
)

result_right = tokenizer(
    test_text, 
    truncation=True, 
    max_length=10,
    truncation_side = "right",
    return_tensors="pt"
)

print("左截断:", tokenizer.decode(result_left.input_ids[0]))
print("右截断:", tokenizer.decode(result_right.input_ids[0]))