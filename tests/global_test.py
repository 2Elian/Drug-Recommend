from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("/data1/nuist_llm/TrainLLM/ModelCkpt/glm/glm4-8b-chat")
print(model)