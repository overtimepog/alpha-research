from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-Coder-32B-Instruct')

print(model)