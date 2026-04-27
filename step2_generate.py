# infer_generate.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "path/to/Qwen3-4B-Instruct-2507"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)

prompt = "Explain KV cache in LLM inference."

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=128,
    temperature=1.2,
    top_p=0.95
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
