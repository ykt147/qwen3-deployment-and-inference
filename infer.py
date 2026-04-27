import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM


def get_kv_cache_size(past_key_values):
    total_bytes = 0
    for layer in past_key_values:
        for tensor in layer:
            total_bytes += tensor.numel() * tensor.element_size()
    return total_bytes

model_path = "/public/home/ykt147/model/Qwen3-4B-Instruct-2507"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype=torch.float16,
    device_map="cuda"
)
model.eval()


prompt = "Explain KV cache in LLM inference."

# 预热3次，减少冷启动影响
warmup_inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    for _ in range(3):
        _ = model.forward(
            input_ids=warmup_inputs["input_ids"],
            use_cache=True
        )

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

input_ids = inputs["input_ids"]

# ======================
#  Prefill
# ======================
torch.cuda.synchronize()
start = time.time()

with torch.no_grad():
    outputs = model.forward(
        input_ids=input_ids,
        use_cache=True
    )

torch.cuda.synchronize()
prefill_time = time.time() - start

past_key_values = outputs.past_key_values
next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)

print(f"TTFT (Prefill time): {prefill_time * 1000:.2f} ms")

kv_size = get_kv_cache_size(past_key_values)
print(f"KV Cache size: {kv_size / 1024 / 1024:.2f} MB")

# ======================
#   Decode
# ======================
decode_times = []
generated = [next_token]

for _ in range(127):  # output_len = 128
    torch.cuda.synchronize()
    start = time.time()

    with torch.no_grad():
        outputs = model.forward(
            input_ids=next_token,
            past_key_values=past_key_values,
            use_cache=True
        )

    torch.cuda.synchronize()
    step_time = time.time() - start
    decode_times.append(step_time)

    past_key_values = outputs.past_key_values
    next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
    generated.append(next_token)

tbt = sum(decode_times) / len(decode_times)

print(f"TBT (avg decode time): {tbt * 1000:.2f} ms")

kv_size = get_kv_cache_size(past_key_values)
print(f"KV Cache size(after decode): {kv_size / 1024 / 1024:.2f} MB")
# 拼接输出
generated_ids = torch.cat(generated, dim=1)
print(tokenizer.decode(generated_ids[0]))