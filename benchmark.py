import torch
import gc
import time
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

#配置区域
MODEL_NAME = "path/to/Qwen3-4B-Instruct-2507"  # 推荐用于测试的轻量模型，或者 "meta-llama/Llama-2-7b-hf"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WARMUP_STEPS = 2  # 预热次数

print(f"当前设备: {DEVICE}")
print(f"正在加载模型: {MODEL_NAME} ...")

# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map=DEVICE)

# 设置 pad_token 避免报错
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def get_gpu_memory():
    """获取当前显存占用 (GB)"""
    if DEVICE == "cpu": return 0
    return torch.cuda.memory_allocated(DEVICE) / 1024**3

def clear_memory():
    """清理显存"""
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

def run_prefill(input_ids, past_key_values=None):
    """
    手动执行 Prefill 阶段
    返回: (logits, new_past_key_values, duration_ms)
    """
    start_time = time.time()
    with torch.no_grad():
        outputs = model(input_ids, past_key_values=past_key_values, use_cache=True)
    # 同步 CUDA 操作以确保时间准确
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    duration_ms = (time.time() - start_time) * 1000
    return outputs.logits, outputs.past_key_values, duration_ms

def run_decode_step(next_token, past_key_values):
    """
    手动执行单步 Decode
    返回: (logits, new_past_key_values, duration_ms)
    """
    start_time = time.time()
    with torch.no_grad():
        outputs = model(next_token, past_key_values=past_key_values, use_cache=True)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    duration_ms = (time.time() - start_time) * 1000
    return outputs.logits, outputs.past_key_values, duration_ms

#预热
print("\n🔥 正在进行预热 (Warmup)...")
dummy_input = torch.randint(0, 1000, (1, 32)).to(DEVICE)
with torch.no_grad():
    model(dummy_input)
    # 简单 generate 一次
    model.generate(dummy_input, max_new_tokens=10, pad_token_id=tokenizer.eos_token_id)
print("预热完成。")
clear_memory()


print(f"{'Input Len':<12} | {'TTFT (ms)':<12} | {'Throughput (tok/s)':<20} | {'VRAM (GB)':<10}")
print("-" * 65)

# 扩展长度范围，小长度很难测出 O(N^2) 规律
input_lens = [8192, 16384, 32768, 65536] 
results_a = []

for input_len in input_lens:
    clear_memory()
    # 构造输入
    input_ids = torch.randint(100, 5000, (1, input_len)).to(DEVICE)
    
    # 使用 CUDA Event 进行精准计时
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    # 预热一次当前长度，防止第一次编译开销
    with torch.no_grad():
        model(input_ids)
    
    # 正式计时
    start_event.record()
    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)
    end_event.record()
    
    # 等待 GPU 完成
    torch.cuda.synchronize()
    ttft = start_event.elapsed_time(end_event) # 毫秒
    
    vram_used = get_gpu_memory()
    throughput = input_len / (ttft / 1000)
    
    print(f"{input_len:<12} | {ttft:<12.2f} | {throughput:<20.2f} | {vram_used:<10.2f}")
    results_a.append([input_len, ttft, throughput, vram_used])
    
# 输出长度对 Decode 的影响

print(f"{'Output Len':<12} | {'TBT Mean (ms)':<14} | {'Throughput (tok/s)':<20} | {'KV Inc (GB)':<12}")
print("-" * 70)

output_lens = [64, 256, 512, 1024]
results_b = []

# 固定输入长度
fixed_input_len = 128
input_ids = torch.randint(100, 5000, (1, fixed_input_len)).to(DEVICE)

for output_len in output_lens:
    clear_memory()
    
    # 1. Prefill (固定)
    _, past_kv, _ = run_prefill(input_ids)
    vram_after_prefill = get_gpu_memory()
    
    # 2. Decode 循环
    current_token = input_ids[:, -1:] # 取最后一个token作为起始（模拟）或者随机一个
    # 为了严谨，我们重新从prefill拿到的logits采样一个，这里简化直接用随机token演示decode耗时
    # 实际应使用上一轮的logits采样，这里为了测速简化处理
    next_token = torch.randint(100, 5000, (1, 1)).to(DEVICE)
    
    tbt_list = []
    
    for _ in range(output_len):
        _, past_kv, step_time = run_decode_step(next_token, past_kv)
        tbt_list.append(step_time)
        # 模拟生成下一个token (实际推理中这里需要采样)
        next_token = torch.randint(100, 5000, (1, 1)).to(DEVICE)

    # 计算指标
    mean_tbt = sum(tbt_list) / len(tbt_list)
    decode_throughput = output_len / sum(tbt_list) * 1000 # ms -> s
    
    vram_after_decode = get_gpu_memory()
    kv_cache_increment = vram_after_decode - vram_after_prefill
    
    print(f"{output_len:<12} | {mean_tbt:<14.2f} | {decode_throughput:<20.2f} | {kv_cache_increment:<12.4f}")
    results_b.append([output_len, mean_tbt, decode_throughput, kv_cache_increment])

df_b = pd.DataFrame(results_b, columns=["Output Length", "TBT Mean (ms)", "Decode Throughput (tok/s)", "KV Cache Inc (GB)"])

# Batch Size 的影响
print(f"{'Batch Size':<12} | {'TTFT (ms)':<12} | {'Throughput (tok/s)':<20} | {'VRAM (GB)':<10}")
print("-" * 65)

batch_sizes = [1, 4, 8, 16]
results_c = []

# 固定输入输出
fixed_input_len = 128
fixed_output_len = 128

for bs in batch_sizes:
    clear_memory()
    try:
        # 构造输入
        input_ids = torch.randint(100, 5000, (bs, fixed_input_len)).to(DEVICE)
        
        # 1. Prefill
        _, past_kv, ttft = run_prefill(input_ids)
        vram_used = get_gpu_memory()
        
        # 2. Decode (模拟生成 fixed_output_len 个token)
        # 注意：这里为了测量整体吞吐量，我们跑完整个序列
        next_tokens = torch.randint(100, 5000, (bs, 1)).to(DEVICE)
        total_decode_time = 0
        
        for _ in range(fixed_output_len):
            _, past_kv, step_time = run_decode_step(next_tokens, past_kv)
            total_decode_time += step_time
            next_tokens = torch.randint(100, 5000, (bs, 1)).to(DEVICE)
            
        # 计算整体吞吐量 (Total Tokens / Total Time)
        # Total Tokens = Batch * (Input + Output) 或者仅计算 Output
        # 这里计算 Decode Throughput (Output tokens)
        total_time_s = (ttft + total_decode_time) / 1000
        throughput = (bs * fixed_output_len) / total_time_s
        
        print(f"{bs:<12} | {ttft:<12.2f} | {throughput:<20.2f} | {vram_used:<10.2f}")
        results_c.append([bs, ttft, throughput, vram_used])
        
    except RuntimeError as e:
        print(f"{bs:<12} | {'OOM/Error':<12} | {'N/A':<20} | {'N/A':<10}")
        print(f"Error: {e}")
        results_c.append([bs, float('inf'), 0, float('inf')])

df_c = pd.DataFrame(results_c, columns=["Batch Size", "TTFT (ms)", "Throughput (tok/s)", "VRAM (GB)"])

print("\n" + "="*30)
print("数据已生成在 DataFrame df_a, df_b, df_c 中。")
