# 🚀 Qwen3 Inference & Performance Analysis (Prefill / Decode)

> 使用 🤗 Transformers 部署 **Qwen3**，深入理解从输入文本到生成文本的完整推理链路，并系统分析 LLM 推理性能。

---

## 📌 项目简介

本项目基于 **Linux + PyTorch + Transformers**，实现并分析大语言模型（LLM）推理过程中的两个核心阶段：

- ⚡ **Prefill（预填充阶段）**
- 🧠 **Decode（逐 token 解码阶段）**

通过手动拆分推理流程，完成：

- ✅ 从 `model.generate()` 到 `model.forward()` 的机制理解  
- ✅ KV Cache 工作机制解析  
- ✅ TTFT / TBT / Throughput 等性能指标测量  
- ✅ 输入长度 / 输出长度 / Batch Size 对性能影响实验  

---

## 🧠 背景知识（核心概念）

| 阶段 | 计算特征 | 瓶颈 |
|------|----------|------|
| 🚀 Prefill | 并行处理所有输入 token，构建 KV Cache | Compute-bound（算力瓶颈） |
| 🔄 Decode | 每步生成 1 个 token，复用 KV Cache | Memory-bound（显存带宽瓶颈） |

### 📊 关键指标

- **TTFT (Time To First Token)**：首 token 延迟（Prefill 耗时）
- **TBT (Time Between Tokens)**：相邻 token 平均间隔（Decode 单步耗时）
- **Throughput (tokens/s)**：吞吐量
- **Peak Memory (GB)**：显存占用峰值

---

## 🧩 项目结构

```bash
.
├── infer.py              # Prefill / Decode 手动拆分计时
├── benchmark.py          # 系统性能实验（A/B/C）
├── step2_generate.py     # 使用 generate() 的基础推理
├── environment.yml       # Conda 环境
├── README.md
└── 实验报告.md
```
## ⚙️ 环境要求

```bash
Python: 3.13.5
PyTorch: 2.8.0 (CUDA 12.8)
Transformers: 4.57.0
```
## ⚙️ 推荐 GPU

- NVIDIA A100 / H100  
- RTX 3090 / 4090（可运行小模型）

---

## 📦 模型选择（按显存）

| 模型 | 显存 | 特点 |
|------|------|------|
| Qwen3-4B-Instruct-FP8 | ~4.5GB | ⚡ 轻量快速 |
| Qwen3-4B-Instruct | ~8GB | 🧠 综合能力强 |

👉 推荐新手先用 **FP8 版本**

---

## 📥 模型下载

### 方式 1：Hugging Face（推荐）

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-4B-Instruct-2507-FP8"
)
```

默认缓存路径：

```bash
~/.cache/huggingface/hub/
```

---

### 方式 2：ModelScope（国内推荐）

```bash
pip install modelscope
modelscope download --model-id qwen/Qwen3-4B-Instruct-2507 --revision master
```

下载后修改本地路径：

```python
model_path = "/your/local/model"
```

---

## 🚀 快速开始

### 1️⃣ 创建环境

```bash
conda env create -f environment.yml
conda activate mls1
```

---

### 2️⃣ 修改模型路径

在以下文件中修改：

- `infer.py`
- `benchmark.py`
- `step2_generate.py`

```python
model_path = "/path/to/your/model"
```

---

### 3️⃣ 基础推理（Step 2）

```bash
python step2_generate.py
```

你可以调整：

```python
temperature = 1.2
top_p = 0.95
```

---

### 4️⃣ Prefill / Decode 分析（Step 3）

```bash
python infer.py
```

输出包括：

- TTFT（Prefill 时间）  
- TBT（Decode 平均时间）  
- KV Cache 大小变化  

---

### 5️⃣ 性能实验（Step 4）

```bash
python benchmark.py
```

包含三组实验：

---

### 🧪 实验 A：输入长度 → Prefill

- 固定 `output = 128`, `batch = 1`
- 观察：
  - TTFT 随输入增长  
  - Compute-bound 特性  

---

### 🧪 实验 B：输出长度 → Decode

- 固定 `input = 128`
- 观察：
  - TBT 基本稳定  
  - KV Cache 线性增长  

---

### 🧪 实验 C：Batch Size

- 固定 `input / output`
- 观察：
  - Throughput 提升  
  - 显存增加  
  - 可能 OOM  

---

## 🔍 核心机制

### KV Cache

- 存储历史 Key / Value  
- 避免重复计算 Attention  
- Decode 阶段性能核心  

---

### 推理流程

```text
Input Text
   ↓
Tokenizer
   ↓
Prefill
   ↓
KV Cache
   ↓
Decode
   ↓
Output Text
```

---

## 📈 实验结论

- Prefill：Compute-bound  
- Decode：Memory-bound  
- TTFT 随输入长度增加  
- TBT 基本稳定  
- Batch ↑ → Throughput ↑ → 显存 ↑  

---

## ⚠️ 常见问题

### OOM

```python
torch.cuda.empty_cache()
```

---

### 计时不准

```python
torch.cuda.synchronize()
```

---

### HuggingFace 下载慢

建议使用 ModelScope

---

## 📚 参考资料

- Hugging Face  
- ModelScope  
- Qwen 官方文档：https://qwen.readthedocs.io  
