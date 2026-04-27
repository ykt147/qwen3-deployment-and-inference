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
|------|--------|------|
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
