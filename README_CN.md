# MiniLoRA：Qwen2.5 医疗问答 LoRA 微调实验

[English README](README.md)

本项目基于 **Qwen2.5-0.5B-Instruct**，使用 **LoRA SFT** 方法在中文医疗健康问答数据上进行微调，完整覆盖数据准备、监督微调、推理对比、批量评测和消融实验。

当前仓库已经补全 `scripts/my_*.py` 中的主要代码，并根据 `res.csv` 整理了实验结果。完整实验分析见 [EXPERIMENT_REPORT.md](EXPERIMENT_REPORT.md)。

## 1. 项目目标

本项目希望验证：在较小规模中文医疗问答数据上，LoRA 是否能让通用指令模型更适合医疗健康问答场景。

整体流程如下：

```text
原始医疗问答数据
  -> JSONL 清洗
  -> SFT messages 格式构造
  -> assistant-only loss mask
  -> LoRA 微调
  -> base 与 LoRA 单条推理对比
  -> 批量评测
  -> 消融实验分析
```

## 2. 项目信息

| 项目 | 内容 |
|---|---|
| 基础模型 | Qwen2.5-0.5B-Instruct |
| 微调方法 | LoRA SFT |
| 任务类型 | 中文医疗健康问答 |
| 正式训练样本数 | 640 |
| 验证集样本数 | 160 |
| 测试集样本数 | 200 |
| 批量评测问题数 | 10 |
| 主要 LoRA 输出目录 | `outputs/qwen-medical-lora` |
| 实验结果记录 | `res.csv` |
| 实验报告 | `EXPERIMENT_REPORT.md` |

## 3. 项目结构

```text
MiniLoRA-master/
├── README.md
├── README_CN.md
├── EXPERIMENT_REPORT.md
├── res.csv
├── requirements.txt
├── download_dataset.py
├── scripts/
│   ├── prepare_medical_sft.py
│   ├── train_lora.py
│   ├── infer_compare.py
│   ├── my_prepare_data.py
│   ├── my_train_lora.py
│   ├── my_infer_compare.py
│   └── my_eval_lora.py
├── data/
│   ├── medical/
│   │   ├── train.jsonl
│   │   ├── valid.jsonl
│   │   ├── test.jsonl
│   │   └── eval_prompts.jsonl
│   └── medical_raw/
├── models/
│   └── Qwen2.5-0.5B-Instruct/
├── outputs/
│   ├── qwen-medical-lora/
│   ├── lora-r4/
│   ├── lora-r8/
│   ├── lora-r16/
│   ├── alpha-8/
│   ├── alpha-16/
│   ├── alpha-32/
│   ├── target-q/
│   └── target-qkv/
└── results/
    └── base_vs_lora.jsonl
```

## 4. 脚本说明

| 模块 | 脚本 | 作用 |
|---|---|---|
| 数据准备 | `scripts/my_prepare_data.py` | 读取原始数据，清洗并保存为 JSONL |
| SFT 预处理 | `scripts/my_train_lora.py` | 构造 messages，tokenize，并生成 labels |
| LoRA 训练 | `scripts/my_train_lora.py` | 加载模型，挂载 LoRA，使用 Trainer 训练 |
| 单条推理对比 | `scripts/my_infer_compare.py` | 对比 base 模型和 LoRA 模型回答 |
| 批量评测 | `scripts/my_eval_lora.py` | 对评测问题批量生成 base 与 LoRA 回答 |
| 参考实现 | `scripts/` 下无 `my_` 前缀脚本 | 项目提供的参考代码 |

## 5. 环境配置

推荐环境：

- Python 3.10+
- PyTorch 2.1+
- Transformers 4.45+
- PEFT 0.12+
- NVIDIA GPU，建议 6GB 以上显存

安装依赖：

```powershell
pip install -r requirements.txt
```

检查 CUDA 是否可用：

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

如果输出 `True`，说明可以使用 GPU 训练。

## 6. 数据与模型准备

项目默认从下面路径加载基础模型：

```text
models/Qwen2.5-0.5B-Instruct
```

数据准备命令：

```powershell
python download_dataset.py
python scripts\my_prepare_data.py
```

处理后会生成：

```text
data/medical/train.jsonl
data/medical/valid.jsonl
data/medical/test.jsonl
data/medical/eval_prompts.jsonl
```

## 7. 训练方法

### 7.1 快速测试

先用少量样本确认环境、模型和代码都能跑通：

```powershell
python scripts\my_train_lora.py --max-train-samples 50 --max-valid-samples 20 --epochs 1 --grad-accum 4 --max-length 256
```

### 7.2 正式训练

正式实验使用 640 条训练样本：

```powershell
python scripts\my_train_lora.py --r 8 --output-dir outputs/qwen-medical-lora --epochs 1 --batch-size 1 --grad-accum 8 --max-length 512
```

训练完成后，LoRA adapter 会保存到：

```text
outputs/qwen-medical-lora
```

## 8. 推理与批量评测

单条问题推理对比：

```powershell
python scripts\my_infer_compare.py --question "高血压患者日常生活中应该注意什么？"
```

批量评测：

```powershell
python scripts\my_eval_lora.py
```

批量评测结果保存到：

```text
results/base_vs_lora.jsonl
```

每一行格式如下：

```json
{
  "question": "...",
  "base_answer": "...",
  "lora_answer": "..."
}
```

## 9. 实验结果

以下结果来自 [res.csv](res.csv)。

| 实验名 | r | alpha | target modules | 训练样本数 | train_loss | eval_loss |
|---|---:|---:|---|---:|---:|---:|
| baseline_lora | 8 | 16 | all | 640 | 2.746 | 2.789 |
| r4_ablation | 4 | 16 | all | 200 | 2.699 | 2.875 |
| r8_ablation | 8 | 16 | all | 200 | 2.700 | 2.881 |
| r16_ablation | 16 | 16 | all | 200 | 2.702 | 2.873 |
| alpha8_ablation | 8 | 8 | all | 200 | 2.718 | 2.854 |
| alpha16_ablation | 8 | 16 | all | 200 | 2.700 | 2.884 |
| alpha32_ablation | 8 | 32 | all | 200 | 2.693 | 2.882 |
| target_q_ablation | 8 | 16 | q_proj | 200 | 2.822 | 3.624 |
| target_qkv_ablation | 8 | 16 | q_proj+k_proj+v_proj | 200 | 2.761 | 3.221 |

## 10. 实验结论

1. **全量训练效果最好。** 使用 640 条训练样本的 `baseline_lora` 取得最低验证损失，`eval_loss = 2.789`。
2. **LoRA rank 的影响较小。** 在 200 条样本下，r=4、r=8、r=16 的 eval_loss 最大差距只有 0.008，说明当前主要瓶颈不是 LoRA 容量。
3. **alpha=8 在小数据实验中表现最好。** 在 200 条样本、r=8、target_modules=all 条件下，alpha=8 的 eval_loss 为 2.854，优于 alpha=16 和 alpha=32。
4. **target_modules 对结果影响明显。** 仅训练 `q_proj` 或 `q_proj+k_proj+v_proj` 的效果明显差于 all 配置。
5. **推荐默认配置。** 综合稳定性和效果，推荐使用 r=8、lora_alpha=16、target_modules=all；如果只在 200 条小数据上实验，可以尝试 alpha=8。

更详细的分析见：

```text
EXPERIMENT_REPORT.md
```

## 11. 核心实现说明

### 11.1 SFT 数据格式

原始数据：

```json
{
  "instruction": "请以谨慎、专业、易懂的方式回答下面的医疗健康问题。",
  "input": "高血压患者日常生活中应注意什么？",
  "output": "高血压患者应注意低盐饮食、规律运动、监测血压..."
}
```

训练时转换为 messages：

```python
[
    {"role": "system", "content": "你是一名谨慎的医疗健康问答助手。回答应清晰、专业，并提醒用户必要时及时就医。"},
    {"role": "user", "content": "请以谨慎、专业、易懂的方式回答下面的医疗健康问题。\n\n问题：高血压患者日常生活中应注意什么？"},
    {"role": "assistant", "content": "高血压患者应注意低盐饮食、规律运动、监测血压..."},
]
```

### 11.2 assistant-only loss mask

训练 labels 的构造方式：

```text
input_ids: prompt tokens + answer tokens
labels:    -100 tokens    + answer token ids
```

这样模型只在 assistant 回答部分计算 loss，不学习复述 system prompt 和用户问题。

### 11.3 LoRA 配置

主实验配置：

```text
r = 8
lora_alpha = 16
target_modules = q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
```

LoRA 只训练新增的低秩矩阵，基础模型参数保持冻结。

### 11.4 显存管理

推理对比时，脚本先加载 base 模型生成回答，再释放显存，然后加载 LoRA 模型生成回答。这样可以避免 6GB 显存同时放不下两个模型的问题。

## 12. 推荐运行流程

```powershell
pip install -r requirements.txt
python scripts\my_prepare_data.py
python scripts\my_train_lora.py --r 8 --output-dir outputs/qwen-medical-lora --epochs 1 --batch-size 1 --grad-accum 8 --max-length 512
python scripts\my_infer_compare.py --question "高血压患者日常生活中应该注意什么？"
python scripts\my_eval_lora.py
```

## 13. 后续改进方向

- 扩大训练数据规模，例如增加到数千条或更多。
- 增加训练轮数，并观察 eval_loss 是否继续下降。
- 对训练数据进行质量筛选，去除重复、过短或低质量回答。
- 增加自动评价指标，例如 ROUGE、BERTScore。
- 增加人工评价维度，例如专业性、清晰度、安全性。
- 尝试更大的基础模型，例如 Qwen2.5-1.5B 或 Qwen2.5-7B。
- 引入 RAG，让模型回答时结合可靠医学资料。
