# MiniLoRA 医疗问答 LoRA 微调实验报告

## 1. 实验目的

本实验基于 Qwen2.5-0.5B-Instruct，在中文医疗健康问答数据上进行 LoRA 微调，并通过多组消融实验分析不同训练数据量、LoRA rank、LoRA alpha 以及 target modules 对训练效果的影响。

实验结果来源于 `res.csv`，主要评价指标为：

- `train_loss`：训练集损失，用于观察模型对训练数据的拟合程度。
- `eval_loss`：验证集损失，用于衡量模型在未见数据上的泛化效果。

其中，`eval_loss` 越低，通常表示模型验证效果越好。

## 2. 实验设置

所有实验均使用相同的基础模型和主要训练参数：

| 参数 | 设置 |
|---|---|
| 基础模型 | Qwen2.5-0.5B-Instruct |
| 任务类型 | 中文医疗健康问答 SFT |
| max_length | 512 |
| epochs | 1 |
| grad_accum | 8 |
| learning_rate | 2e-4 |
| LoRA dropout | 默认配置 |

实验主要改变以下变量：

- 训练样本数：200 或 640
- LoRA rank：r=4、r=8、r=16
- LoRA alpha：8、16、32
- target modules：全部模块、仅 q_proj、q_proj+k_proj+v_proj

## 3. 实验结果

| 实验名 | r | lora_alpha | target_modules | 训练样本数 | train_loss | eval_loss | 输出目录 |
|---|---:|---:|---|---:|---:|---:|---|
| baseline_lora | 8 | 16 | all | 640 | 2.746 | 2.789 | outputs/qwen-medical-lora |
| r4_ablation | 4 | 16 | all | 200 | 2.699 | 2.875 | outputs/lora-r4 |
| r8_ablation | 8 | 16 | all | 200 | 2.700 | 2.881 | outputs/lora-r8 |
| r16_ablation | 16 | 16 | all | 200 | 2.702 | 2.873 | outputs/lora-r16 |
| alpha8_ablation | 8 | 8 | all | 200 | 2.718 | 2.854 | outputs/alpha-8 |
| alpha16_ablation | 8 | 16 | all | 200 | 2.700 | 2.884 | outputs/alpha-16 |
| alpha32_ablation | 8 | 32 | all | 200 | 2.693 | 2.882 | outputs/alpha-32 |
| target_q_ablation | 8 | 16 | q_proj | 200 | 2.822 | 3.624 | outputs/target-q |
| target_qkv_ablation | 8 | 16 | q_proj+k_proj+v_proj | 200 | 2.761 | 3.221 | outputs/target-qkv |

## 4. 结果分析

### 4.1 数据量对效果的影响

`baseline_lora` 使用 640 条训练样本，eval_loss 为 2.789，是全部实验中最低的验证损失。

与 200 条样本下的 r8 消融实验相比：

| 实验 | 训练样本数 | eval_loss |
|---|---:|---:|
| baseline_lora | 640 | 2.789 |
| r8_ablation | 200 | 2.881 |

当训练样本数从 200 增加到 640 后，eval_loss 从 2.881 降低到 2.789，下降了 0.092。

这说明在当前实验规模下，增加训练数据量能够明显改善验证集表现。相比单纯调整 LoRA 超参数，数据量对最终效果的影响更稳定。

### 4.2 LoRA rank 消融实验

在训练样本数均为 200、lora_alpha 均为 16、target_modules 均为 all 的条件下，对比不同 rank：

| rank | train_loss | eval_loss |
|---:|---:|---:|
| 4 | 2.699 | 2.875 |
| 8 | 2.700 | 2.881 |
| 16 | 2.702 | 2.873 |

三组实验的 eval_loss 最大差距为 0.008，差异非常小。其中 r=16 的 eval_loss 最低，为 2.873，但相比 r=4 仅降低 0.002。

因此，本实验中增大 LoRA rank 并没有带来显著收益。说明在 200 条训练样本下，模型效果主要受数据规模限制，而不是 LoRA rank 容量限制。

从效率角度看，r=4 或 r=8 已经足够；从默认配置的稳定性看，r=8 是较合理的折中选择。

### 4.3 LoRA alpha 消融实验

在训练样本数均为 200、rank 均为 8、target_modules 均为 all 的条件下，对比不同 lora_alpha：

| lora_alpha | train_loss | eval_loss |
|---:|---:|---:|
| 8 | 2.718 | 2.854 |
| 16 | 2.700 | 2.884 |
| 32 | 2.693 | 2.882 |

从训练损失看，alpha 越大，train_loss 越低：

- alpha=8：train_loss 2.718
- alpha=16：train_loss 2.700
- alpha=32：train_loss 2.693

但从验证损失看，alpha=8 的 eval_loss 最低，为 2.854，优于 alpha=16 和 alpha=32。

这说明较大的 alpha 可能增强了模型对训练集的拟合能力，但没有改善验证集表现，甚至可能带来轻微过拟合。当前数据规模较小时，较温和的 alpha=8 反而泛化更好。

### 4.4 target_modules 消融实验

在训练样本数均为 200、rank 均为 8、lora_alpha 均为 16 的条件下，对比不同 target modules：

| target_modules | train_loss | eval_loss |
|---|---:|---:|
| all | 2.700 | 2.881 |
| q_proj | 2.822 | 3.624 |
| q_proj+k_proj+v_proj | 2.761 | 3.221 |

结果显示，只训练 `q_proj` 时效果最差，eval_loss 达到 3.624；扩展到 `q_proj+k_proj+v_proj` 后有所改善，eval_loss 降至 3.221，但仍明显差于 all 配置。

这说明对于该医疗问答 SFT 任务，仅调整注意力中的部分投影层不足以获得良好效果。将 LoRA 应用于更多模块，包括注意力输出层和 FFN 层，能够显著提升模型的适配能力。

因此，本项目中推荐继续使用 `all` target modules 配置。

## 5. 综合结论

本次实验表明，LoRA 微调能够在较小参数更新量的情况下完成中文医疗问答场景适配。综合所有实验结果，可以得到以下结论：

1. 增加训练数据量是提升效果最直接的方式。640 条样本训练得到的 baseline_lora 取得最低 eval_loss，为 2.789。
2. 在 200 条样本下，rank=4、rank=8、rank=16 的差异很小，说明当前瓶颈不在 LoRA rank。
3. alpha=8 在验证集上表现最好，说明较小 alpha 在小数据场景下可能具有更好的泛化能力。
4. target_modules 对结果影响显著。仅训练 q_proj 或 qkv 投影层效果明显不足，all 配置表现更好。
5. 最推荐的默认配置是 r=8、lora_alpha=16、target_modules=all；如果只考虑 200 条样本下的 eval_loss，可尝试 r=8、lora_alpha=8、target_modules=all。

## 6. 推荐配置

### 6.1 正式训练推荐

```powershell
python scripts\my_train_lora.py --r 8 --output-dir outputs/qwen-medical-lora --epochs 1 --batch-size 1 --grad-accum 8 --max-length 512
```

该配置使用全部训练数据，最终 eval_loss 为 2.789，是本次实验中的最佳结果。

### 6.2 小数据实验推荐

```powershell
python scripts\my_train_lora.py --r 8 --output-dir outputs/alpha-8 --max-train-samples 200 --max-valid-samples 50 --epochs 1 --max-length 512
```

在 200 条训练样本的实验中，alpha=8 的 eval_loss 最低，为 2.854。

## 7. 后续改进方向

后续如果希望进一步提升模型效果，可以优先考虑：

1. 增加训练数据量，例如从 640 条扩展到数千条或更多。
2. 增加训练轮数，并观察 eval_loss 是否继续下降。
3. 对医疗问答数据进行质量筛选，去除过短、重复或低质量回答。
4. 引入人工评价或自动指标，对 base_answer 和 lora_answer 进行生成质量对比。
5. 在更大模型上重复实验，例如 Qwen2.5-1.5B 或 7B，以观察模型规模带来的提升。

