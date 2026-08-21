"""
Base vs LoRA 推理对比脚本：比较原始模型和微调模型在医疗问答上的差异

功能:
    1. 加载 base 模型（原始 Qwen），生成回答
    2. 加载 LoRA adapter（微调后的），生成回答
    3. 对比两者输出的差异

运行方式:
    python scripts/infer_compare.py --question "高血压患者日常生活中应该注意什么？"
    python scripts/infer_compare.py --model-name models/Qwen2.5-0.5B-Instruct --question "感冒怎么办？"

核心概念:
    - Base 模型: 通用 Qwen，没有针对医疗领域微调
    - LoRA 模型: 同一个 Qwen + 训练好的 LoRA adapter，医疗领域更专业
    - 对比目的: 验证微调是否让模型回答更专业、更有条理
"""

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
# 系统提示词：定义模型的角色和行为准则
# ============================================================

SYSTEM_PROMPT = "你是一名谨慎的医疗健康问答助手。回答应清晰、专业，并提醒用户必要时及时就医。"
# 为什么需要 system prompt？
# - 设定模型的角色（医疗助手），让回答风格一致
# - Qwen 是 instruct 模型，会根据 system prompt 调整回答风格
# - 在医疗场景中，"谨慎"很重要：不能过度诊断，要建议就医


# ============================================================
# 函数 1: 加载模型
# ============================================================

def load_model(model_name: str, adapter_dir: Path | None):
    """加载模型，可选择是否挂载 LoRA adapter

    参数:
        model_name: 基础模型路径（如 models/Qwen2.5-1.5B-Instruct）
        adapter_dir: LoRA adapter 目录（如果为 None，加载 base 模型）

    返回:
        (tokenizer, model) 元组

    加载 base 模型 vs LoRA 模型的区别:
    - Base: load_model("models/Qwen2.5-1.5B-Instruct", None)
    - LoRA: load_model("models/Qwen2.5-1.5B-Instruct", Path("outputs/qwen-medical-lora"))
    """
    # --- 步骤 1: 加载 tokenizer ---
    # 如果有 adapter_dir，优先从 adapter 目录加载 tokenizer
    # 因为训练时 tokenizer.save_pretrained() 保存到了 adapter 目录
    # 如果没有 adapter，就从基础模型目录加载
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir or model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Qwen 没有 pad_token，用 eos 代替

    # --- 步骤 2: 加载基础模型 ---
    # 注意：这里加载的总是同一个基础模型
    # LoRA 的差异在于后续是否挂载 adapter
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )

    # --- 步骤 3: 挂载 LoRA adapter（如果有）---
    # PeftModel.from_pretrained 做了什么？
    # 1. 加载 adapter_config.json（读取 LoRA 配置：rank, alpha, target_modules 等）
    # 2. 在基础模型对应层后面插入 LoRA 层（A 和 B 矩阵）
    # 3. 加载 adapter_model.bin（训练好的 LoRA 权重）
    #
    # 关键点：
    # - 基础模型的权重仍然被冻结（不更新）
    # - 只有 LoRA 层的参数被加载
    # - 前向传播时：h = Wx + (alpha/r) * BAx
    if adapter_dir is not None:
        model = PeftModel.from_pretrained(model, adapter_dir)

    # --- 步骤 4: 设置为评估模式 ---
    # model.eval() 的作用：
    # 1. 关闭 dropout（推理时不应该随机丢弃神经元）
    # 2. 关闭 batch norm 的 running stats 更新（虽然因果模型一般不用 BN）
    # 3. 告诉模型"现在是推理阶段，不需要计算梯度"
    model.eval()
    return tokenizer, model


# ============================================================
# 函数 2: 生成回答
# ============================================================

def generate(tokenizer, model, question: str, max_new_tokens: int) -> str:
    """基于问题生成回答

    参数:
        tokenizer: 分词器
        model: 模型（base 或 LoRA）
        question: 用户问题
        max_new_tokens: 最多生成多少新 token

    返回:
        生成的回答字符串

    生成流程:
        构造 messages → tokenize → model.generate() → decode
    """
    # --- 步骤 1: 构造对话消息 ---
    # 推理时只需要 system + user，不需要 assistant
    # 因为 assistant 的回答是要生成的，不是给定的
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"请以谨慎、专业、易懂的方式回答下面的医疗健康问题。\n\n问题：{question}"},
    ]

    # --- 步骤 2: 应用 chat template 并 tokenize ---
    # apply_chat_template 会把 messages 转成带特殊标记的 token 序列
    # 例如 Qwen 会生成类似这样的结构：
    # <|im_start|>system\n你是一名...<|im_end|>\n
    # <|im_start|>user\n问题...<|im_end|>\n
    # <|im_start|>assistant\n
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # --- 步骤 3: 生成回答 ---
    # model.generate() 参数说明:
    # - max_new_tokens: 最多生成多少个新 token（不包括 prompt）
    # - do_sample=True: 启用采样，让回答更有多样性
    # - temperature=0.7: 控制随机性，越低越确定，越高越随机
    # - top_p=0.9: nucleus sampling，只从累积概率 > 0.9 的 token 中采样
    # - repetition_penalty=1.1: 重复惩罚，>1 降低重复内容
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
    )

    # --- 步骤 4: 解码 ---
    # 只解码新生成的部分（去掉 prompt 部分）
    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response


# ============================================================
# 函数 3: 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Base vs LoRA 推理对比")
    parser.add_argument(
        "--model-name",
        default="models/Qwen2.5-0.5B-Instruct",
        help="基础模型路径",
    )
    parser.add_argument(
        "--adapter-dir",
        type=Path,
        default=Path("outputs/qwen-medical-lora"),
        help="LoRA adapter 目录",
    )
    parser.add_argument(
        "--question",
        type=str,
        default="高血压患者日常生活中应该注意什么？",
        help="要提问的问题",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="最多生成的新 token 数",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Base vs LoRA 推理对比")
    print("=" * 60)
    print(f"\n问题: {args.question}\n")

    # --- 加载 Base 模型 ---
    print("[1/2] 加载 Base 模型...")
    tokenizer_base, model_base = load_model(args.model_name, adapter_dir=None)
    print("  Base 模型加载完成\n")

    # --- Base 模型生成 ---
    print("  Base 模型回答:")
    base_response = generate(tokenizer_base, model_base, args.question, args.max_new_tokens)
    print(f"  {base_response}\n")

    # --- 加载 LoRA 模型 ---
    print("[2/2] 加载 LoRA 模型...")
    if not args.adapter_dir.exists():
        print(f"  ⚠️  LoRA adapter 不存在: {args.adapter_dir}")
        print("  请先运行训练脚本生成 LoRA adapter")
        return
    tokenizer_lora, model_lora = load_model(args.model_name, args.adapter_dir)
    print("  LoRA 模型加载完成\n")

    # --- LoRA 模型生成 ---
    print("  LoRA 模型回答:")
    lora_response = generate(tokenizer_lora, model_lora, args.question, args.max_new_tokens)
    print(f"  {lora_response}\n")

    # --- 对比总结 ---
    print("=" * 60)
    print("对比总结")
    print("=" * 60)
    print(f"\nBase 模型输出长度: {len(base_response)} 字符")
    print(f"LoRA 模型输出长度: {len(lora_response)} 字符")
    print("\n观察点:")
    print("  1. LoRA 模型的回答是否更专业？")
    print("  2. LoRA 模型的回答是否更有条理？")
    print("  3. LoRA 模型是否更倾向于建议就医？")


if __name__ == "__main__":
    main()