"""
Module 5: compare base model and LoRA model inference.

Usage:
    python scripts/my_infer_compare.py --question "高血压患者日常生活中应该注意什么？"
"""

import argparse
import gc
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


SYSTEM_PROMPT = "你是一名谨慎的医疗健康问答助手。回答应清晰、专业，并提醒用户必要时及时就医。"
USER_INSTRUCTION = "请以谨慎、专业、易懂的方式回答下面的医疗健康问题。"

# 这个函数既可以加载：原始模型也可以加载：原始模型 + LoRA
def load_model(model_name, adapter_dir=None):
    """Load the base model, optionally with a LoRA adapter."""
    tokenizer_path = adapter_dir if adapter_dir is not None else model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )

    if adapter_dir is not None:
        model = PeftModel.from_pretrained(model, adapter_dir)

    model.eval()
    return tokenizer, model

# 确认模型在哪个设备上
def _model_device(model):
    """Return a usable device for tokenized inputs."""
    if hasattr(model, "device"):
        return model.device
    return next(model.parameters()).device


def generate(tokenizer, model, question, max_new_tokens=256):
    """Generate an answer for one question."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{USER_INSTRUCTION}\n\n问题：{question}"},
    ]

    try:
        model_inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = tokenizer([prompt], return_tensors="pt")

    device = _model_device(model)
    if isinstance(model_inputs, torch.Tensor):
        model_inputs = {"input_ids": model_inputs.to(device)}
    else:
        model_inputs = model_inputs.to(device)

    # 这里只做推理，不计算梯度。
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9, # 只在概率比较高的一部分候选词里选择
            repetition_penalty=1.1, 
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # 从[问题token][回答token]当中获取回答部分的token
    input_ids = model_inputs["input_ids"]
    new_tokens = generated_ids[:, input_ids.shape[-1] :]
    # 把回答从token转换为中文字符串
    return tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()

# 清理模型和 GPU 显存
def release_model(*objects):
    """Release model references and clear CUDA cache when available."""
    for obj in objects:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="Compare base and LoRA responses.")
    parser.add_argument("--model-name", default="models/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter-dir", type=Path, default=Path("outputs/qwen-medical-lora"))
    parser.add_argument("--question", default="高血压患者日常生活中应该注意什么？")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    # 把命令行当中的参数读取进来
    args = parser.parse_args()

    print("=" * 60)
    print("Base vs LoRA inference comparison")
    print("=" * 60)
    print(f"\nQuestion: {args.question}\n")

    # 先加载base model
    print("[1/2] Loading base model...")
    tokenizer, model = load_model(args.model_name, adapter_dir=None)
    print("Base response:")
    # 使用base model生成回答
    base_response = generate(tokenizer, model, args.question, args.max_new_tokens)
    print(base_response)

    release_model(model, tokenizer)

    # 再加载LoRA model
    # LoRA 并不是重新训练出了一个完整的 Qwen 模型，而是在原模型旁边训练了一小部分额外参数；推理时再把这部分参数挂回原模型上。
    print("\n[2/2] Loading LoRA model...")
    if not args.adapter_dir.exists():
        print(f"LoRA adapter not found: {args.adapter_dir}")
        print("Run scripts/my_train_lora.py first to create the adapter.")
        return

    tokenizer, model = load_model(args.model_name, adapter_dir=args.adapter_dir)
    print("LoRA response:")
    lora_response = generate(tokenizer, model, args.question, args.max_new_tokens)
    print(lora_response)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Base response length: {len(base_response)} chars")
    print(f"LoRA response length: {len(lora_response)} chars")


if __name__ == "__main__":
    main()
