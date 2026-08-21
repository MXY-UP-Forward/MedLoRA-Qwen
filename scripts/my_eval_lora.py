"""
模块 6：批量评测 eval_lora.py

目标：批量跑 base vs LoRA 对比，保存结果到 results/base_vs_lora.jsonl

运行方式：
    python scripts/my_eval_lora.py --model-name models/Qwen2.5-0.5B-Instruct

本模块没有参考代码，需要你自己写。
可以复用模块 5 的 load_model 和 generate 函数。
"""

import argparse
import gc
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM_PROMPT = "你是一名谨慎的医疗健康问答助手。回答应清晰、专业，并提醒用户必要时及时就医。"
USER_INSTRUCTION = "请以谨慎、专业、易懂的方式回答下面的医疗健康问题。"


def load_model(model_name, adapter_dir=None):
    """加载模型（复用模块 5）

    adapter_dir=None  → 加载 base 模型
    adapter_dir="路径" → 加载 base + LoRA adapter
    """
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


def _model_device(model):
    """返回模型所在设备，用于把输入张量移动到同一设备。"""
    if hasattr(model, "device"):
        return model.device
    return next(model.parameters()).device


def generate(tokenizer, model, question, max_new_tokens=256):
    """用模型生成回答（复用模块 5）"""
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

    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    input_ids = model_inputs["input_ids"]
    new_tokens = generated_ids[:, input_ids.shape[-1] :]
    return tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()


def load_jsonl(path):
    """读取 jsonl 文件，返回 list[dict]"""
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def write_jsonl(path, items):
    """写出 jsonl 文件，ensure_ascii=False 保证中文不被转义"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


# 读取一批问题 → 先让 Base 模型全部回答 → 再让 LoRA 模型全部回答 → 把两组结果保存到 JSONL 文件。
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="models/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter-dir", type=Path, default=Path("outputs/qwen-medical-lora"))
    parser.add_argument("--eval-file", type=Path, default=Path("data/medical/eval_prompts.jsonl"))
    parser.add_argument("--output-file", type=Path, default=Path("results/base_vs_lora.jsonl"))
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    if not args.eval_file.exists():
        print(f"评测文件不存在: {args.eval_file}")
        print("请先创建 data/medical/eval_prompts.jsonl")
        print("每行一个 JSON: {\"question\": \"你的问题\"}")
        return

    # 读取评测数据
    eval_items = load_jsonl(args.eval_file)
    questions = [item["question"] for item in eval_items]

    print(f"读取评测问题: {len(questions)} 条")
    print("[1/2] 加载 base 模型并生成回答...")
    tokenizer, model = load_model(args.model_name, adapter_dir=None)
    results = []
    for index, question in enumerate(questions, start=1):
        print(f"  Base [{index}/{len(questions)}] {question}")
        results.append(
            {
                "question": question,
                "base_answer": generate(tokenizer, model, question, args.max_new_tokens),
            }
        )
    # 释放，接下来还要加载 LoRA 模型，要给显存腾位置
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not args.adapter_dir.exists():
        print(f"LoRA adapter 不存在: {args.adapter_dir}")
        print("请先运行 scripts/my_train_lora.py 生成 LoRA adapter")
        return

    print("[2/2] 加载 LoRA 模型并生成回答...")
    tokenizer, model = load_model(args.model_name, adapter_dir=args.adapter_dir)
    for index, result in enumerate(results, start=1):
        question = result["question"]
        print(f"  LoRA [{index}/{len(results)}] {question}")
        result["lora_answer"] = generate(tokenizer, model, question, args.max_new_tokens)

    write_jsonl(args.output_file, results)
    print(f"评测结果已保存到: {args.output_file}")

if __name__ == "__main__":
    main()
