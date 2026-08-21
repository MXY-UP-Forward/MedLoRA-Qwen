"""
Modules 2-4: SFT preprocessing, LoRA configuration, and training.

Usage:
    python scripts/my_train_lora.py
    python scripts/my_train_lora.py --max-train-samples 50 --max-valid-samples 20 --epochs 1 --grad-accum 4 --max-length 256
"""

import argparse
import inspect
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)


IGNORE_INDEX = -100
SYSTEM_PROMPT = "你是一名谨慎的医疗健康问答助手。回答应清晰、专业，并提醒用户必要时及时就医。"
DEFAULT_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def load_jsonl(path):
    """Read a JSONL file."""
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_messages(example):
    """Convert one cleaned SFT item to chat messages."""
    instruction = str(example.get("instruction", "")).strip()
    input_text = str(example.get("input", "")).strip()
    output = str(example.get("output", "")).strip()

    user_content = instruction
    if input_text:
        user_content = f"{user_content}\n\n问题：{input_text}" if user_content else f"问题：{input_text}"

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": output},
    ]

# 统一 tokenizer 输出格式，最终得到 [token1, token2, token3, ...]
def _to_1d_list(token_ids):
    """Normalize tokenizer outputs to a plain one-dimensional Python list."""
    if hasattr(token_ids, "input_ids"):
        token_ids = token_ids.input_ids
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    return list(token_ids)


def preprocess(example, tokenizer, max_length):
    """Tokenize one example and apply assistant-only loss masking."""
    messages = build_messages(example)
    prompt_messages = messages[:-1]
    answer = messages[-1]["content"]

    prompt_ids = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors=None,
    )
    prompt_ids = _to_1d_list(prompt_ids)

    eos_token = tokenizer.eos_token or ""
    answer_ids = tokenizer(
        answer + eos_token,
        add_special_tokens=False,
    )["input_ids"]

    input_ids = prompt_ids + answer_ids
    labels = [IGNORE_INDEX] * len(prompt_ids) + answer_ids

    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]

    attention_mask = [1] * len(input_ids)

    return {
        "input_ids": input_ids, # 模型真正看到的完整文本
        "attention_mask": attention_mask, # 哪些 token 有效
        "labels": labels, # 哪些 token 需要计算 loss
    }


def load_model_and_lora(
    model_name,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=None,
):
    """Load tokenizer/model and attach LoRA adapters."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    # KV Cache 训练阶段关闭推理用的 KV Cache
    model.config.use_cache = False
    model.enable_input_require_grads()

    if target_modules is None:
        target_modules = DEFAULT_TARGET_MODULES

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, # 微调的是一个因果语言模型
        r=r,
        lora_alpha=lora_alpha, # r 控制 LoRA 的容量，alpha 控制 LoRA 更新的强度
        lora_dropout=lora_dropout,
        target_modules=target_modules, # 到底给哪些 Linear 层增加 LoRA
    )
    # 把lora插入到原始模型当中，原来的大模型参数基本全部冻结，只训练新插进去的小型 A、B 矩阵
    model = get_peft_model(model, lora_config)
    # 打印可训练参数数量，确认 LoRA 已经生效
    model.print_trainable_parameters()

    return tokenizer, model


def _training_arguments(**kwargs):
    """Create TrainingArguments across Transformers versions."""
    # 查看当前安装的 Transformers 版本里，TrainingArguments() 到底支持哪些参数。
    params = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" not in params and "eval_strategy" in kwargs:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
    return TrainingArguments(**kwargs)

# 把原始 Python 数据转换成 Hugging Face Dataset，并对每条数据做 preprocess
def _build_dataset(items, tokenizer, max_length):
    if not items:
        raise ValueError("Dataset is empty.")
    # 对数据集里的每一条数据执行一次函数
    return Dataset.from_list(items).map(
        lambda example: preprocess(example, tokenizer, max_length),
        remove_columns=list(items[0].keys()),#preprocess 完之后，把这些原始文本字段删掉
    )


def main():
    parser = argparse.ArgumentParser(description="Train Qwen LoRA on Chinese medical SFT data.")
    parser.add_argument("--model-name", default="models/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--train-file", type=Path, default=Path("data/medical/train.jsonl"))
    parser.add_argument("--valid-file", type=Path, default=Path("data/medical/valid.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/qwen-medical-lora"))
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-valid-samples", type=int, default=0)
    parser.add_argument("--r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument(
        "--target-modules",
        default=",".join(DEFAULT_TARGET_MODULES),
        help="Comma-separated projection module names for LoRA.",
    )
    # 读取参数
    args = parser.parse_args()

    target_modules = [module.strip() for module in args.target_modules.split(",") if module.strip()]
    if not target_modules:
        parser.error("--target-modules must contain at least one module name")

    print(
        f"LoRA configuration: r={args.r}, alpha={args.lora_alpha}, "
        f"scaling={args.lora_alpha / args.r:g}, target_modules={target_modules}"
    )
    tokenizer, model = load_model_and_lora(
        args.model_name,
        r=args.r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
    )

    train_items = load_jsonl(args.train_file)
    valid_items = load_jsonl(args.valid_file)
    if args.max_train_samples > 0:
        train_items = train_items[: args.max_train_samples]
    if args.max_valid_samples > 0:
        valid_items = valid_items[: args.max_valid_samples]

    train_dataset = _build_dataset(train_items, tokenizer, args.max_length)
    valid_dataset = _build_dataset(valid_items, tokenizer, args.max_length)

    training_args = _training_arguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=10,
        save_steps=100,
        eval_steps=100,
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=2,
        bf16=torch.cuda.is_available(),
        fp16=False,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )

    trainer.train()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"LoRA adapter saved to {args.output_dir}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:
        print("=" * 50)
        print("Module 2 test: validate preprocess output")
        print("=" * 50)

        tokenizer = AutoTokenizer.from_pretrained(
            "models/Qwen2.5-0.5B-Instruct",
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        items = load_jsonl(Path("data/medical/train.jsonl"))
        if not items:
            print("Error: data/medical/train.jsonl is empty or missing.")
            print("Please run scripts/my_prepare_data.py first.")
            sys.exit(1)

        result = preprocess(items[0], tokenizer, max_length=256)
        label_ignore_count = result["labels"].count(IGNORE_INDEX)

        print(f"input_ids length: {len(result['input_ids'])}")
        print(f"labels length: {len(result['labels'])}")
        print(f"attention_mask length: {len(result['attention_mask'])}")
        print(f"label ignored tokens: {label_ignore_count}")
        print(f"label trained tokens: {len(result['labels']) - label_ignore_count}")
        print("[OK] preprocess produced aligned input_ids, labels, and attention_mask.")

        print()
        print("=" * 50)
        print("Module 3 test: load model + LoRA")
        print("=" * 50)
        load_model_and_lora("models/Qwen2.5-0.5B-Instruct")
    else:
        main()
