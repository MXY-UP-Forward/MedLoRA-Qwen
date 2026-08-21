"""
数据准备脚本：从 shibing624/medical 原始数据生成训练用的 jsonl 文件

原始数据位置: data/medical_raw/finetune/train_zh_0.json
输出位置:       data/medical/train.jsonl / valid.jsonl / test.jsonl

运行方式:
    python scripts/prepare_medical_sft.py --include-train --max-samples 1000

原始数据格式（Alpaca 格式，每行一个 JSON）:
    {"instruction": "血热的临床表现是什么?", "input": "", "output": "初发或复发病不久..."}
    {"instruction": "帕金森叠加综合征的辅助治疗有些什么？", "input": "", "output": "综合治疗；康复训练..."}
    ...

说明:
    - instruction: 问题或指令（有的为空）
    - input: 补充输入（这批数据中基本都为空）
    - output: 回答（必须非空，否则丢弃）
"""

import argparse
import json
import random
from pathlib import Path
from typing import Iterable


# ============================================================
# 函数 1: 读取 jsonl 文件
# ============================================================

def read_jsonl(path: Path) -> list[dict]:
    """读取 jsonl 文件，返回 list[dict]

    jsonl 格式: 每行一个独立的 JSON 对象，不是整个文件一个 JSON
    例如:
        {"instruction": "问题1", "output": "回答1"}
        {"instruction": "问题2", "output": "回答2"}

    和普通 JSON 的区别:
    - JSON: 整个文件是一个数组或对象，用逗号分隔元素
    - jsonl: 每行都是完整的 JSON，不需要逗号，适合大数据流式处理
    """
    items: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()           # 去掉每行末尾的换行符和空白
            if not line:                   # 跳过空行
                continue
            try:
                items.append(json.loads(line))  # 把 JSON 字符串解析成 Python 字典
            except json.JSONDecodeError:
                continue                   # 如果某行 JSON 格式有误，跳过而不是报错
    return items


# ============================================================
# 函数 2: 清洗单条数据
# ============================================================

def normalize_item(item: dict) -> dict | None:
    """清洗一条原始数据，返回统一格式；如果数据无效则返回 None

    原始数据的问题:
    - instruction 有的是空的
    - input 有的是空的
    - output 有的是空的（这种直接丢弃）
    - instruction 内容各不相同，需要统一成一个固定的前缀

    为什么要统一 instruction？
    - 原始数据的 instruction 五花八门："血热的临床表现是什么?"、"帕金森叠加综合征的..."
    - 如果不统一，模型会学到这些不同的指令风格，训练目标不一致
    - 统一后，所有数据都用同一个指令，模型只学习"怎么回答医疗问题"这个任务

    清洗后的统一格式:
    {
        "instruction": "请以谨慎、专业、易懂的方式回答下面的医疗健康问题。",
        "input": "具体问题内容",
        "output": "回答内容"
    }
    """
    # 获取三个字段，用 str() 保证类型安全，.strip() 去掉首尾空白
    # 为什么要 str() 包一层？原始数据可能有 int、None 等非字符串值
    instruction = str(item.get("instruction", "")).strip()
    input_text = str(item.get("input", "")).strip()
    output = str(item.get("output", "")).strip()

    # --- 过滤规则 1: output 为空 → 丢弃 ---
    # 没有回答的数据对 SFT（监督微调）没用
    # SFT 的核心是：给定问题，学习生成回答。没有回答就没法学
    if not output:
        return None

    # --- 构造 "问题" 内容 ---
    # 原始数据有三种情况:
    #   情况 A: instruction="血热的临床表现是什么?", input=""
    #   情况 B: instruction="", input="具体问题"
    #   情况 C: instruction="请回答", input="具体问题"
    # 我们需要把 "问题" 统一放到 input 字段中
    if input_text:
        # 情况 B 或 C: input 不为空
        if instruction:
            # 情况 C: instruction 和 input 都有，拼起来作为完整问题
            prompt = f"{instruction}\n{input_text}"
        else:
            # 情况 B: 只有 input，直接用它作为问题
            prompt = input_text
    else:
        # 情况 A: 只有 instruction，用它作为问题
        prompt = instruction

    # --- 过滤规则 2: 问题为空 → 丢弃 ---
    # 如果 instruction 和 input 都是空的，没有问题就没法训练
    if not prompt:
        return None

    # --- 返回统一格式 ---
    # instruction 统一替换成固定的系统指令
    # 这样所有数据的 instruction 都一样，模型学到的是对"医疗问题"的回答风格
    # 而不是学各种不同的"指令措辞"
    return {
        "instruction": "请以谨慎、专业、易懂的方式回答下面的医疗健康问题。",
        "input": prompt,
        "output": output,
    }


# ============================================================
# 函数 3: 找到可用的原始数据文件
# ============================================================

def load_available_raw_files(raw_dir: Path, include_train: bool) -> list[Path]:
    """根据参数决定加载哪些原始文件

    数据集有三个文件:
    - train_zh_0.json: 训练集（最大，主要数据来源）
    - valid_zh_0.json: 验证集
    - test_zh_0.json:  测试集

    为什么有 include_train 参数？
    - train_zh_0.json 文件很大（百万级条目），全量读取需要较长时间
    - 测试流程时可以先用 valid + test（数据量小），确认流程没问题后再用 train
    """
    finetune_dir = raw_dir / "finetune"
    preferred = []
    if include_train:
        preferred.append(finetune_dir / "train_zh_0.json")
    preferred.extend([
        finetune_dir / "valid_zh_0.json",
        finetune_dir / "test_zh_0.json",
    ])
    # 只返回确实存在的 .json 文件，避免 FileNotFoundError
    return [path for path in preferred if path.exists() and path.suffix == ".json"]


# ============================================================
# 函数 4: 写出 jsonl 文件
# ============================================================

def write_jsonl(path: Path, items: Iterable[dict]) -> None:
    """将数据写入 jsonl 文件

    jsonl = JSON Lines，每行一个 JSON 对象
    ensure_ascii=False: 中文不转义成 \\uXXXX

    为什么用 jsonl 而不是 JSON？
    - 流式读取：可以逐行读，不需要一次性加载整个文件到内存
    - 追加友好：可以在文件末尾追加新数据
    - 适合训练：HuggingFace datasets 库原生支持 jsonl
    """
    path.parent.mkdir(parents=True, exist_ok=True)  # 如果目录不存在就创建
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            # json.dumps 把字典转成 JSON 字符串
            # ensure_ascii=False 保证中文正常显示，不会变成 中文
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


# ============================================================
# 主函数：串联以上所有步骤
# ============================================================

def main() -> None:
    # --- 命令行参数 ---
    parser = argparse.ArgumentParser(description="Prepare Chinese medical SFT jsonl data.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/medical_raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/medical"))
    parser.add_argument("--max-samples", type=int, default=5000,
                        help="训练集最大条数")
    parser.add_argument("--valid-size", type=int, default=300,
                        help="验证集条数")
    parser.add_argument("--test-size", type=int, default=300,
                        help="测试集条数")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子（保证可复现）")
    parser.add_argument("--include-train", action="store_true",
                        help="使用 train_zh_0.json（数据量大）")
    args = parser.parse_args()

    # --- 步骤 1: 找到可用的原始文件 ---
    raw_files = load_available_raw_files(args.raw_dir, args.include_train)
    if not raw_files:
        raise FileNotFoundError(f"找不到原始数据: {args.raw_dir / 'finetune'}")

    print("Using raw files:")
    for path in raw_files:
        print(f"  - {path}")

    # --- 步骤 2: 读取所有原始数据 ---
    # 把多个文件的数据合并到一个列表里
    # extend 和 append 的区别:
    #   append: 把整个列表作为一个元素加入 → [1,2] + [[3,4]] = [1,2,[3,4]]
    #   extend: 把列表中的每个元素分别加入 → [1,2] + [3,4] = [1,2,3,4]
    raw_items: list[dict] = []
    for path in raw_files:
        raw_items.extend(read_jsonl(path))
    # 此时 raw_items 包含所有原始数据，格式不统一

    # --- 步骤 3: 清洗 ---
    # 用列表推导式: 对每条数据调用 normalize_item，过滤掉 None
    # 等价于:
    #   normalized = []
    #   for item in raw_items:
    #       result = normalize_item(item)
    #       if result is not None:
    #           normalized.append(result)
    # 两层推导式:
    #   (normalize_item(item) for item in raw_items) → 生成器，逐个处理
    #   [item for item in 生成器 if item is not None] → 过滤掉 None
    normalized = [item for item in (normalize_item(item) for item in raw_items) if item is not None]

    # --- 步骤 4: 打乱顺序 ---
    # 设随机种子保证每次运行结果一样（可复现）
    # 为什么打乱？原始数据可能按疾病类别排列（前 1000 条全是内科）
    # 打乱后 train/valid/test 的疾病分布更均匀
    random.seed(args.seed)
    random.shuffle(normalized)

    # --- 步骤 5: 限制总数据量 ---
    # 总量 = max_samples + valid_size + test_size
    # 但不能超过实际数据量（min 防止越界）
    total_limit = min(args.max_samples + args.valid_size + args.test_size, len(normalized))
    selected = normalized[:total_limit]

    # --- 步骤 6: 划分 train / valid / test ---
    # 划分逻辑:
    #   test_size = min(要求的test数, 总数的1/5)
    #   valid_size = min(要求数, 剩余的1/5)
    #   train_size = 剩下的全给 train
    # 结果大致是: train 80%, valid 10%, test 10%
    # 为什么这样划分？
    #   - train: 用来训练模型参数（需要最多数据）
    #   - valid: 训练中每隔 N 步评估一次，用来监控是否过拟合
    #   - test: 训练结束后最终评估，训练过程中绝不使用（防止数据泄漏）
    test_size = min(args.test_size, max(0, len(selected) // 5))
    valid_size = min(args.valid_size, max(0, (len(selected) - test_size) // 5))
    train_size = len(selected) - valid_size - test_size

    if train_size <= 0:
        raise ValueError("数据太少，无法划分训练集")

    # 切片取数据
    # 列表切片 [start:end]，end 是开区间（不包含）
    train_data = selected[:train_size]                                    # 前 train_size 条
    valid_data = selected[train_size : train_size + valid_size]           # 接着 valid_size 条
    test_data = selected[train_size + valid_size :]                       # 剩余的

    # --- 步骤 7: 写出文件 ---
    write_jsonl(args.out_dir / "train.jsonl", train_data)
    write_jsonl(args.out_dir / "valid.jsonl", valid_data)
    write_jsonl(args.out_dir / "test.jsonl", test_data)

    print(f"Saved train: {len(train_data)} -> {args.out_dir / 'train.jsonl'}")
    print(f"Saved valid: {len(valid_data)} -> {args.out_dir / 'valid.jsonl'}")
    print(f"Saved test : {len(test_data)} -> {args.out_dir / 'test.jsonl'}")


if __name__ == "__main__":
    main()
