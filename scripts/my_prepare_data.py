"""
Module 1: prepare medical SFT data.

Read the raw shibing624/medical JSONL files and write cleaned train/valid/test
JSONL files under data/medical/.

Usage:
    python scripts/my_prepare_data.py
"""

import argparse
import json
from pathlib import Path


DEFAULT_INSTRUCTION = "请以谨慎、专业、易懂的方式回答下面的医疗健康问题。"


def read_jsonl(path):
    """Read a JSONL file and return a list of dictionaries."""
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def normalize_item(item):
    """Clean one raw Alpaca-style item.

    Returns a normalized dict, or None when the item cannot be used for SFT.
    """
    instruction = str(item.get("instruction", "")).strip()
    input_text = str(item.get("input", "")).strip()
    output = str(item.get("output", "")).strip()

    if not output:
        return None

    if input_text:
        prompt = f"{instruction}\n{input_text}" if instruction else input_text
    else:
        prompt = instruction

    if not prompt:
        return None

    return {
        "instruction": DEFAULT_INSTRUCTION,
        "input": prompt,
        "output": output,
    }


def write_jsonl(path, items):
    """Write items to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def clean_split(path):
    """Read, normalize, and drop invalid rows from one split."""
    return [item for item in (normalize_item(i) for i in read_jsonl(path)) if item is not None]


def main():
    parser = argparse.ArgumentParser(description="Prepare medical SFT data from raw JSON files.")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/medical_raw"),
        help="Raw data directory",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/medical"),
        help="Output directory for processed data",
    )
    args = parser.parse_args()

    finetune_dir = args.raw_dir / "finetune"
    train_path = finetune_dir / "train_zh_0.json"
    valid_path = finetune_dir / "valid_zh_0.json"
    test_path = finetune_dir / "test_zh_0.json"

    required_paths = [train_path, valid_path, test_path]
    missing_paths = [path for path in required_paths if not path.exists()]
    if missing_paths:
        print(f"原始数据不存在: {finetune_dir}")
        print("缺少文件:")
        for path in missing_paths:
            print(f"  - {path}")
        print("请先运行 python download_dataset.py 下载数据集")
        return

    print("读取原始文件: train_zh_0.json, valid_zh_0.json, test_zh_0.json")

    train_data = clean_split(train_path)
    valid_data = clean_split(valid_path)
    test_data = clean_split(test_path)

    print(f"清洗后: train={len(train_data)}, valid={len(valid_data)}, test={len(test_data)}")

    write_jsonl(args.out_dir / "train.jsonl", train_data)
    write_jsonl(args.out_dir / "valid.jsonl", valid_data)
    write_jsonl(args.out_dir / "test.jsonl", test_data)

    print(f"\n输出到 {args.out_dir}/")
    print(f"  train.jsonl: {len(train_data)} 条")
    print(f"  valid.jsonl: {len(valid_data)} 条")
    print(f"  test.jsonl:  {len(test_data)} 条")


if __name__ == "__main__":
    main()
