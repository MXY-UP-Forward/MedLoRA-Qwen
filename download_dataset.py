import os
import json
import random
import time

# 国内用户可设置 HF_ENDPOINT=https://hf-mirror.com 使用镜像加速
# 默认使用 HuggingFace 官方源
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://huggingface.co"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

from huggingface_hub import hf_hub_download

endpoint = os.environ["HF_ENDPOINT"]
print(f"下载 shibing624/medical 数据集...")
print(f"源: {endpoint}")

repo_id = "shibing624/medical"

files_to_download = [
    ("finetune/train_zh_0.json", "训练集"),
    ("finetune/valid_zh_0.json", "验证集"),
    ("finetune/test_zh_0.json", "测试集"),
]

downloaded_files = []
for filename, desc in files_to_download:
    print(f"\n下载 {desc}: {filename}...")
    for attempt in range(3):
        try:
            fpath = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="dataset",
                local_dir="data/medical_raw",
                force_download=True,
            )
            print(f"  下载成功: {fpath}")
            downloaded_files.append((fpath, desc))
            break
        except Exception as e:
            print(f"  第 {attempt+1} 次尝试失败: {e}")
            if attempt < 2:
                print(f"  等待 5 秒后重试...")
                time.sleep(5)
            else:
                print(f"  {desc} 下载失败，跳过")

if not downloaded_files:
    print("\n所有文件下载失败！")
    exit(1)

print("\n读取并处理数据...")
all_data = []
for fpath, desc in downloaded_files:
    print(f"读取 {desc}: {fpath}...")
    with open(fpath, "r", encoding="utf-8") as f:
        content = f.read().strip()
        if not content:
            continue
        # 兼容 JSON 数组和 JSONL 两种格式
        if content.startswith("["):
            # JSON 数组格式
            try:
                items = json.loads(content)
                all_data.extend(items)
            except json.JSONDecodeError as e:
                print(f"  解析 JSON 数组失败: {e}")
        else:
            # JSONL 格式（每行一个 JSON 对象）
            for line in content.split("\n"):
                line = line.strip()
                if line:
                    try:
                        item = json.loads(line)
                        all_data.append(item)
                    except json.JSONDecodeError:
                        pass

print(f"共读取 {len(all_data)} 条数据")

if all_data:
    print(f"数据字段: {list(all_data[0].keys())}")
    print(f"\n前3条数据示例:")
    for i in range(min(3, len(all_data))):
        print(f"\n--- 示例 {i+1} ---")
        print(json.dumps(all_data[i], ensure_ascii=False, indent=2)[:500])

    sample_size = min(10000, len(all_data))
    random.seed(42)
    random.shuffle(all_data)
    samples = all_data[:sample_size]

    train_size = int(sample_size * 0.9)
    valid_size = sample_size - train_size

    train_data = samples[:train_size]
    valid_data = samples[train_size:]

    os.makedirs("data/medical", exist_ok=True)

    train_file = "data/medical/train.jsonl"
    with open(train_file, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    valid_file = "data/medical/valid.jsonl"
    with open(valid_file, "w", encoding="utf-8") as f:
        for item in valid_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n已保存 {train_size} 条训练数据到 {train_file}")
    print(f"已保存 {valid_size} 条验证数据到 {valid_file}")
    print("\n数据集下载和处理完成！")
else:
    print("未读取到数据。")
