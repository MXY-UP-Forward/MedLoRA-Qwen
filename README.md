# MiniLoRA: Qwen2.5 Medical LoRA Fine-Tuning

[中文 README](README_CN.md)

This project fine-tunes **Qwen2.5-0.5B-Instruct** on a Chinese medical question-answering task with **LoRA SFT**. It covers the full workflow from data preparation to training, inference comparison, batch evaluation, and ablation analysis.

The repository is structured as a hands-on learning project. The `scripts/my_*.py` files contain the completed student implementations, while the non-`my_` scripts are reference implementations.

## Project Overview

The goal is to adapt a general instruction model to a medical health Q&A style:

```text
Raw medical data
  -> JSONL cleaning
  -> SFT message formatting
  -> assistant-only loss masking
  -> LoRA fine-tuning
  -> base vs LoRA inference comparison
  -> batch evaluation
  -> ablation experiments
```

The main model and data setup:

| Item | Value |
|---|---|
| Base model | Qwen2.5-0.5B-Instruct |
| Fine-tuning method | LoRA SFT |
| Task | Chinese medical health Q&A |
| Training samples | 640 for the full run |
| Validation samples | 160 |
| Test samples | 200 |
| Batch evaluation prompts | 10 |
| Main output adapter | `outputs/qwen-medical-lora` |
| Experiment records | `res.csv` |
| Experiment report | `EXPERIMENT_REPORT.md` |

## Repository Structure

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

## Main Scripts

| Module | Script | Purpose |
|---|---|---|
| Data preparation | `scripts/my_prepare_data.py` | Clean raw medical data and write JSONL files |
| SFT preprocessing | `scripts/my_train_lora.py` | Convert data to chat messages and construct labels |
| LoRA training | `scripts/my_train_lora.py` | Load Qwen2.5, attach LoRA modules, train with `Trainer` |
| Inference comparison | `scripts/my_infer_compare.py` | Compare one base response with one LoRA response |
| Batch evaluation | `scripts/my_eval_lora.py` | Generate base and LoRA answers for all eval prompts |
| Reference code | `scripts/*.py` without `my_` | Complete reference implementations |

## Environment Setup

Recommended environment:

- Python 3.10+
- PyTorch 2.1+
- `transformers >= 4.45.0`
- `peft >= 0.12.0`
- NVIDIA GPU with 6GB+ VRAM is recommended

Install dependencies:

```powershell
pip install -r requirements.txt
```

Check whether PyTorch can use CUDA:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

If the output is `True`, GPU training is available.

## Data and Model Preparation

The project expects the Qwen2.5 model under:

```text
models/Qwen2.5-0.5B-Instruct
```

Prepare the dataset:

```powershell
python download_dataset.py
python scripts\my_prepare_data.py
```

After preprocessing, the following files are used for training and evaluation:

```text
data/medical/train.jsonl
data/medical/valid.jsonl
data/medical/test.jsonl
data/medical/eval_prompts.jsonl
```

## Training

### Quick Sanity Check

Use a small subset first to verify the environment and code:

```powershell
python scripts\my_train_lora.py --max-train-samples 50 --max-valid-samples 20 --epochs 1 --grad-accum 4 --max-length 256
```

### Full Training

The main experiment uses 640 training samples:

```powershell
python scripts\my_train_lora.py --r 8 --output-dir outputs/qwen-medical-lora --epochs 1 --batch-size 1 --grad-accum 8 --max-length 512
```

The trained LoRA adapter is saved to:

```text
outputs/qwen-medical-lora
```

## Inference and Evaluation

Run a single-question comparison:

```powershell
python scripts\my_infer_compare.py --question "高血压患者日常生活中应该注意什么？"
```

Run batch evaluation:

```powershell
python scripts\my_eval_lora.py
```

The batch result is saved as JSONL:

```text
results/base_vs_lora.jsonl
```

Each line contains:

```json
{
  "question": "...",
  "base_answer": "...",
  "lora_answer": "..."
}
```

## Ablation Experiments

The project records several ablation runs in `res.csv`, including:

- LoRA rank: `r=4`, `r=8`, `r=16`
- LoRA alpha: `8`, `16`, `32`
- Target modules: `all`, `q_proj`, `q_proj+k_proj+v_proj`
- Training data size: `200` vs `640`

Example commands:

```powershell
python scripts\my_train_lora.py --r 4 --output-dir outputs/lora-r4 --max-train-samples 200 --epochs 1 --max-length 512
python scripts\my_train_lora.py --r 8 --output-dir outputs/lora-r8 --max-train-samples 200 --epochs 1 --max-length 512
python scripts\my_train_lora.py --r 16 --output-dir outputs/lora-r16 --max-train-samples 200 --epochs 1 --max-length 512
```

## Experiment Results

Results from `res.csv`:

| Experiment | r | alpha | target modules | samples | train loss | eval loss |
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

Main observations:

1. The full-data run performs best, with `eval_loss = 2.789`.
2. LoRA rank has little effect under the 200-sample setting. The gap between r=4, r=8, and r=16 is only 0.008 in eval loss.
3. `alpha=8` performs best among the 200-sample alpha experiments, suggesting a milder LoRA scaling can generalize better on small data.
4. Target modules matter a lot. Training only `q_proj` or `q_proj+k_proj+v_proj` is clearly worse than using `all`.
5. The most practical default is `r=8`, `lora_alpha=16`, and `target_modules=all`.

For a fuller discussion, see:

```text
EXPERIMENT_REPORT.md
```

## Key Implementation Details

### Assistant-Only Loss Masking

The training script builds labels so that only the assistant answer contributes to loss:

```text
input_ids: prompt tokens + answer tokens
labels:    -100 tokens    + answer token ids
```

This prevents the model from learning to copy the prompt and focuses training on answer generation.

### LoRA Configuration

The main configuration uses:

```text
r = 8
lora_alpha = 16
target_modules = q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
```

Only LoRA parameters are trainable; the base Qwen2.5 weights remain frozen.

### Memory Management

The inference scripts load the base model first, generate answers, release it, and then load the LoRA model. This is important on 6GB GPUs because both models may not fit in memory at the same time.

## Recommended Workflow

```powershell
pip install -r requirements.txt
python scripts\my_prepare_data.py
python scripts\my_train_lora.py --r 8 --output-dir outputs/qwen-medical-lora --epochs 1 --batch-size 1 --grad-accum 8 --max-length 512
python scripts\my_infer_compare.py --question "高血压患者日常生活中应该注意什么？"
python scripts\my_eval_lora.py
```

## Future Work

- Increase the amount of medical SFT data.
- Train for more epochs and monitor validation loss.
- Add automatic generation metrics such as ROUGE or BERTScore.
- Add human evaluation for medical helpfulness, clarity, and safety.
- Try larger base models such as Qwen2.5-1.5B or Qwen2.5-7B.
- Build a retrieval-augmented generation pipeline for more factual medical answers.
