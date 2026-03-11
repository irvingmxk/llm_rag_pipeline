# -*- coding: utf-8 -*-
"""
Run LoRA SFT with LLaMA-Factory for Qwen3.5-0.8B on train_law / dev_law.
Execute from project root: python run_train.py
Output is written to ./output/sft_qwen35_<timestamp>.
"""

import os
import subprocess
import sys
from datetime import datetime

import config

PROJECT_ROOT = config.PROJECT_ROOT
LLAMA_FACTORY_DIR = config.LLAMA_FACTORY_DIR
OUTPUT_BASE = config.OUTPUT_DIR
QWEN_MODEL = config.get_qwen_base_model_path()


def main():
    if not os.path.isdir(LLAMA_FACTORY_DIR):
        print(f"LLaMA-Factory not found at {LLAMA_FACTORY_DIR}")
        print("Run: git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git")
        sys.exit(1)

    os.makedirs(OUTPUT_BASE, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 使用相对路径：从 LLaMA-Factory 目录到项目 output/
    run_dir_rel = os.path.join("..", config.REL_OUTPUT_DIR, f"sft_qwen35_{timestamp}")
    run_dir_abs = os.path.join(OUTPUT_BASE, f"sft_qwen35_{timestamp}")
    os.makedirs(run_dir_abs, exist_ok=True)

    env = os.environ.copy()
    src_dir = os.path.join(LLAMA_FACTORY_DIR, "src")
    env["PYTHONPATH"] = f"{src_dir}{os.pathsep}{env.get('PYTHONPATH', '')}"

    cmd = [
        sys.executable,
        "-m",
        "llamafactory.cli",
        "train",
        "--model_name_or_path",
        QWEN_MODEL,
        "--trust_remote_code",
        "true",
        "--stage",
        "sft",
        "--do_train",
        "true",
        "--finetuning_type",
        "lora",
        "--lora_rank",
        "8",
        "--lora_target",
        "all",
        "--dataset",
        "train_law",
        "--eval_dataset",
        "dev_law",
        "--template",
        "qwen3_5_nothink",
        "--cutoff_len",
        "2048",
        "--preprocessing_num_workers",
        "4",
        "--dataloader_num_workers",
        "2",
        "--output_dir",
        run_dir_rel,
        "--logging_steps",
        "10",
        "--save_steps",
        "500",
        "--eval_strategy",
        "steps",
        "--eval_steps",
        "200",
        "--plot_loss",
        "true",
        "--overwrite_output_dir",
        "true",
        "--save_only_model",
        "false",
        "--report_to",
        "none",
        "--per_device_train_batch_size",
        "2",
        "--gradient_accumulation_steps",
        "4",
        "--learning_rate",
        "1e-4",
        "--num_train_epochs",
        "3.0",
        "--lr_scheduler_type",
        "cosine",
        "--warmup_ratio",
        "0.1",
        "--bf16",
        "true",
        "--ddp_timeout",
        "180000000",
    ]

    result = subprocess.run(cmd, cwd=LLAMA_FACTORY_DIR, env=env)
    if result.returncode != 0:
        sys.exit(result.returncode)
    print(f"Training finished. Output: {run_dir_abs}")


if __name__ == "__main__":
    main()
