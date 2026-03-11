#!/bin/bash
# Run LoRA SFT with LLaMA-Factory for Qwen3.5-0.8B on train_law / dev_law.
# Execute from project root: ./run_train.sh
# Output is written to ./output/sft_qwen35_<timestamp>.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
# 使用相对路径：输出在项目根 output/ 下
OUTPUT_BASE="output"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${OUTPUT_BASE}/sft_qwen35_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

LLAMA_FACTORY_DIR="${SCRIPT_DIR}/LLaMA-Factory"
if [ ! -d "$LLAMA_FACTORY_DIR" ]; then
  echo "LLaMA-Factory not found at $LLAMA_FACTORY_DIR. Run: git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git"
  exit 1
fi

cd "$LLAMA_FACTORY_DIR"
export PYTHONPATH="${LLAMA_FACTORY_DIR}:${PYTHONPATH}"

# 相对路径：从 LLaMA-Factory 目录到项目根 output/（../output/...）
OUTPUT_DIR_REL="../${OUTPUT_DIR}"
# 基座模型：优先使用已下载的 models/Qwen3.5-0.8B（相对项目根）
QWEN_MODEL="${SCRIPT_DIR}/models/Qwen3.5-0.8B"
if [ ! -d "$QWEN_MODEL" ]; then
  QWEN_MODEL="Qwen/Qwen3.5-0.8B"
fi

llamafactory-cli train \
  --model_name_or_path "$QWEN_MODEL" \
  --trust_remote_code true \
  --stage sft \
  --do_train true \
  --finetuning_type lora \
  --lora_rank 8 \
  --lora_target all \
  --dataset train_law \
  --eval_dataset dev_law \
  --template qwen3_5_nothink \
  --cutoff_len 2048 \
  --preprocessing_num_workers 4 \
  --dataloader_num_workers 2 \
  --output_dir "$OUTPUT_DIR_REL" \
  --logging_steps 10 \
  --save_steps 500 \
  --eval_strategy steps \
  --eval_steps 200 \
  --plot_loss true \
  --overwrite_output_dir true \
  --save_only_model false \
  --report_to none \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-4 \
  --num_train_epochs 3.0 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.1 \
  --bf16 true \
  --ddp_timeout 180000000

echo "Training finished. Output: $SCRIPT_DIR/$OUTPUT_DIR (relative to project root)"
