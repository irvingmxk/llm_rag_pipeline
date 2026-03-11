#!/bin/bash
# 使用 Hugging Face 最新的 hf CLI 将 Qwen3.5-0.8B 下载到固定目录 models/Qwen3.5-0.8B
# 先安装: curl -LsSf https://hf.co/cli/install.sh | bash
# 再登录: hf login
# 然后执行: bash download_qwen_hf.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
TARGET_DIR="${SCRIPT_DIR}/models/Qwen3.5-0.8B"

if ! command -v hf &>/dev/null; then
  echo "未检测到 hf 命令。请先安装 Hugging Face 最新 CLI："
  echo "  curl -LsSf https://hf.co/cli/install.sh | bash"
  echo "安装后重新打开终端，或执行: source ~/.bashrc 或 source ~/.zshrc"
  exit 1
fi

mkdir -p "$(dirname "$TARGET_DIR")"
echo "正在用 hf 下载 Qwen/Qwen3.5-0.8B 到 ${TARGET_DIR} ..."
hf download Qwen/Qwen3.5-0.8B --local-dir "$TARGET_DIR"
echo "已保存到: ${TARGET_DIR}"
