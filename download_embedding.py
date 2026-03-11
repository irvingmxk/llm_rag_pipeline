# -*- coding: utf-8 -*-
"""
将 RAG 用 embedding 模型（BAAI/bge-small-zh-v1.5）下载到项目固定目录 models/bge-small-zh-v1.5（非软链接）。
执行：python download_embedding.py
"""

import os
import sys

import config

TARGET_DIR = config.EMBEDDING_LOCAL_DIR
HF_ID = config.EMBEDDING_HF_ID


def main():
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    if os.path.isdir(TARGET_DIR) and any(
        f for f in os.listdir(TARGET_DIR)
        if not f.startswith(".")
    ):
        print(f"Embedding 模型已存在: {TARGET_DIR}")
        return
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("请安装: pip install huggingface_hub")
        sys.exit(1)
    print(f"正在下载 {HF_ID} 到 {TARGET_DIR} ...")
    snapshot_download(
        repo_id=HF_ID,
        local_dir=TARGET_DIR,
        local_dir_use_symlinks=False,
    )
    print(f"已保存到: {TARGET_DIR}")


if __name__ == "__main__":
    main()
