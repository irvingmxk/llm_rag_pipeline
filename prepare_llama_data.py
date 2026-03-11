# -*- coding: utf-8 -*-
"""
将 data/train.json、data/dev.json 拷贝到 LLaMA-Factory/data/ 下为 train_law.json、dev_law.json（实文件，非软链接）。
执行：python prepare_llama_data.py
"""

import os
import shutil

import config

SRC_TRAIN = config.TRAIN_JSON
SRC_DEV = config.DEV_JSON
LF_DATA = os.path.join(config.LLAMA_FACTORY_DIR, "data")
DST_TRAIN = os.path.join(LF_DATA, "train_law.json")
DST_DEV = os.path.join(LF_DATA, "dev_law.json")


def main():
    if not os.path.isdir(config.LLAMA_FACTORY_DIR):
        print(f"未找到 LLaMA-Factory 目录: {config.LLAMA_FACTORY_DIR}")
        return
    os.makedirs(LF_DATA, exist_ok=True)
    if not os.path.isfile(SRC_TRAIN):
        print(f"未找到: {SRC_TRAIN}")
        return
    if not os.path.isfile(SRC_DEV):
        print(f"未找到: {SRC_DEV}")
        return
    # 若目标是软链接则先删除，再拷贝为实文件
    for dst in (DST_TRAIN, DST_DEV):
        if os.path.lexists(dst):
            os.remove(dst)
    shutil.copy2(SRC_TRAIN, DST_TRAIN)
    shutil.copy2(SRC_DEV, DST_DEV)
    print(f"已拷贝: {SRC_TRAIN} -> {DST_TRAIN}")
    print(f"已拷贝: {SRC_DEV} -> {DST_DEV}")


if __name__ == "__main__":
    main()
