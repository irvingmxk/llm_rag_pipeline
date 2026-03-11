# -*- coding: utf-8 -*-
"""Project configuration: paths (relative to project root where possible), model names, and defaults."""

import os

# Project root (directory containing this file)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def _rel(*parts: str) -> str:
    """Path relative to project root (joined)."""
    return os.path.join(PROJECT_ROOT, *parts)


# ----- 相对路径常量（便于迁移与阅读） -----
REL_DATA_DIR = "data"
REL_OUTPUT_DIR = "output"
REL_MODELS_DIR = "models"
REL_LLAMA_FACTORY = "LLaMA-Factory"
REL_SESSIONS_DIR = "sessions"
REL_TEST_RESULTS_DIR = "test_results"

# 数据路径（项目内相对：data/...）
DATA_DIR = _rel(REL_DATA_DIR)
TRAIN_JSON = _rel(REL_DATA_DIR, "train.json")
DEV_JSON = _rel(REL_DATA_DIR, "dev.json")
CORPUS_JSONL = _rel(REL_DATA_DIR, "corpus.jsonl")
QUERIES_JSON = _rel(REL_DATA_DIR, "queries.json")

# 训练输出目录（训练结果展示从这里读）
OUTPUT_DIR = _rel(REL_OUTPUT_DIR)

# 模型统一下载到固定目录（非软链接）
MODELS_DIR = _rel(REL_MODELS_DIR)
QWEN_08B_HF_ID = "Qwen/Qwen3.5-0.8B"
QWEN_08B_DIRNAME = "Qwen3.5-0.8B"
# 本地基座路径：优先使用已下载的 models/Qwen3.5-0.8B
QWEN_08B_LOCAL_DIR = _rel(REL_MODELS_DIR, QWEN_08B_DIRNAME)


def get_qwen_base_model_path() -> str:
    """训练/推理使用的基座路径：本地目录存在则用本地，否则用 HF id（会走缓存）。"""
    if os.path.isdir(QWEN_08B_LOCAL_DIR):
        return QWEN_08B_LOCAL_DIR
    return QWEN_08B_HF_ID


# 默认基座显示名（实际路径由 get_qwen_base_model_path() 决定）
DEFAULT_BASE_MODEL = QWEN_08B_HF_ID

# Embedding 模型：固定下载到 models/ 下
EMBEDDING_HF_ID = "BAAI/bge-small-zh-v1.5"
EMBEDDING_DIRNAME = "bge-small-zh-v1.5"
EMBEDDING_LOCAL_DIR = _rel(REL_MODELS_DIR, EMBEDDING_DIRNAME)

# LLaMA-Factory 位置（项目内相对）
LLAMA_FACTORY_DIR = _rel(REL_LLAMA_FACTORY)

# RAG
DEFAULT_TOP_K = 5
MAX_CONTEXT_CHARS = 8000

# 会话与测试结果目录
SESSIONS_DIR = _rel(REL_SESSIONS_DIR)
TEST_RESULTS_DIR = _rel(REL_TEST_RESULTS_DIR)
