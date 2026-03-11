# -*- coding: utf-8 -*-
"""
RAG service: load corpus, build BM25 and/or embedding index, search.
知识库：支持 BM25（默认）与可选的 embedding 检索；embedding 模型固定下载到 models/ 下。
"""

import json
import os
from typing import Any, Optional

import config

try:
    import jieba
except ImportError:
    jieba = None

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

# Embedding：优先使用项目内 models/bge-small-zh-v1.5
_embedding_model = None


def _get_embedding_model():
    """Lazy load embedding model from config.EMBEDDING_LOCAL_DIR or HF id."""
    global _embedding_model
    if _embedding_model is not None:
        return _embedding_model
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None
    path = config.EMBEDDING_LOCAL_DIR if os.path.isdir(config.EMBEDDING_LOCAL_DIR) else config.EMBEDDING_HF_ID
    _embedding_model = SentenceTransformer(path)
    return _embedding_model


DEFAULT_CORPUS_PATH = config.CORPUS_JSONL
DEFAULT_TOP_K = config.DEFAULT_TOP_K
MAX_CONTEXT_CHARS = config.MAX_CONTEXT_CHARS


def _tokenize(text: str) -> list[str]:
    if jieba:
        return list(jieba.cut_for_search(text))
    return list(text)  # fallback: char-level


class RAGService:
    """BM25 + 可选 embedding 检索。use_embedding=True 时需已执行 download_embedding.py。"""

    def __init__(self, corpus_path: Optional[str] = None, use_embedding: bool = False):
        self.corpus_path = corpus_path or DEFAULT_CORPUS_PATH
        self.use_embedding = use_embedding
        self.documents: list[dict[str, Any]] = []
        self._tokenized_corpus: list[list[str]] = []
        self._bm25: Optional[Any] = None
        self._embeddings: Optional[Any] = None  # numpy array [n_docs, dim]
        self._built = False

    def build_index(self, corpus_path: Optional[str] = None) -> None:
        if corpus_path:
            self.corpus_path = corpus_path
        self.documents = []
        self._tokenized_corpus = []
        self._bm25 = None
        self._embeddings = None
        self._built = False
        if not os.path.isfile(self.corpus_path):
            return
        with open(self.corpus_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    doc = json.loads(line)
                except json.JSONDecodeError:
                    continue
                content = doc.get("content", "")
                if isinstance(content, str):
                    content = content.replace("\\n", "\n")
                text = f"{doc.get('name', '')} {content}"
                self.documents.append(
                    {"id": doc.get("id"), "name": doc.get("name", ""), "content": content}
                )
                self._tokenized_corpus.append(_tokenize(text))
        if BM25Okapi is not None and self._tokenized_corpus:
            self._bm25 = BM25Okapi(self._tokenized_corpus)
        if self.use_embedding and self.documents:
            model = _get_embedding_model()
            if model is not None:
                texts = [f"{d.get('name', '')} {d.get('content', '')}" for d in self.documents]
                self._embeddings = model.encode(texts, show_progress_bar=False)
        self._built = True

    def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[dict[str, Any]]:
        if not self._built:
            self.build_index()
        if self.use_embedding and self._embeddings is not None:
            model = _get_embedding_model()
            if model is not None:
                q_emb = model.encode([query], show_progress_bar=False)
                import numpy as np
                scores = np.dot(self._embeddings, q_emb.T).ravel()
                indices = np.argsort(scores)[::-1][:top_k]
                return [self.documents[i] for i in indices]
        if not self._bm25 or not self.documents:
            return []
        tokenized_query = _tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)
        indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [self.documents[i] for i in indices]

    def get_context(self, query: str, top_k: int = DEFAULT_TOP_K) -> str:
        """Return concatenated top-k doc contents for LLM context, capped by MAX_CONTEXT_CHARS."""
        docs = self.search(query, top_k=top_k)
        parts = []
        total = 0
        for d in docs:
            name = d.get("name", "")
            content = (d.get("content") or "").replace("\\n", "\n")
            block = f"【{name}】\n{content}"
            if total + len(block) > MAX_CONTEXT_CHARS:
                break
            parts.append(block)
            total += len(block)
        return "\n\n".join(parts)


# Singleton or keyed by path for default corpus
_default_service: Optional[RAGService] = None
_default_use_embedding = False


def get_rag_service(corpus_path: Optional[str] = None, use_embedding: bool = False) -> RAGService:
    global _default_service, _default_use_embedding
    path = corpus_path or DEFAULT_CORPUS_PATH
    if (
        _default_service is None
        or _default_service.corpus_path != path
        or _default_use_embedding != use_embedding
    ):
        _default_use_embedding = use_embedding
        _default_service = RAGService(corpus_path=path, use_embedding=use_embedding)
        _default_service.build_index()
    return _default_service
