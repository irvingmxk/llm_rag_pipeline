# -*- coding: utf-8 -*-
"""
Model inference: load base model + optional LoRA adapter, generate with RAG context.
Batch test on dev/queries for model-test module.
"""

import json
import os
import uuid
from typing import Any, Optional

import config

DEV_JSON = config.DEV_JSON
QUERIES_JSON = config.QUERIES_JSON
TEST_RESULTS_DIR = config.TEST_RESULTS_DIR

# Lazy-loaded model and tokenizer
_model = None
_tokenizer = None
_adapter_path = None
_base_model_path = None


def _get_model_and_tokenizer(
    base_model_path: Optional[str] = None,
    adapter_path: Optional[str] = None,
):
    """Load transformers model and tokenizer; optional PEFT adapter."""
    global _model, _tokenizer, _adapter_path, _base_model_path
    base = base_model_path or config.get_qwen_base_model_path()
    if _model is not None and _base_model_path == base and _adapter_path == adapter_path:
        return _model, _tokenizer
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        raise RuntimeError("transformers is required for inference. pip install transformers")
    tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    mod = AutoModelForCausalLM.from_pretrained(
        base,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype="auto",
    )
    _base_model_path = base
    _adapter_path = adapter_path
    if adapter_path and os.path.isdir(adapter_path):
        try:
            from peft import PeftModel
            mod = PeftModel.from_pretrained(mod, adapter_path)
        except Exception:
            pass
    mod.eval()
    _model = mod
    _tokenizer = tok
    return _model, _tokenizer


def build_prompt_with_context(question: str, context: str) -> str:
    """Build Qwen-style chat prompt with RAG context."""
    if context.strip():
        user_content = f"""以下为参考知识库内容，请基于此回答用户问题。

【参考内容】
{context}

【用户问题】
{question}"""
    else:
        user_content = question
    return f"<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n"


def generate(
    question: str,
    context: str = "",
    base_model_path: Optional[str] = None,
    adapter_path: Optional[str] = None,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
) -> str:
    """Generate one reply given question and optional RAG context."""
    import torch
    model, tokenizer = _get_model_and_tokenizer(base_model_path, adapter_path)
    prompt = build_prompt_with_context(question, context)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            pad_token_id=tokenizer.eos_token_id,
        )
    reply = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return reply.strip()


def run_batch_test(
    model_path: Optional[str] = None,
    adapter_path: Optional[str] = None,
    corpus_path: Optional[str] = None,
    eval_source: str = "dev",
    top_k: int = 5,
    max_samples: Optional[int] = 100,
) -> dict[str, Any]:
    """
    Run RAG + LLM on dev or queries. Returns task_id and writes result to test_results_dir.
    eval_source: "dev" -> dev.json conversations, "queries" -> queries.json 问题
    """
    from rag_service import get_rag_service

    os.makedirs(TEST_RESULTS_DIR, exist_ok=True)
    task_id = str(uuid.uuid4())[:8]
    result_path = os.path.join(TEST_RESULTS_DIR, f"test_{task_id}.jsonl")
    rag = get_rag_service(corpus_path)
    rag.build_index(corpus_path)

    questions_and_refs = []
    if eval_source == "queries" and os.path.isfile(QUERIES_JSON):
        with open(QUERIES_JSON, encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            q = item.get("问题") or item.get("query", "")
            if q:
                questions_and_refs.append((q, item.get("相关法规") or item.get("match_name")))
    else:
        with open(DEV_JSON, encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            conv = item.get("conversations", [])
            user_msg = next((c.get("content", "") for c in conv if c.get("role") == "user"), "")
            ref = next((c.get("content", "") for c in conv if c.get("role") == "assistant"), "")
            if user_msg:
                questions_and_refs.append((user_msg, ref))

    if max_samples:
        questions_and_refs = questions_and_refs[:max_samples]

    log_entries = []
    for i, (q, ref) in enumerate(questions_and_refs):
        context = rag.get_context(q, top_k=top_k)
        try:
            answer = generate(q, context, base_model_path=model_path, adapter_path=adapter_path)
        except Exception as e:
            answer = f"[Error: {e}]"
        log_entries.append({
            "idx": i,
            "question": q,
            "reference": ref if isinstance(ref, str) else (ref or ""),
            "answer": answer,
            "context_len": len(context),
        })
        with open(result_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entries[-1], ensure_ascii=False) + "\n")

    summary_path = os.path.join(TEST_RESULTS_DIR, f"test_{task_id}_summary.json")
    summary = {
        "task_id": task_id,
        "eval_source": eval_source,
        "num_samples": len(log_entries),
        "result_path": result_path,
        "summary_path": summary_path,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return {"task_id": task_id, "result_path": result_path, "summary": summary, "log_entries": log_entries}


def get_test_result(task_id: str) -> dict[str, Any]:
    """Load test result by task_id."""
    summary_path = os.path.join(TEST_RESULTS_DIR, f"test_{task_id}_summary.json")
    result_path = os.path.join(TEST_RESULTS_DIR, f"test_{task_id}.jsonl")
    out = {"task_id": task_id, "found": False, "summary": None, "log_entries": []}
    if os.path.isfile(summary_path):
        with open(summary_path, encoding="utf-8") as f:
            out["summary"] = json.load(f)
        out["found"] = True
    if os.path.isfile(result_path):
        with open(result_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out["log_entries"].append(json.loads(line))
    return out
