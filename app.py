# -*- coding: utf-8 -*-
"""
LLM + RAG + 训练展示系统：FastAPI 后端 + Gradio 三模块前端。
"""

import json
import os
import uuid
from typing import Any, Optional

import config

# In-memory chat sessions: session_id -> list of {role, content}
_sessions: dict[str, list[dict[str, str]]] = {}


def _get_fastapi_app():
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware

    app = FastAPI(title="LLM RAG Pipeline", version="1.0")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    # ----- 训练展示 -----
    @app.get("/runs")
    def list_runs():
        import training_loader
        return training_loader.list_runs()

    @app.get("/runs/{run_id}/logs")
    def get_run_logs(run_id: str):
        import training_loader
        logs = training_loader.get_run_logs(run_id)
        return {"run_id": run_id, "logs": logs}

    @app.get("/runs/{run_id}/metrics")
    def get_run_metrics(run_id: str):
        import training_loader
        metrics = training_loader.get_run_metrics(run_id)
        return {"run_id": run_id, "metrics": metrics}

    # ----- 模型测试 -----
    @app.post("/test/run")
    def run_test(
        model_path: Optional[str] = None,
        adapter_path: Optional[str] = None,
        corpus_path: Optional[str] = None,
        eval_source: str = "dev",
        max_samples: Optional[int] = 50,
    ):
        import model_inference
        try:
            result = model_inference.run_batch_test(
                model_path=model_path,
                adapter_path=adapter_path,
                corpus_path=corpus_path or config.CORPUS_JSONL,
                eval_source=eval_source,
                max_samples=max_samples,
            )
            return {"task_id": result["task_id"], "summary": result["summary"], "log_entries": result["log_entries"]}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/test/result/{task_id}")
    def get_test_result(task_id: str):
        import model_inference
        return model_inference.get_test_result(task_id)

    # ----- 智能问答 -----
    @app.get("/sessions")
    def list_sessions():
        return {"session_ids": list(_sessions.keys())}

    @app.get("/sessions/{session_id}/messages")
    def get_session_messages(session_id: str):
        if session_id not in _sessions:
            return {"session_id": session_id, "messages": []}
        return {"session_id": session_id, "messages": _sessions[session_id]}

    @app.post("/chat")
    def chat(
        session_id: Optional[str] = None,
        message: str = "",
        model_path: Optional[str] = None,
        adapter_path: Optional[str] = None,
        corpus_path: Optional[str] = None,
        top_k: int = 5,
    ):
        import model_inference
        from rag_service import get_rag_service

        if not message.strip():
            raise HTTPException(status_code=400, detail="message is empty")
        sid = session_id or str(uuid.uuid4())
        if sid not in _sessions:
            _sessions[sid] = []
        rag = get_rag_service(corpus_path)
        context = rag.get_context(message, top_k=top_k)
        try:
            reply = model_inference.generate(
                message,
                context=context,
                base_model_path=model_path,
                adapter_path=adapter_path,
            )
        except Exception as e:
            reply = f"[生成错误: {e}]"
        _sessions[sid].append({"role": "user", "content": message})
        _sessions[sid].append({"role": "assistant", "content": reply})
        return {"session_id": sid, "reply": reply, "messages": _sessions[sid]}

    return app


def _build_gradio_ui():
    import gradio as gr

    runs = []
    try:
        import training_loader
        runs = training_loader.list_runs()
    except Exception:
        pass
    run_choices = [r["run_id"] for r in runs] if runs else ["（暂无训练结果）"]
    output_dir = config.OUTPUT_DIR
    adapter_choices = ["（基座模型）"]
    if os.path.isdir(output_dir):
        for name in sorted(os.listdir(output_dir), reverse=True):
            path = os.path.join(output_dir, name)
            if os.path.isdir(path):
                adapter_choices.append(path)

    def on_select_run(run_id):
        if not run_id or run_id == "（暂无训练结果）":
            return "（选择一次训练结果以加载日志与图表）", None
        import training_loader
        logs = training_loader.get_run_logs(run_id)
        metrics = training_loader.get_run_metrics(run_id)
        fig = None
        if metrics and len(metrics) > 0:
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                steps = [m["step"] for m in metrics]
                losses = [m["loss"] for m in metrics]
                plt.figure(figsize=(8, 4))
                plt.plot(steps, losses, alpha=0.6, label="loss")
                plt.xlabel("step")
                plt.ylabel("loss")
                plt.legend()
                plt.tight_layout()
                fig = plt.gcf()
            except Exception:
                pass
        return (logs[:20000] if logs else "（无日志）"), fig

    def do_run_test(model_sel, kb_sel, src, n):
        adapter = None if model_sel == "（基座模型）" else model_sel
        corpus = None if kb_sel == "（默认知识库）" else kb_sel
        import model_inference
        result = model_inference.run_batch_test(
            adapter_path=adapter,
            corpus_path=corpus or config.CORPUS_JSONL,
            eval_source=src,
            max_samples=int(n),
        )
        entries = result.get("log_entries", [])
        lines = [f"任务ID: {result['task_id']}\n样本数: {len(entries)}\n"]
        for e in entries[:20]:
            q = (e.get("question") or "")[:80]
            a = (e.get("answer") or "")[:80]
            lines.append(f"Q: {q}...")
            lines.append(f"A: {a}...\n")
        if len(entries) > 20:
            lines.append(f"... 共 {len(entries)} 条")
        fig = None
        if entries:
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                lens = [e.get("context_len", 0) for e in entries]
                plt.figure(figsize=(6, 3))
                plt.hist(lens, bins=min(20, len(set(lens)) or 1), edgecolor="black", alpha=0.7)
                plt.xlabel("context length")
                plt.ylabel("count")
                plt.title("Context length distribution")
                fig = plt.gcf()
            except Exception:
                pass
        return "\n".join(lines), fig

    def send_msg(message, history, sid, model_sel, kb_sel):
        if not message.strip():
            return history, sid, ""
        adapter = None if model_sel == "（基座模型）" else model_sel
        corpus = None if kb_sel == "（默认知识库）" else kb_sel
        import model_inference
        from rag_service import get_rag_service
        rag = get_rag_service(corpus)
        context = rag.get_context(message, top_k=5)
        try:
            reply = model_inference.generate(
                message,
                context=context,
                adapter_path=adapter,
            )
        except Exception as e:
            reply = f"[错误: {e}]"
        new_history = (history or []) + [[message, reply]]
        new_sid = sid or str(uuid.uuid4())
        if new_sid not in _sessions:
            _sessions[new_sid] = []
        _sessions[new_sid].append({"role": "user", "content": message})
        _sessions[new_sid].append({"role": "assistant", "content": reply})
        return new_history, new_sid, ""

    with gr.Blocks(title="LLM + RAG + 训练展示") as demo:
        gr.Markdown("# LLM + RAG + 训练展示系统")
        with gr.Tabs():
            with gr.TabItem("模型训练"):
                run_drop = gr.Dropdown(
                    choices=run_choices,
                    value=run_choices[0] if run_choices else None,
                    label="选择训练结果",
                )
                log_text = gr.Textbox(
                    label="训练日志",
                    value="（选择一次训练结果以加载日志与图表）",
                    lines=20,
                    max_lines=30,
                )
                plot_placeholder = gr.Plot(label="Loss 曲线")
                run_drop.change(on_select_run, run_drop, [log_text, plot_placeholder])
            with gr.TabItem("模型测试"):
                model_drop = gr.Dropdown(
                    choices=adapter_choices,
                    value=adapter_choices[0],
                    label="选择微调后的模型权重（或基座）",
                )
                kb_drop = gr.Dropdown(
                    choices=[config.CORPUS_JSONL, "（默认知识库）"],
                    value=config.CORPUS_JSONL,
                    label="选择 RAG 知识库",
                )
                eval_src = gr.Radio(choices=["dev", "queries"], value="dev", label="评测数据")
                max_samples = gr.Slider(10, 200, value=30, step=10, label="最大样本数")
                run_btn = gr.Button("开始测试")
                test_log = gr.Textbox(label="测试日志", lines=15)
                test_plot = gr.Plot(label="结果概览")
                run_btn.click(
                    do_run_test,
                    [model_drop, kb_drop, eval_src, max_samples],
                    [test_log, test_plot],
                )
            with gr.TabItem("智能问答"):
                session_id = gr.State(None)
                model_drop_c = gr.Dropdown(
                    choices=adapter_choices,
                    value=adapter_choices[0],
                    label="选择模型",
                )
                kb_drop_c = gr.Dropdown(
                    choices=[config.CORPUS_JSONL, "（默认知识库）"],
                    value=config.CORPUS_JSONL,
                    label="选择知识库",
                )
                session_list = gr.Textbox(label="会话记录", value="（当前会话）", lines=2)
                chat_area = gr.Chatbot(label="问答内容", height=400)
                msg_in = gr.Textbox(label="提问框", placeholder="输入问题...")
                send_btn = gr.Button("发送")
                send_btn.click(
                    send_msg,
                    [msg_in, chat_area, session_id, model_drop_c, kb_drop_c],
                    [chat_area, session_id, msg_in],
                )
    return demo


def main():
    import uvicorn
    import gradio as gr

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.TEST_RESULTS_DIR, exist_ok=True)
    os.makedirs(config.SESSIONS_DIR, exist_ok=True)

    app = _get_fastapi_app()
    demo = _build_gradio_ui()
    app = gr.mount_gradio_app(app, demo, path="/")
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
