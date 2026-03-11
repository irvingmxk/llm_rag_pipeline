# -*- coding: utf-8 -*-
"""
Load training results from output_dir for display: list runs, get logs and metrics.
Expects LLaMA-Factory output: trainer_log.jsonl and/or trainer_state.json.
"""

import json
import os
from typing import Any

import config

OUTPUT_DIR = config.OUTPUT_DIR
TRAINER_LOG = "trainer_log.jsonl"
TRAINER_STATE = "trainer_state.json"


def list_runs() -> list[dict[str, Any]]:
    """List all training runs under output_dir. Each run is a subdirectory."""
    if not os.path.isdir(OUTPUT_DIR):
        return []
    runs = []
    for name in sorted(os.listdir(OUTPUT_DIR), reverse=True):
        path = os.path.join(OUTPUT_DIR, name)
        if not os.path.isdir(path):
            continue
        if name.startswith("."):
            continue
        has_log = os.path.isfile(os.path.join(path, TRAINER_LOG))
        has_state = os.path.isfile(os.path.join(path, TRAINER_STATE))
        if has_log or has_state:
            runs.append({"run_id": name, "path": path})
    return runs


def _load_trainer_log(run_path: str) -> tuple[str, list[dict[str, Any]]]:
    """Load trainer_log.jsonl. Return (log_text, list of log entries for charts)."""
    log_path = os.path.join(run_path, TRAINER_LOG)
    if not os.path.isfile(log_path):
        return "", []
    lines_text = []
    entries = []
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            lines_text.append(line)
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    log_text = "\n".join(lines_text[-500:])  # last 500 lines for display
    return log_text, entries


def _load_trainer_state(run_path: str) -> tuple[str, list[dict[str, Any]]]:
    """Load trainer_state.json log_history. Return (state_summary_text, metrics list)."""
    state_path = os.path.join(run_path, TRAINER_STATE)
    if not os.path.isfile(state_path):
        return "", []
    with open(state_path, encoding="utf-8") as f:
        data = json.load(f)
    log_history = data.get("log_history", [])
    entries = []
    for h in log_history:
        if "loss" in h:
            entries.append({"current_steps": h.get("step", 0), "loss": h["loss"]})
    summary_lines = [f"log_history entries: {len(log_history)}"]
    if log_history:
        last = log_history[-1]
        summary_lines.append(json.dumps(last, ensure_ascii=False, indent=2))
    return "\n".join(summary_lines), entries


def get_run_logs(run_id: str) -> str:
    """Return raw log text for the run (trainer_log.jsonl content or state summary)."""
    run_path = os.path.join(OUTPUT_DIR, run_id)
    if not os.path.isdir(run_path):
        return ""
    log_text, _ = _load_trainer_log(run_path)
    if log_text:
        return log_text
    state_text, _ = _load_trainer_state(run_path)
    return state_text


def get_run_metrics(run_id: str) -> list[dict[str, Any]]:
    """
    Return metrics for chart: list of {step, loss}.
    Prefer trainer_log.jsonl (has current_steps, loss), else trainer_state.json log_history.
    """
    run_path = os.path.join(OUTPUT_DIR, run_id)
    if not os.path.isdir(run_path):
        return []
    _, entries = _load_trainer_log(run_path)
    if entries:
        return [
            {"step": e.get("current_steps", 0), "loss": e.get("loss")}
            for e in entries
            if e.get("loss") is not None
        ]
    _, entries = _load_trainer_state(run_path)
    return [
        {"step": e.get("current_steps", 0), "loss": e.get("loss")}
        for e in entries
        if e.get("loss") is not None
    ]
