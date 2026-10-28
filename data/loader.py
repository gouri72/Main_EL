"""Helpers for reading data/tasks.jsonl. Import from rl/ or sim/ as: from data.loader import load_tasks"""
import json
from pathlib import Path

TASKS_PATH = Path(__file__).resolve().parent / "tasks.jsonl"


def load_tasks(path=TASKS_PATH):
    """Return {task_id: task_dict} for every line, including the *_T transfer variants."""
    tasks = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                t = json.loads(line)
                tasks[t["task_id"]] = t
    return tasks


def session_pool(tasks, domain=None):
    """Tasks usable inside a training session (transfer variants excluded)."""
    return [t for tid, t in tasks.items()
            if not tid.endswith("_T") and (domain is None or t["domain"] == domain)]


def transfer_of(tasks, task):
    """The unaided transfer variant for a session task."""
    return tasks[task["transfer_task_id"]]