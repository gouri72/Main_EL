"""Helpers for reading data/tasks.jsonl. Import from rl/ or sim/ as: from data.loader import load_tasks

Backwards compatible: load_tasks, session_pool and transfer_of keep their old behaviour.
New: abstract_key and draw_session (leakage-safe, difficulty-balanced session sampler).
"""
import json
import re
from pathlib import Path

TASKS_PATH = Path(__file__).resolve().parent / "tasks.jsonl"
_ABSTRACT_RE = re.compile(r"Abstract \((.*?)\):\n", re.S)


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


def abstract_key(task):
    """Group id for tasks that share evidence. Science: the abstract title. Coding: the task id."""
    if task["domain"] == "science":
        m = _ABSTRACT_RE.search(task["prompt"])
        if m:
            return "sci:" + m.group(1)
    return task["task_id"]


def draw_session(pool, rng, n=8, domain=None):
    """Draw n session tasks without replacement from pool using a numpy Generator `rng`.

    Guarantees: no two tasks share an abstract (so one task cannot leak evidence for another),
    and difficulty is balanced as evenly as possible (round-robin over difficulty 1,2,3 with a
    random starting level). Order within the session is shuffled. Deterministic given rng state.
    """
    cand = [t for t in pool if domain is None or t["domain"] == domain]
    if len(cand) < n:
        raise ValueError("pool has %d tasks, need %d" % (len(cand), n))
    order = rng.permutation(len(cand))
    by_diff = {1: [], 2: [], 3: []}
    for i in order:
        by_diff[cand[int(i)]["difficulty"]].append(cand[int(i)])
    levels = [1, 2, 3]
    start = int(rng.integers(0, 3))
    levels = levels[start:] + levels[:start]
    chosen, used = [], set()
    guard = 0
    while len(chosen) < n and guard < 10 * n + 30:
        progressed = False
        for d in levels:
            while by_diff[d]:
                t = by_diff[d].pop()
                k = abstract_key(t)
                if k in used:
                    continue
                used.add(k)
                chosen.append(t)
                progressed = True
                break
            if len(chosen) >= n:
                break
        guard += 1
        if not progressed:
            break
    if len(chosen) < n:
        raise ValueError("could not draw %d non-overlapping tasks" % n)
    perm = rng.permutation(len(chosen))
    return [chosen[int(i)] for i in perm]