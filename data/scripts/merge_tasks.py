#!/usr/bin/env python3
"""Merge tasks_coding.jsonl + tasks_science.jsonl -> data/tasks.jsonl, then validate."""
import subprocess, sys
from pathlib import Path

D = Path(__file__).resolve().parents[1]
parts = [D / "tasks_coding.jsonl", D / "tasks_science.jsonl"]
lines = []
for p in parts:
    if p.exists():
        lines += [l for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    else:
        print(f"(skipping missing {p.name})")
(D / "tasks.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
print(f"merged {len(lines)} lines -> data/tasks.jsonl")
sys.exit(subprocess.call([sys.executable, str(D / "validate_tasks.py"), "--run-tests"]))
