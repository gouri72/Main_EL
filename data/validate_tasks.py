#!/usr/bin/env python3
"""Validator for data/tasks.jsonl (task schema v1, FROZEN).

Run from anywhere:
  python data/validate_tasks.py                # schema + cross-reference checks
  python data/validate_tasks.py --run-tests    # also run every coding solution against its tests

Conventions (agreed on top of the frozen schema, no schema change):
  * Transfer variants are normal lines in tasks.jsonl with id "<id>_T".
  * A normal task's transfer_task_id is "<id>_T"; a variant's transfer_task_id is its own id.
  * Reference solutions for coding tasks live in data/solutions.jsonl as {task_id, solution}.
"""
import argparse, json, os, subprocess, sys, tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OPS = {"understand_plan", "execute", "debug_verify", "explain"}
DOMAINS = {"coding", "science"}
LABEL_SRC = {"gold", "heuristic"}
SCI_LABELS = {"SUPPORT", "CONTRADICT"}
KEYS = {"task_id", "domain", "prompt", "difficulty", "operations",
        "gold", "transfer_task_id", "error_types"}


def check_task(t):
    errs = []
    missing = KEYS - t.keys()
    extra = t.keys() - KEYS
    if missing:
        errs.append(f"missing keys {sorted(missing)}")
    if extra:
        errs.append(f"unknown keys {sorted(extra)} (schema is frozen)")
    if missing:
        return errs

    if not (isinstance(t["task_id"], str) and t["task_id"]):
        errs.append("task_id must be non-empty str")
    if t["domain"] not in DOMAINS:
        errs.append(f"domain must be one of {sorted(DOMAINS)}")
    if not (isinstance(t["prompt"], str) and t["prompt"].strip()):
        errs.append("prompt must be non-empty str")
    if isinstance(t["difficulty"], bool) or t["difficulty"] not in (1, 2, 3):
        errs.append("difficulty must be 1, 2 or 3")

    ops = t["operations"]
    if not (isinstance(ops, list) and ops):
        errs.append("operations must be a non-empty list")
    else:
        seen = set()
        for o in ops:
            if not isinstance(o, dict) or set(o) != {"op", "label_source"}:
                errs.append("each operation needs exactly {op, label_source}")
                continue
            if o["op"] not in OPS:
                errs.append(f"bad op {o['op']!r}")
            if o["label_source"] not in LABEL_SRC:
                errs.append(f"bad label_source {o['label_source']!r}")
            if o["op"] in seen:
                errs.append(f"duplicate op {o['op']!r}")
            seen.add(o["op"])

    g = t["gold"]
    if not isinstance(g, dict):
        errs.append("gold must be an object")
    elif t["domain"] == "coding":
        if set(g) != {"tests"} or not (isinstance(g.get("tests"), str) and g["tests"].strip()):
            errs.append("coding gold must be {tests: non-empty str}")
    elif t["domain"] == "science":
        ev = g.get("evidence")
        if set(g) != {"label", "evidence"}:
            errs.append("science gold must be {label, evidence}")
        elif g["label"] not in SCI_LABELS:
            errs.append(f"science label must be one of {sorted(SCI_LABELS)}")
        elif not (isinstance(ev, list) and ev and all(isinstance(e, str) and e for e in ev)):
            errs.append("evidence must be a non-empty list of str")

    if not (isinstance(t["transfer_task_id"], str) and t["transfer_task_id"]):
        errs.append("transfer_task_id must be non-empty str")
    if not (isinstance(t["error_types"], list) and all(isinstance(e, str) for e in t["error_types"])):
        errs.append("error_types must be a list of str")
    return errs


def run_solution(solution, tests, timeout=10):
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(solution + "\n\n" + tests + "\n")
        path = f.name
    try:
        r = subprocess.run([sys.executable, path], capture_output=True,
                           encoding="utf-8", errors="replace", timeout=timeout)
        return r.returncode == 0, r.stderr.strip().splitlines()[-1:]
    except subprocess.TimeoutExpired:
        return False, ["timeout"]
    finally:
        os.unlink(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=str(ROOT / "data" / "tasks.jsonl"))
    ap.add_argument("--solutions", default=str(ROOT / "data" / "solutions.jsonl"))
    ap.add_argument("--run-tests", action="store_true")
    a = ap.parse_args()

    tasks, errors = {}, 0
    with open(a.tasks, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                t = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"line {i}: invalid JSON ({e})")
                errors += 1
                continue
            errs = check_task(t)
            tid = t.get("task_id", f"<line {i}>")
            if tid in tasks:
                errs.append("duplicate task_id")
            for e in errs:
                print(f"{tid}: {e}")
            errors += len(errs)
            tasks[tid] = t

    for tid, t in tasks.items():
        if "transfer_task_id" not in t:
            continue
        is_variant = tid.endswith("_T")
        want = tid if is_variant else tid + "_T"
        if t["transfer_task_id"] != want:
            print(f"{tid}: transfer_task_id should be {want!r}")
            errors += 1
        elif want not in tasks:
            print(f"{tid}: transfer task {want!r} not found")
            errors += 1
        elif not is_variant:
            v = tasks[want]
            if v.get("domain") != t["domain"] or v.get("difficulty") != t["difficulty"]:
                print(f"{tid}: variant must match domain and difficulty")
                errors += 1

    if a.run_tests:
        sols = {}
        if os.path.exists(a.solutions):
            with open(a.solutions, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        s = json.loads(line)
                        sols[s["task_id"]] = s["solution"]
        for tid, t in tasks.items():
            if t.get("domain") != "coding" or "gold" not in t:
                continue
            if tid not in sols:
                print(f"{tid}: no solution in {a.solutions}")
                errors += 1
                continue
            ok, msg = run_solution(sols[tid], t["gold"]["tests"])
            if not ok:
                print(f"{tid}: solution FAILS tests {msg}")
                errors += 1

    print(f"\n{len(tasks)} tasks | domain {dict(Counter(t.get('domain') for t in tasks.values()))}"
          f" | difficulty {dict(sorted(Counter(t.get('difficulty') for t in tasks.values()).items()))}")
    print("OK" if errors == 0 else f"{errors} problem(s)")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
