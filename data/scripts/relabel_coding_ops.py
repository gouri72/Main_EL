#!/usr/bin/env python3
"""Re-derive coding operation lists from reference-solution features (not from difficulty).

Problem this fixes: build_coding.operations_for(difficulty) made the op list a pure function of
difficulty (d1 = 2 ops, d2 = 3, d3 = 4), so the operation-aware vs operation-blind ablation was
confounded with difficulty for the whole coding domain.

Rules (heuristic, label_source stays "heuristic"; every task keeps understand_plan + execute):
  debug_verify : solution contains a loop, a conditional, or index/slice access (off-by-one risk)
  explain      : solution has nested control flow, >= 2 distinct control constructs, a helper def,
                 a comprehension, recursion, or more than 8 statements (logic worth explaining)

Transfer variants (<id>_T) get exactly the ops of their base task.

Usage (repo root, PowerShell):
  python data\\scripts\\relabel_coding_ops.py            # rewrite data\\tasks_coding.jsonl, then merge
  python data\\scripts\\relabel_coding_ops.py --dry-run  # only print the crosstab
Then: python data\\scripts\\merge_tasks.py
"""
import argparse, ast, json
from collections import Counter
from pathlib import Path

D = Path(__file__).resolve().parents[1]
TASKS = D / "tasks_coding.jsonl"
SOLS = D / "solutions.jsonl"


def features(src):
    tree = ast.parse(src)
    loops = conds = subs = comps = helpers = stmts = 0
    max_depth = 0
    fn_names = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    recursion = False

    def depth(node, d=0):
        nonlocal max_depth
        for ch in ast.iter_child_nodes(node):
            nd = d + 1 if isinstance(ch, (ast.For, ast.While, ast.If)) else d
            max_depth = max(max_depth, nd)
            depth(ch, nd)

    depth(tree)
    for n in ast.walk(tree):
        if isinstance(n, (ast.For, ast.While)):
            loops += 1
        elif isinstance(n, ast.If):
            conds += 1
        elif isinstance(n, (ast.Subscript, ast.Slice)):
            subs += 1
        elif isinstance(n, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            comps += 1
        elif isinstance(n, ast.stmt) and not isinstance(n, (ast.FunctionDef, ast.Import, ast.ImportFrom)):
            stmts += 1
    helpers = max(0, len(fn_names) - 1)
    for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
        for c in ast.walk(fn):
            if isinstance(c, ast.Call) and getattr(c.func, "id", None) == fn.name:
                recursion = True
    kinds = sum(1 for x in (loops, conds, comps) if x)
    return dict(loops=loops, conds=conds, subs=subs, comps=comps, helpers=helpers,
                stmts=stmts, depth=max_depth, recursion=recursion, kinds=kinds)


def ops_for(f):
    ops = ["understand_plan", "execute"]
    if f["loops"] or f["conds"] or f["subs"]:
        ops.append("debug_verify")
    if f["depth"] >= 2 or f["kinds"] >= 2 or f["helpers"] or f["comps"] or f["recursion"] or f["stmts"] > 8:
        ops.append("explain")
    return ops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    tasks = [json.loads(l) for l in TASKS.read_text(encoding="utf-8").splitlines() if l.strip()]
    sols = {}
    for l in SOLS.read_text(encoding="utf-8").splitlines():
        if l.strip():
            r = json.loads(l)
            sols[r["task_id"]] = r["solution"]
    base_ops = {}
    for t in tasks:
        if t["domain"] != "coding" or t["task_id"].endswith("_T"):
            continue
        base_ops[t["task_id"]] = ops_for(features(sols[t["task_id"]]))
    cross = Counter()
    for t in tasks:
        bid = t["task_id"][:-2] if t["task_id"].endswith("_T") else t["task_id"]
        ops = base_ops[bid]
        t["operations"] = [{"op": o, "label_source": "heuristic"} for o in ops]
        if not t["task_id"].endswith("_T"):
            cross[(t["difficulty"], tuple(ops))] += 1
    print("difficulty x op-list (session tasks):")
    for k in sorted(cross):
        print(" ", k[0], "+".join(o[:4] for o in k[1]), cross[k])
    ndistinct = {d: len({k[1] for k in cross if k[0] == d}) for d in (1, 2, 3)}
    print("distinct op-lists per difficulty:", ndistinct)
    if a.dry_run:
        return
    with open(TASKS, "w", encoding="utf-8", newline="\n") as f:
        for t in tasks:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    print("rewrote", TASKS.name)


if __name__ == "__main__":
    main()