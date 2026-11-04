#!/usr/bin/env python3
"""Data-quality checks on data/tasks.jsonl (and data/hints.jsonl if present). Prints a report; exit 1 on hard failures.

  python data/scripts/data_checks.py
HARD = must be zero.  INFO/WARN = numbers for the Data limitations section.
"""
import json, re, sys
from collections import Counter
from pathlib import Path

D = Path(__file__).resolve().parents[1]
tasks = {}
with open(D / "tasks.jsonl", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            t = json.loads(line)
            tasks[t["task_id"]] = t
base = {k: t for k, t in tasks.items() if not k.endswith("_T")}
var = {k: t for k, t in tasks.items() if k.endswith("_T")}
hard = []


def title(t):
    m = re.search(r"\nAbstract \((.*?)\):\n\[1\]", t["prompt"], re.S)
    return m.group(1).strip() if m else None


def fname(t):
    m = re.search(r"assert (\w+)\(", t["gold"]["tests"])
    return m.group(1) if m else None


print(f"tasks: {len(base)} session, {len(var)} transfer variants")

# 1. prompts
dup = [p for p, c in Counter(t["prompt"] for t in base.values()).items() if c > 1]
if dup:
    hard.append(f"{len(dup)} duplicated session prompts")
same = [k for k, t in base.items() if tasks[t["transfer_task_id"]]["prompt"] == t["prompt"]]
if same:
    hard.append(f"{len(same)} variants with the same prompt as their base task")
long_ = [k for k, t in base.items() if len(t["prompt"]) > 6000]
print(f"WARN prompts > 6000 chars (Qwen context risk): {len(long_)} {long_}")

# 2. coding
cb = {k: t for k, t in base.items() if t["domain"] == "coding"}
same_fn = [k for k, t in cb.items() if fname(t) == fname(tasks[t["transfer_task_id"]])]
if same_fn:
    hard.append(f"{len(same_fn)} coding variants keep the base function name")
shared_tests = 0
for k, t in cb.items():
    a = set(t["gold"]["tests"].splitlines())
    b = set(tasks[t["transfer_task_id"]]["gold"]["tests"].splitlines())
    shared_tests += len(a & b)
if shared_tests:
    hard.append(f"{shared_tests} identical test lines shared between coding bases and variants")
fn_dups = [n for n, c in Counter(fname(t) for t in cb.values()).items() if c > 1]
print(f"INFO coding: base function names reused by more than one session task: {len(fn_dups)}")
rule_ok = sum(len(t["operations"]) == 2 + (t["difficulty"] >= 2) + (t["difficulty"] >= 3) for t in cb.values())
print(f"INFO coding: operation list fully determined by difficulty in {rule_ok}/{len(cb)} tasks")

# 3. science
sb = {k: t for k, t in base.items() if t["domain"] == "science"}
bad_ev = [k for k, t in tasks.items() if t["domain"] == "science" and not all(e in t["prompt"] for e in t["gold"]["evidence"])]
if bad_ev:
    hard.append(f"{len(bad_ev)} science tasks whose gold evidence is not verbatim in the prompt abstract")
badlab = [k for k, t in sb.items() if tasks[t["transfer_task_id"]]["gold"]["label"] != t["gold"]["label"]]
if badlab:
    hard.append(f"{len(badlab)} science variants with a different label than their base")
tb = Counter(title(t) for t in sb.values())
tv = Counter(title(tasks[t["transfer_task_id"]]) for t in sb.values())
ss = sum(c - 1 for c in tb.values() if c > 1)
vs = sum(1 for t in sb.values() if title(tasks[t["transfer_task_id"]]) in tb)
vv = sum(c - 1 for c in tv.values() if c > 1)
vown = sum(1 for t in sb.values() if title(tasks[t["transfer_task_id"]]) == title(t))
print(f"INFO science: session tasks sharing an abstract with another session task: {ss}")
if vs:
    hard.append(f"{vs} science variants reuse an abstract that a session task uses (transfer contamination) -> run repair_science_variants.py")
if vv:
    hard.append(f"{vv} science variants share an abstract with another variant -> run repair_science_variants.py")
if vown:
    hard.append(f"{vown} science variants use the same abstract as their own base task")

# 4. hints
hp = D / "hints.jsonl"
if hp.exists():
    h = [json.loads(l) for l in open(hp, encoding="utf-8") if l.strip()]
    need = {(k, o["op"], lv) for k, t in base.items() for o in t["operations"] for lv in (1, 2, 3, 4)}
    have = Counter((x["task_id"], x["op"], x["level"]) for x in h)
    miss = need - set(have)
    extra = set(have) - need
    print(f"INFO hints: {len(h)} lines | missing {len(miss)} | extra {len(extra)} | sources {dict(Counter(x['source'] for x in h))}")
    if miss:
        hard.append(f"{len(miss)} (task, op, level) combinations have no hint")
    if any(x["task_id"].endswith("_T") for x in h):
        hard.append("hints exist for transfer variants (should be unaided)")

print()
for m in hard:
    print("HARD FAIL:", m)
print("OK" if not hard else f"{len(hard)} hard failure(s)")
sys.exit(1 if hard else 0)