#!/usr/bin/env python3
"""Integrity check of data/hints.jsonl against data/tasks.jsonl.  python assist/check_bank.py  (exit 1 on any problem)
Checks: every (base task, operation, level 1-4) present exactly once; no hints for *_T ids; no empty text; stored flags match a
fresh validation; no flagged hint; and reports the source mix (qwen / handwritten / fallback)."""
import collections, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from assist.validator import validate, load_tasks, load_solutions
ROOT = Path(__file__).resolve().parents[1]
T, S = load_tasks(), load_solutions()
rows = [json.loads(l) for l in open(ROOT / "data" / "hints.jsonl", encoding="utf-8") if l.strip()]
expected = {(tid, o["op"], lv) for tid, t in T.items() if not tid.endswith("_T") for o in t["operations"] for lv in (1, 2, 3, 4)}
seen = collections.Counter((r["task_id"], r["op"], r["level"]) for r in rows)
problems = []
missing = expected - set(seen); extra = set(seen) - expected; dups = [k for k, n in seen.items() if n > 1]
if missing: problems.append(f"{len(missing)} missing items, e.g. {sorted(missing)[:3]}")
if extra: problems.append(f"{len(extra)} unexpected items (e.g. transfer ids), e.g. {sorted(extra)[:3]}")
if dups: problems.append(f"{len(dups)} duplicated items")
stale = flagged = empty = 0
for r in rows:
    t = T.get(r["task_id"])
    if t is None or not r["text"].strip():
        empty += 1; continue
    v = validate(t, r["op"], r["level"], r["text"], S.get(r["task_id"]))
    if v["leakage_flag"] != r["leakage_flag"] or v["compliance_flag"] != r["compliance_flag"]: stale += 1
    if v["leakage_flag"] or not v["compliance_flag"]: flagged += 1
if empty: problems.append(f"{empty} empty/unknown rows")
if stale: problems.append(f"{stale} rows whose stored flags differ from a fresh validation (rebuild the bank)")
if flagged: problems.append(f"{flagged} rows fail validation")
src = collections.Counter(r["source"] for r in rows)
print(f"coverage: {len(set(seen) & expected)}/{len(expected)} items | rows {len(rows)} | sources {dict(src)}")
for dom in ("coding", "science"):
    ds = [r for r in rows if T[r["task_id"]]["domain"] == dom] if not extra else []
    if ds: print(f"  {dom}: {len(ds)} hints, fallback share {sum(r['source']=='fallback' for r in ds)/len(ds):.0%}")
print("BANK OK" if not problems else "PROBLEMS:\n  " + "\n  ".join(problems)); sys.exit(1 if problems else 0)