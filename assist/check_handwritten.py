#!/usr/bin/env python3
"""After data changes: check every handwritten hint still points at a real task + operation and passes the validator.
  python assist/check_handwritten.py        (exit code 1 if anything needs attention)"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from assist.validator import validate, load_tasks, load_solutions
ROOT = Path(__file__).resolve().parents[1]
T, S = load_tasks(), load_solutions()
bad = 0
for l in open(ROOT / "data" / "hints_handwritten.jsonl", encoding="utf-8"):
    if not l.strip(): continue
    h = json.loads(l); t = T.get(h["task_id"])
    if t is None:
        print("MISSING TASK", h["task_id"]); bad += 1; continue
    ops = [o["op"] for o in t["operations"]]
    v = validate(t, h["op"], h["level"], h["text"], S.get(h["task_id"]))
    status = "OK" if h["op"] in ops and not v["leakage_flag"] and v["compliance_flag"] else "CHECK"
    if status == "CHECK": bad += 1
    print(f'{status:5} {h["task_id"]} {h["op"]} L{h["level"]} ops={ops} flags={v}')
    print("      task:", t["prompt"][:110].replace("\n", " "))
    print("      hint:", h["text"][:110])
print("ALL OK" if not bad else f"{bad} hint(s) need attention (retarget or rewrite them)"); sys.exit(1 if bad else 0)