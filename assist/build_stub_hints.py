#!/usr/bin/env python3
"""Day-1 stub: write data/hints.jsonl with a complete ladder (levels 1-4) for every op of every BASE task
(transfer variants *_T are unaided, so they need no hints). Source is 'fallback' (templates + gold at L4).
Handwritten entries in data/hints_handwritten.jsonl override these (source 'handwritten').
  python assist/build_stub_hints.py"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from assist.validator import validate, load_tasks, load_solutions
from assist.fallback import fallback_text

ROOT = Path(__file__).resolve().parents[1]
T, S = load_tasks(), load_solutions()
hand = {}
hp = ROOT / "data" / "hints_handwritten.jsonl"
if hp.exists():
    for l in open(hp, encoding="utf-8"):
        if l.strip():
            h = json.loads(l); hand[(h["task_id"], h["op"], h["level"])] = h["text"]

rows = []
for tid, t in T.items():
    if tid.endswith("_T"): continue
    for o in t["operations"]:
        for lv in (1, 2, 3, 4):
            key = (tid, o["op"], lv)
            src = "handwritten" if key in hand else "fallback"
            text = hand.get(key) or fallback_text(t, o["op"], lv, S.get(tid))
            v = validate(t, o["op"], lv, text, S.get(tid))
            rows.append({"task_id": tid, "op": o["op"], "level": lv, "text": text,
                         "leakage_flag": v["leakage_flag"], "compliance_flag": v["compliance_flag"], "source": src})
with open(ROOT / "data" / "hints.jsonl", "w", encoding="utf-8", newline="\n") as f:
    for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
bad = [r for r in rows if r["leakage_flag"] or not r["compliance_flag"]]
print(f"wrote {len(rows)} hints; {len(bad)} flagged; handwritten={sum(r['source']=='handwritten' for r in rows)}")
for r in bad: print("  FLAGGED", r["task_id"], r["op"], r["level"])