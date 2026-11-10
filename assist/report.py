#!/usr/bin/env python3
"""Paste-ready summary of the hint bank for the group chat / paper.  python assist/report.py
Reads assist/cache/stats.json, assist/cache/raw_first_pass.jsonl and data/hints.jsonl (run generate_hints.py --backend cacheonly first)."""
import collections, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from assist.validator import validate, load_tasks, load_solutions
ROOT = Path(__file__).resolve().parents[1]
T, S = load_tasks(), load_solutions()
st = json.load(open(ROOT / "assist/cache/stats.json"))
rows = [json.loads(l) for l in open(ROOT / "data/hints.jsonl", encoding="utf-8") if l.strip()]
raw = [json.loads(l) for l in open(ROOT / "assist/cache/raw_first_pass.jsonl", encoding="utf-8") if l.strip()]
src = collections.Counter(r["source"] for r in rows)
tasks = {r["task_id"] for r in rows}
dom = collections.Counter(T[t]["domain"] for t in tasks)
tot, rej = collections.Counter(), collections.Counter()
for r in raw:
    t = T[r["task_id"]]
    v = validate(t, r["op"], r["level"], r["text"], S.get(r["task_id"]))
    tot[(t["domain"], r["level"])] += 1
    rej[(t["domain"], r["level"])] += bool(v["reasons"] or v["leakage_flag"])
fb = src["fallback"]; n = len(rows)
first_pass = sum(rej.values())
def rates(d): return ", ".join(f"L{l} {rej[(d, l)]}/{tot[(d, l)]} ({rej[(d, l)] / max(tot[(d, l)], 1):.0%})" for l in (1, 2, 3, 4))
print(f"Hint bank: {n} items for {len(tasks)} session tasks ({dom['coding']} coding, {dom['science']} science); "
      f"model {st['model']}, prompt {st['prompt_version']}.")
print(f"Sources: {src['qwen']} validated Qwen3-0.6B ({src['qwen'] / n:.1%}), {src['handwritten']} handwritten, "
      f"{fb} template fallback ({fb / n:.1%}).")
print(f"First-pass validator rejections: {first_pass} of {sum(tot.values())} generated "
      f"({first_pass / max(sum(tot.values()), 1):.1%}); {st['regenerated_ok']} fixed by retry "
      f"(up to 3 attempts, temperature raised each time, rejection reason fed back into the prompt).")
print(f"  coding : {rates('coding')}")
print(f"  science: {rates('science')}")
print(f"missing_cache: {st.get('missing_cache', 'n/a')}")