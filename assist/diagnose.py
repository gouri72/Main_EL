#!/usr/bin/env python3
"""Why does the validator reject first-pass outputs?  python assist/diagnose.py [raw_first_pass.jsonl] [--examples 2]
Counts rejection reasons by (domain, level, reason) and prints example texts for the most common ones. No GPU needed."""
import argparse, collections, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from assist.validator import validate, load_tasks, load_solutions
ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser(); ap.add_argument("raw", nargs="?", default=str(ROOT / "assist/cache/raw_first_pass.jsonl"))
ap.add_argument("--examples", type=int, default=2); a = ap.parse_args()
T, S = load_tasks(), load_solutions()
raw = [json.loads(l) for l in open(a.raw, encoding="utf-8") if l.strip()]
tot, rej, why, ex = collections.Counter(), collections.Counter(), collections.Counter(), collections.defaultdict(list)
for r in raw:
    t = T[r["task_id"]]; d = t["domain"]; lv = r["level"]
    v = validate(t, r["op"], lv, r["text"], S.get(r["task_id"]))
    reasons = v["reasons"] + ["LEAK:" + w for w in v["leak_why"]]
    tot[(d, lv)] += 1
    if reasons:
        rej[(d, lv)] += 1
        for x in reasons:
            why[(d, lv, x)] += 1
            if len(ex[(d, lv, x)]) < a.examples: ex[(d, lv, x)].append((r["task_id"], r["op"], r["text"][:230].replace("\n", " | ")))
print("first-pass rejection rate by (domain, level):")
for k in sorted(tot): print(f"  {k[0]:8} L{k[1]}  {rej[k]:4}/{tot[k]:4} = {rej[k]/tot[k]:.0%}")
print("\nreasons (a text can have several):")
for k, n in sorted(why.items(), key=lambda kv: (kv[0][0], kv[0][1], -kv[1])): print(f"  {k[0]:8} L{k[1]}  {k[2]:22} {n}")
print("\nexamples:")
for k, n in sorted(why.items(), key=lambda kv: -kv[1])[:8]:
    for e in ex[k]: print(f"  [{k[0]} L{k[1]} {k[2]}] {e[0]} {e[1]}: {e[2]}")