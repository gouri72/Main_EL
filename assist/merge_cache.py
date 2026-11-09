#!/usr/bin/env python3
"""Merge hint-cache files (e.g. one per shard) into assist/cache/hints_cache.jsonl.
  python assist/merge_cache.py cache_shard0.jsonl cache_shard1.jsonl cache_shard2.jsonl cache_shard3.jsonl
Keeps what is already in the output cache, adds new keys, reports duplicates and conflicts (same key, different text:
the first one is kept). Malformed lines are skipped and counted. Safe to run repeatedly."""
import argparse, json, os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("files", nargs="+")
ap.add_argument("--out", default=str(ROOT / "assist" / "cache" / "hints_cache.jsonl"))
a = ap.parse_args()

def read(path):
    rows, bad = [], 0
    for l in open(path, encoding="utf-8"):
        if not l.strip(): continue
        try:
            d = json.loads(l); rows.append((d["key"], d["text"]))
        except (ValueError, KeyError, TypeError):
            bad += 1
    return rows, bad

out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
merged = {}
if out.exists():
    rows, bad = read(out); merged.update(rows)
    print(f"existing {out.name}: {len(merged)} entries ({bad} bad lines)")
for f in a.files:
    rows, bad = read(f); new = dup = conf = 0
    for k, t in rows:
        if k not in merged: merged[k] = t; new += 1
        elif merged[k] == t: dup += 1
        else: conf += 1
    print(f"{Path(f).name}: {len(rows)} lines, {new} new, {dup} identical duplicates, {conf} conflicts kept-first, {bad} bad")
tmp = out.with_suffix(".tmp")
with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
    for k, t in merged.items(): fh.write(json.dumps({"key": k, "text": t}, ensure_ascii=False) + "\n")
os.replace(tmp, out)
print(f"wrote {len(merged)} entries -> {out}")