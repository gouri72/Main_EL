#!/usr/bin/env python3
"""Sample ~100 hints for hand labeling. Run AFTER the real Qwen run.
Source = assist/cache/raw_first_pass.jsonl (raw first-attempt Qwen outputs, so violations actually occur).
Falls back to data/hints.jsonl only if the raw file is missing (weaker: those hints were already filtered).
Writes assist/labels/label_sheet.csv with a dev/test split (first 30 dev for tuning rules, rest frozen test).
Annotators fill <name>_leak (1 = hint LEAKS the answer) and <name>_compliant (1 = hint obeys its level)
independently, WITHOUT looking at validator output.   python assist/make_label_sheet.py --n 100 --seed 0"""
import argparse, csv, json, random, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--annotators", default="p4,p3")
a = ap.parse_args()
src = ROOT / "assist" / "cache" / "raw_first_pass.jsonl"
if not src.exists():
    print("WARNING: no raw_first_pass.jsonl, sampling from data/hints.jsonl"); src = ROOT / "data" / "hints.jsonl"
rows = [json.loads(l) for l in open(src, encoding="utf-8") if l.strip()]
rows = [r for r in rows if r["level"] <= 3]      # level 4 is allowed to leak; label L1-L3 only
rng = random.Random(a.seed); rng.shuffle(rows)
rows = rows[:a.n]
n_dev = min(30, round(0.3 * len(rows)))
out = ROOT / "assist" / "labels"; out.mkdir(exist_ok=True)
names = a.annotators.split(",")
cols = ["idx", "split", "task_id", "op", "level", "text"] + [f"{p}_{k}" for p in names for k in ("leak", "compliant")]
with open(out / "label_sheet.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
    for i, r in enumerate(rows):
        w.writerow({"idx": i, "split": "dev" if i < n_dev else "test", "task_id": r["task_id"], "op": r["op"],
                    "level": r["level"], "text": r["text"]})
print(f"wrote {len(rows)} rows to assist/labels/label_sheet.csv (dev={n_dev}, test={len(rows)-n_dev})")