#!/usr/bin/env python3
"""Dataset statistics table for the paper.  python data/scripts/dataset_stats.py
Writes data/stats/dataset_stats.md and dataset_stats.csv."""
import csv, json, re, statistics as st
from collections import Counter
from pathlib import Path

D = Path(__file__).resolve().parents[1]
OUT = D / "stats"
OUT.mkdir(exist_ok=True)
tasks = []
with open(D / "tasks.jsonl", encoding="utf-8") as f:
    tasks = [json.loads(l) for l in f if l.strip()]

OPS = ["understand_plan", "execute", "debug_verify", "explain"]
rows = []


def add(metric, fn):
    rows.append((metric, *[fn(dom) for dom in ("coding", "science")]))


def prim(dom):
    return [t for t in tasks if t["domain"] == dom and not t["task_id"].endswith("_T")]


def var(dom):
    return [t for t in tasks if t["domain"] == dom and t["task_id"].endswith("_T")]


mean = lambda xs: f"{st.mean(xs):.1f}" if xs else "-"
add("session tasks", lambda d: len(prim(d)))
add("transfer variants", lambda d: len(var(d)))
for k in (1, 2, 3):
    add(f"difficulty {k} (session tasks)", lambda d, k=k: sum(t["difficulty"] == k for t in prim(d)))
add("prompt length, mean chars", lambda d: mean([len(t["prompt"]) for t in prim(d)]))
for o in OPS:
    add(f"tasks containing op: {o}", lambda d, o=o: sum(any(x["op"] == o for x in t["operations"]) for t in prim(d)))
add("op labels from gold", lambda d: sum(x["label_source"] == "gold" for t in prim(d) for x in t["operations"]))
add("op labels heuristic", lambda d: sum(x["label_source"] == "heuristic" for t in prim(d) for x in t["operations"]))
add("coding: mean tests per task", lambda d: mean([t["gold"]["tests"].count("assert") for t in prim(d)]) if d == "coding" else "-")
add("science: SUPPORT", lambda d: sum(t["gold"]["label"] == "SUPPORT" for t in prim(d)) if d == "science" else "-")
add("science: CONTRADICT", lambda d: sum(t["gold"]["label"] == "CONTRADICT" for t in prim(d)) if d == "science" else "-")
add("science: mean evidence sentences", lambda d: mean([len(t["gold"]["evidence"]) for t in prim(d)]) if d == "science" else "-")
add("science: mean abstract sentences", lambda d: mean([len(re.findall(r"(?m)^\[\d+\]", t["prompt"])) for t in prim(d)]) if d == "science" else "-")
for d in ("coding", "science"):
    c = Counter(e for t in prim(d) for e in t["error_types"])
    add(f"error types ({d}) top 5", lambda dom, c=c, d=d: ", ".join(f"{k}:{v}" for k, v in c.most_common(5)) if dom == d else "-")

with open(OUT / "dataset_stats.csv", "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(["metric", "coding", "science"])
    w.writerows(rows)
md = ["| metric | coding | science |", "|---|---|---|"] + [f"| {m} | {a} | {b} |" for m, a, b in rows]
(OUT / "dataset_stats.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print("\n".join(md))
print("\nwrote data/stats/dataset_stats.md and dataset_stats.csv")