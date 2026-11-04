#!/usr/bin/env python3
"""Merge completed per-domain hint runs and verify full task/op/level coverage.

Example:
  python assist/merge_hint_runs.py \
    --hints coding=data/hints_coding.jsonl \
    --hints science=data/hints_science.jsonl \
    --raw coding=assist/cache/raw_coding.jsonl \
    --raw science=assist/cache/raw_science.jsonl \
    --stats coding=assist/cache/stats_coding.json \
    --stats science=assist/cache/stats_science.json
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from assist.validator import load_tasks


def named_paths(values, flag):
    out = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"{flag} expects DOMAIN=PATH, got {value!r}")
        name, path = value.split("=", 1)
        if name in out:
            raise SystemExit(f"duplicate {flag} domain: {name}")
        out[name] = Path(path)
    if set(out) != {"coding", "science"}:
        raise SystemExit(f"{flag} needs exactly coding=... and science=...")
    return out


def read_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def key(row):
    return row["task_id"], row["op"], int(row["level"])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hints", action="append", required=True, metavar="DOMAIN=PATH")
    ap.add_argument("--raw", action="append", required=True, metavar="DOMAIN=PATH")
    ap.add_argument("--stats", action="append", required=True, metavar="DOMAIN=PATH")
    ap.add_argument("--tasks", default="data/tasks.jsonl")
    ap.add_argument("--hints-out", default="data/hints.jsonl")
    ap.add_argument("--raw-out", default="assist/cache/raw_first_pass.jsonl")
    ap.add_argument("--stats-out", default="assist/cache/stats.json")
    a = ap.parse_args()

    hint_paths = named_paths(a.hints, "--hints")
    raw_paths = named_paths(a.raw, "--raw")
    stat_paths = named_paths(a.stats, "--stats")
    tasks = load_tasks(ROOT / a.tasks)

    expected_by_domain = {"coding": set(), "science": set()}
    for task in tasks.values():
        if task["task_id"].endswith("_T"):
            continue
        expected_by_domain[task["domain"]].update(
            (task["task_id"], op["op"], level)
            for op in task["operations"] for level in (1, 2, 3, 4)
        )

    hints = []
    for domain, path in hint_paths.items():
        rows = read_jsonl(path)
        actual = [key(row) for row in rows]
        if len(actual) != len(set(actual)):
            raise SystemExit(f"duplicate task/op/level rows in {domain} hint file")
        if set(actual) != expected_by_domain[domain]:
            missing = len(expected_by_domain[domain] - set(actual))
            extra = len(set(actual) - expected_by_domain[domain])
            raise SystemExit(f"{domain} hint coverage mismatch: missing={missing}, extra={extra}")
        if any(tasks[row["task_id"]]["domain"] != domain for row in rows):
            raise SystemExit(f"wrong-domain task found in {domain} hint file")
        hints.extend(rows)

    hints.sort(key=key)
    write_jsonl(ROOT / a.hints_out, hints)

    raw = []
    raw_keys = set()
    for domain, path in raw_paths.items():
        rows = read_jsonl(path)
        for row in rows:
            k = key(row)
            if k in raw_keys:
                raise SystemExit(f"duplicate raw first-pass entry: {k}")
            if k not in expected_by_domain[domain]:
                raise SystemExit(f"unexpected raw entry for {domain}: {k}")
            raw_keys.add(k)
            raw.append(row)
    raw.sort(key=key)
    write_jsonl(ROOT / a.raw_out, raw)

    stats = [json.loads(path.read_text(encoding="utf-8")) for path in stat_paths.values()]
    merged = {
        "domains": {domain: json.loads(path.read_text(encoding="utf-8"))
                    for domain, path in stat_paths.items()},
        "total": sum(s.get("total", 0) for s in stats),
        "first_pass_violations": sum(s.get("first_pass_violations", 0) for s in stats),
        "regenerated_ok": sum(s.get("regenerated_ok", 0) for s in stats),
        "fallback": sum(s.get("fallback", 0) for s in stats),
        "handwritten": sum(s.get("handwritten", 0) for s in stats),
        "missing_cache": sum(s.get("missing_cache", 0) for s in stats),
        "violations_by_level": {
            str(level): sum(s.get("violations_by_level", {}).get(level,
                            s.get("violations_by_level", {}).get(str(level), 0)) for s in stats)
            for level in (1, 2, 3, 4)
        },
    }
    out = ROOT / a.stats_out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")

    print(f"merged hints: {len(hints)} (expected {sum(map(len, expected_by_domain.values()))})")
    print(f"raw first-pass generations: {len(raw)}")
    print(f"hint source counts: { {source: sum(row.get('source') == source for row in hints) for source in ('qwen', 'handwritten', 'fallback')} }")
    print(f"leakage flagged: {sum(bool(row.get('leakage_flag')) for row in hints)}")
    print(f"noncompliant: {sum(not bool(row.get('compliance_flag')) for row in hints)}")
    print(f"wrote {a.hints_out}, {a.raw_out}, and {a.stats_out}")


if __name__ == "__main__":
    main()
