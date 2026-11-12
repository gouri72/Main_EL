#!/usr/bin/env python3
"""Independent P3 annotation agreement: per-operation agreement/kappa and difficulty.

Usage:
  python data/scripts/audit_agreement.py data/audit/annotations_P3.csv data/audit/annotations_second.csv

Requires fully completed, non-duplicate sheets with exactly the sampled task IDs.
Does not create or infer human annotations.
"""
import csv
import json
import sys
from pathlib import Path

D = Path(__file__).resolve().parents[1]
OPS = ["understand_plan", "execute", "debug_verify", "explain"]


def read_sheet(path):
    rows = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"task_id", "domain", *OPS, "difficulty"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path}: missing required columns {sorted(required - set(reader.fieldnames or []))}")
        for line, r in enumerate(reader, 2):
            tid = (r.get("task_id") or "").strip()
            if not tid:
                raise ValueError(f"{path}:{line}: empty task_id")
            if tid in rows:
                raise ValueError(f"{path}:{line}: duplicate task_id {tid}")
            try:
                ops = {o: int(r[o]) for o in OPS}
                diff = int(r["difficulty"])
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f"{path}:{line} ({tid}): blank or invalid annotation") from exc
            if any(v not in (0, 1) for v in ops.values()) or diff not in (1, 2, 3):
                raise ValueError(f"{path}:{line} ({tid}): operations must be 0/1, difficulty 1/2/3")
            domain = r["domain"].strip()
            if domain not in ("coding", "science"):
                raise ValueError(f"{path}:{line} ({tid}): unexpected domain {domain!r}")
            rows[tid] = {"domain": domain, "diff": diff, **ops}
    return rows


def kappa(x, y):
    if not x or len(x) != len(y):
        raise ValueError("kappa requires equal-length nonempty inputs")
    n = len(x)
    po = sum(a == b for a, b in zip(x, y)) / n
    cats = set(x) | set(y)
    pe = sum((x.count(c) / n) * (y.count(c) / n) for c in cats)
    return None if abs(1 - pe) < 1e-12 else (po - pe) / (1 - pe)


def fmt(x, y):
    n = len(x)
    ag = 100 * sum(a == b for a, b in zip(x, y)) / n
    k = kappa(x, y)
    return f"{ag:.0f}% / " + ("n/a" if k is None else f"{k:.2f}")


def validate_sheets(a, b, heur, sample_ids):
    expected = set(sample_ids)
    if len(sample_ids) != len(expected):
        raise ValueError("sample_ids.json contains duplicates")
    if set(a) != expected or set(b) != expected:
        raise ValueError(
            "annotation sheets must match sample_ids.json exactly; "
            f"A missing={sorted(expected - set(a))} extra={sorted(set(a) - expected)}; "
            f"B missing={sorted(expected - set(b))} extra={sorted(set(b) - expected)}"
        )
    for tid in expected:
        if tid not in heur:
            raise ValueError(f"sample task {tid} missing from tasks.jsonl")
        if a[tid]["domain"] != b[tid]["domain"] or a[tid]["domain"] != heur[tid]["domain"]:
            raise ValueError(f"{tid}: domain mismatch between annotations and dataset")


def make_report(a, b, heur, sample_ids):
    validate_sheets(a, b, heur, sample_ids)
    ids = sorted(sample_ids)
    sources = {"A": a, "B": b, "H": heur}
    names = {"A": "annotator1", "B": "annotator2", "H": "heuristic"}
    out = ["# Independent operation and difficulty audit", "",
           f"{len(ids)} fully annotated tasks. Cells: agreement / Cohen's kappa. ",
           "Kappa is n/a if expected chance agreement is 1; interpret small-sample results cautiously.", ""]
    for dom in ("coding", "science"):
        sub = [i for i in ids if a[i]["domain"] == dom]
        if not sub:
            continue
        out += [f"## {dom} (n={len(sub)})", "",
                "| pair | " + " | ".join(OPS) + " | difficulty exact | within 1 | difficulty kappa |",
                "|---|" + "---|" * (len(OPS) + 3)]
        for p, q in (("A", "B"), ("A", "H"), ("B", "H")):
            row = [fmt([sources[p][i][o] for i in sub], [sources[q][i][o] for i in sub]) for o in OPS]
            dp = [sources[p][i]["diff"] for i in sub]
            dq = [sources[q][i]["diff"] for i in sub]
            exact = 100 * sum(x == y for x, y in zip(dp, dq)) / len(sub)
            within_one = 100 * sum(abs(x - y) <= 1 for x, y in zip(dp, dq)) / len(sub)
            k = kappa(dp, dq)
            out.append(f"| {names[p]} vs {names[q]} | " + " | ".join(row) +
                       f" | {exact:.0f}% | {within_one:.0f}% | {'n/a' if k is None else f'{k:.2f}'} |")
        out.append("")
    out += ["## Methodological notes", "",
            "- Both human sheets must be completed independently and cannot contain inferred/AI-filled labels presented as human.",
            "- Operation annotations are multi-label; kappa is calculated separately for each operation.",
            "- The dataset's heuristic labels are a reference, not a ground-truth human gold standard.", ""]
    return "\n".join(out)


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    try:
        a, b = read_sheet(sys.argv[1]), read_sheet(sys.argv[2])
        sample_ids = json.loads((D / "audit" / "sample_ids.json").read_text(encoding="utf-8"))
        tasks = [json.loads(l) for l in (D / "tasks.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        heur = {}
        for t in tasks:
            present = {x["op"] for x in t["operations"]}
            heur[t["task_id"]] = {"domain": t["domain"], "diff": t["difficulty"],
                                  **{o: int(o in present) for o in OPS}}
        report = make_report(a, b, heur, sample_ids)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        sys.exit(f"Annotation validation failed: {exc}")
    path = D / "audit" / "audit_report.md"
    path.write_text(report, encoding="utf-8")
    print(report)
    print(f"Report: {path}")


if __name__ == "__main__":
    main()
