#!/usr/bin/env python3
"""Agreement between two annotators and the heuristic labels in tasks.jsonl.

  python data/scripts/audit_agreement.py data/audit/annotations_P3.csv data/audit/annotations_P4.csv

Per domain and per operation: % agreement and Cohen's kappa (n/a when a label never varies).
Difficulty: exact %, within-1 %, kappa. Writes data/audit/audit_report.md.
"""
import csv, json, sys
from pathlib import Path

D = Path(__file__).resolve().parents[1]
OPS = ["understand_plan", "execute", "debug_verify", "explain"]


def read_sheet(path):
    rows = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            try:
                ops = {o: int(r[o]) for o in OPS}
                diff = int(r["difficulty"])
            except (ValueError, KeyError, TypeError):
                sys.exit(f"{path}: row {r.get('task_id')} has a blank/invalid cell (ops must be 0/1, difficulty 1-3)")
            if any(v not in (0, 1) for v in ops.values()) or diff not in (1, 2, 3):
                sys.exit(f"{path}: row {r['task_id']} has out-of-range values")
            rows[r["task_id"]] = {"domain": r["domain"], "diff": diff, **ops}
    return rows


def kappa(x, y):
    n = len(x)
    po = sum(a == b for a, b in zip(x, y)) / n
    cats = set(x) | set(y)
    pe = sum((x.count(c) / n) * (y.count(c) / n) for c in cats)
    return None if pe == 1 else (po - pe) / (1 - pe)


def fmt(x, y):
    n = len(x)
    ag = 100 * sum(a == b for a, b in zip(x, y)) / n
    k = kappa(x, y)
    return f"{ag:.0f}% / " + ("n/a" if k is None else f"{k:.2f}")


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    A, B = read_sheet(sys.argv[1]), read_sheet(sys.argv[2])
    heur = {}
    with open(D / "tasks.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                t = json.loads(line)
                present = {o["op"] for o in t["operations"]}
                heur[t["task_id"]] = {"domain": t["domain"], "diff": t["difficulty"],
                                      **{o: int(o in present) for o in OPS}}
    ids = sorted(set(A) & set(B))
    if set(A) != set(B):
        print(f"WARNING: sheets differ; using the {len(ids)} common tasks")
    sources = {"A": A, "B": B, "H": heur}
    names = {"A": "annotator1", "B": "annotator2", "H": "heuristic"}
    out = ["# Label-noise audit", "", f"{len(ids)} tasks. Cells: % agreement / Cohen's kappa. "
           "Small sample: treat kappa as indicative only.", ""]
    for dom in sorted({A[i]["domain"] for i in ids}):
        sub = [i for i in ids if A[i]["domain"] == dom]
        out += [f"## {dom} (n={len(sub)})", "",
                "| pair | " + " | ".join(OPS) + " | difficulty exact | within 1 | difficulty kappa |",
                "|---|" + "---|" * (len(OPS) + 3)]
        for p, q in (("A", "B"), ("A", "H"), ("B", "H")):
            row = [fmt([sources[p][i][o] for i in sub], [sources[q][i][o] for i in sub]) for o in OPS]
            dp, dq = [sources[p][i]["diff"] for i in sub], [sources[q][i]["diff"] for i in sub]
            exact = 100 * sum(a == b for a, b in zip(dp, dq)) / len(sub)
            w1 = 100 * sum(abs(a - b) <= 1 for a, b in zip(dp, dq)) / len(sub)
            k = kappa(dp, dq)
            out.append(f"| {names[p]} vs {names[q]} | " + " | ".join(row) +
                       f" | {exact:.0f}% | {w1:.0f}% | {'n/a' if k is None else f'{k:.2f}'} |")
        out.append("")
    text = "\n".join(out)
    (D / "audit" / "audit_report.md").write_text(text, encoding="utf-8")
    print(text)
    print("\nwrote data/audit/audit_report.md")


if __name__ == "__main__":
    main()