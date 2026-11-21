#!/usr/bin/env python3
"""Manage an exploratory LLM-assisted audit. NEVER call this human agreement.

Run from repository root:
  python data/scripts/audit_llm_assisted.py --status
  python data/scripts/audit_llm_assisted.py --import-json path/to/model_labels.json --model-id MODEL_ID
  python data/scripts/audit_llm_assisted.py --report

JSON input must be an array with 40 objects, one per fixed sample ID:
{"task_id":"code_021","understand_plan":1,"execute":1,
 "debug_verify":0,"explain":0,"difficulty":2,
 "review_status":"model_only","notes":"optional"}
or a top-level {"labels":[...]} object.

review_status must be "model_only" or "human_reviewed"; "human_reviewed"
must be accurate and is still NOT an independent blind human annotation.
"""
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

DATA = Path(__file__).resolve().parents[1]
ROOT = DATA.parent
SHEET = DATA / "audit" / "annotations_llm_assisted.csv"
SAMPLE = DATA / "audit" / "sample_ids.json"
FIELDS = ("task_id", "domain", "understand_plan", "execute", "debug_verify",
          "explain", "difficulty", "model_id", "review_status", "notes")
OPS = ("understand_plan", "execute", "debug_verify", "explain")


def ids_and_domains():
    ids = json.loads(SAMPLE.read_text(encoding="utf-8"))
    if len(ids) != 40 or len(ids) != len(set(ids)):
        raise ValueError("The frozen sample must have 40 unique task IDs")
    domains = {tid: "coding" if tid.startswith("code_") else "science"
               if tid.startswith("sci_") else "" for tid in ids}
    if any(not d for d in domains.values()):
        raise ValueError("Unexpected ID in frozen sample")
    return ids, domains


def validate_labels(rows, ids, domains, *, allow_incomplete):
    if len(rows) != 40 or {r.get("task_id") for r in rows} != set(ids):
        raise ValueError("Exactly the 40 frozen task IDs are required")
    if len({r.get("task_id") for r in rows}) != len(rows):
        raise ValueError("Duplicate task IDs")
    for row in rows:
        tid = row["task_id"]
        if row.get("domain") != domains[tid]:
            raise ValueError(f"{tid}: incorrect domain")
        values = [str(row.get(c, "")).strip() for c in (*OPS, "difficulty")]
        empty = all(v == "" for v in values)
        if empty and allow_incomplete:
            if any(str(row.get(c, "")).strip() for c in ("model_id","review_status")):
                raise ValueError(f"{tid}: empty labels but nonempty provenance")
            continue
        if any(v not in ("0", "1") for v in values[:4]) or values[4] not in ("1", "2", "3"):
            raise ValueError(f"{tid}: labels must be four 0/1 flags and difficulty 1/2/3")
        if not str(row.get("model_id", "")).strip():
            raise ValueError(f"{tid}: model_id required")
        if row.get("review_status") not in ("model_only", "human_reviewed"):
            raise ValueError(f"{tid}: review_status must be model_only or human_reviewed")


def load_sheet(path=SHEET, *, allow_incomplete=True):
    ids, domains = ids_and_domains()
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if tuple(reader.fieldnames or []) != FIELDS:
            raise ValueError("CSV columns do not match the AI-assisted template")
        rows = list(reader)
    validate_labels(rows, ids, domains, allow_incomplete=allow_incomplete)
    return rows


def import_json(src, model_id, target=SHEET):
    ids, domains = ids_and_domains()
    if not model_id.strip():
        raise ValueError("--model-id is required when importing")
    raw = json.loads(Path(src).read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("labels")
    if not isinstance(raw, list):
        raise ValueError("JSON must contain an array of 40 label objects")
    if len(raw) != 40:
        raise ValueError(f"Expected 40 objects; got {len(raw)}")
    rows = []
    for r in raw:
        if not isinstance(r, dict):
            raise ValueError("Each annotation must be an object")
        tid = r.get("task_id")
        if tid not in domains:
            raise ValueError(f"Unexpected ID {tid!r}")
        row = {"task_id": tid, "domain": domains[tid],
               **{o: str(r.get(o, "")) for o in OPS},
               "difficulty": str(r.get("difficulty", "")),
               "model_id": model_id,
               "review_status": r.get("review_status", "model_only"),
               "notes": str(r.get("notes", ""))}
        rows.append(row)
    validate_labels(rows, ids, domains, allow_incomplete=False)
    by_id = {r["task_id"]: r for r in rows}
    if len(by_id) != len(rows):
        raise ValueError("Duplicate IDs in JSON")
    ordered = [by_id[tid] for tid in ids]
    if Path(target).exists() and any(
        str(r.get("model_id", "")).strip() for r in load_sheet(target)):
        raise ValueError("Refusing to overwrite populated audit CSV; save a backup first")
    with Path(target).open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(ordered)
    print(f"Imported {len(ordered)} LLM-assisted ratings -> {target}")


def agreement(a, b):
    return sum(x == y for x, y in zip(a, b)) / len(a) if a else 0.0


def report(rows):
    ids, _ = ids_and_domains()
    heur = {}
    for line in (DATA / "tasks.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if obj["task_id"] in ids:
            ops = {x["op"] for x in obj["operations"]}
            heur[obj["task_id"]] = {**{o: int(o in ops) for o in OPS},
                                    "difficulty": obj["difficulty"]}
    if len(heur) != 40:
        raise ValueError("Frozen sample is not fully present in the dataset")
    lines = [
        "# Exploratory LLM-assisted audit vs frozen heuristic labels",
        "",
        "**NOT an independent human annotation study.** Agreement here is",
        "model-assisted vs heuristic agreement, not human reliability or gold accuracy.",
        "",
        "## Provenance",
        "",
    ]
    for name, count in sorted(Counter(r["model_id"] for r in rows).items()):
        lines.append(f"- Model: {name}; tasks: {count}")
    for name, count in sorted(Counter(r["review_status"] for r in rows).items()):
        lines.append(f"- Review status: {name}; tasks: {count}")
    lines += ["", "## Agreement by domain", "",
              "| Domain | N | Plan | Execute | Debug/verify | Explain | Difficulty exact |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for domain in ("coding", "science"):
        part = [r for r in rows if r["domain"] == domain]
        rates = [
            agreement([int(r[o]) for r in part],
                      [heur[r["task_id"]][o] for r in part]) for o in OPS
        ]
        diff = agreement([int(r["difficulty"]) for r in part],
                         [heur[r["task_id"]]["difficulty"] for r in part])
        lines.append(f"| {domain} | {len(part)} | " +
                     " | ".join(f"{x:.1%}" for x in (*rates, diff)) + " |")
    lines += ["", "## Interpretation", "",
              "- Differences identify items for expert review; agreement cannot prove label correctness.",
              "- Outputs depend on the model, rubric, prompt, and human review process.",
              "- Do not report these figures as inter-annotator agreement between two humans.",
              "- Independent human validation remains unavailable/pending.", ""]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sheet", type=Path, default=SHEET)
    p.add_argument("--status", action="store_true")
    p.add_argument("--report", action="store_true")
    p.add_argument("--import-json", type=Path)
    p.add_argument("--model-id")
    a = p.parse_args()
    try:
        if a.import_json:
            import_json(a.import_json, a.model_id or "", a.sheet)
        else:
            rows = load_sheet(a.sheet, allow_incomplete=not a.report)
            done = [r for r in rows if str(r["model_id"]).strip()]
            print(f"LLM-assisted audit: {len(done)}/40 rated (independent-human audit: not conducted)")
            if a.report:
                if len(done) != 40:
                    raise ValueError("Report requires all 40 complete AI-assisted records")
                target = DATA / "audit" / "llm_assisted_report.md"
                out = report(rows)
                target.write_text(out, encoding="utf-8")
                print(out)
                print(f"Report: {target}")
    except (OSError, ValueError, json.JSONDecodeError, KeyError) as exc:
        p.exit(1, f"Audit error: {exc}\n")


if __name__ == "__main__":
    main()
