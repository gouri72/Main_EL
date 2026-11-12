#!/usr/bin/env python3
"""Independent P3 human annotation CLI. No heuristic labels are displayed.

Examples (run from repo root):
  python data/scripts/annotate_tasks.py --status
  python data/scripts/annotate_tasks.py --sheet data/audit/annotations_P3.csv
  python data/scripts/annotate_tasks.py --sheet data/audit/annotations_second.csv --init
  python data/scripts/annotate_tasks.py --sheet data/audit/annotations_second.csv

Only human-entered labels are saved; existing completed rows are not overwritten
without --revise. It never writes dataset tasks, hints or solutions.
"""
import argparse
import csv
import json
import os
import re
import tempfile
from pathlib import Path

DATA = Path(__file__).resolve().parents[1]
FIELDS = ("task_id", "domain", "understand_plan", "execute", "debug_verify", "explain", "difficulty")
LABELS = FIELDS[2:]
BLOCK_RE = re.compile(r"(?m)^={50,}\s*\n(code_\d{3}|sci_\d{3})\s+\[(coding|science)\]\s*\n={50,}\s*\n")


def sample():
    ids = json.loads((DATA / "audit" / "sample_ids.json").read_text(encoding="utf-8"))
    if len(ids) != 40 or len(ids) != len(set(ids)):
        raise ValueError("Expected exactly 40 distinct fixed sample IDs")
    return ids


def question_blocks():
    """Read only the human-facing audit text, not tasks.jsonl or heuristic labels."""
    text = (DATA / "audit" / "audit_items.txt").read_text(encoding="utf-8")
    found = list(BLOCK_RE.finditer(text))
    blocks = {}
    for i, m in enumerate(found):
        tid, domain = m.group(1), m.group(2)
        end = found[i + 1].start() if i + 1 < len(found) else len(text)
        if tid in blocks:
            raise ValueError(f"Duplicate question {tid} in audit_items.txt")
        blocks[tid] = (domain, text[m.end():end].strip())
    ids = sample()
    if set(blocks) != set(ids):
        raise ValueError(f"Fixed questions do not match sample IDs: missing={set(ids)-set(blocks)}, extra={set(blocks)-set(ids)}")
    return blocks


def completed(row):
    return all((row.get(field) or "").strip() in (("0", "1") if field != "difficulty" else ("1", "2", "3"))
               for field in LABELS)


def read_rows(path):
    """Accept partial sheets but reject invalid populated cells and IDs."""
    with path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValueError(f"{path}: expected columns {FIELDS!r}")
        rows = {}
        for row in reader:
            tid = row["task_id"]
            if tid in rows:
                raise ValueError(f"Duplicate annotation ID {tid!r}")
            if any(row.get(k) is None for k in FIELDS):
                raise ValueError(f"{tid}: malformed row")
            for field in LABELS:
                value = (row[field] or "").strip()
                allowed = ("1", "2", "3") if field == "difficulty" else ("0", "1")
                if value and value not in allowed:
                    raise ValueError(f"{tid}: invalid {field}={value!r}")
            rows[tid] = row
        return rows


def validate_rows(rows, blocks, ids):
    if set(rows) != set(ids):
        raise ValueError(f"Annotation sheet must have the 40 fixed IDs; missing={sorted(set(ids)-set(rows))}, extra={sorted(set(rows)-set(ids))}")
    for tid, row in rows.items():
        if row["domain"] != blocks[tid][0]:
            raise ValueError(f"{tid}: wrong domain; expected {blocks[tid][0]}")


def new_rows(ids, blocks):
    return {tid: {"task_id": tid, "domain": blocks[tid][0],
                  **{field: "" for field in LABELS}} for tid in ids}


def save_rows(path, rows, ids):
    """Atomic, order-preserving write to avoid partially written CSVs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="",
                                         dir=path.parent, prefix=".annotation-", suffix=".tmp",
                                         delete=False) as fh:
            tmp = Path(fh.name)
            writer = csv.DictWriter(fh, fieldnames=FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows[tid] for tid in ids)
        os.replace(tmp, path)
    finally:
        if tmp is not None and tmp.exists():
            tmp.unlink()


def status(rows, ids):
    total = sum(completed(rows[tid]) for tid in ids)
    coding = sum(completed(rows[tid]) for tid in ids if tid.startswith("code_"))
    science = total - coding
    remaining = [tid for tid in ids if not completed(rows[tid])]
    return total, coding, science, remaining


def prompt_rating(name):
    choices = ("1", "2", "3") if name == "difficulty" else ("0", "1")
    while True:
        value = input(f"  {name} [{'/'.join(choices)}] (q to stop): ").strip().lower()
        if value == "q":
            return None
        if value in choices:
            return value
        print("  Invalid value; choose a listed number or q.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sheet", type=Path, default=DATA / "audit" / "annotations_P3.csv")
    parser.add_argument("--init", action="store_true", help="create a new blank sheet; refuse overwrite")
    parser.add_argument("--status", action="store_true", help="show progress only; never prompt")
    parser.add_argument("--revise", action="store_true", help="also display already completed questions")
    args = parser.parse_args()
    ids = sample()
    blocks = question_blocks()
    if args.init:
        if args.sheet.exists():
            parser.error(f"refusing to overwrite existing sheet {args.sheet}")
        save_rows(args.sheet, new_rows(ids, blocks), ids)
        print(f"Created blank sheet: {args.sheet}")
        return
    if not args.sheet.is_file():
        parser.error(f"sheet missing: {args.sheet}. Use --init for a new independent sheet.")
    rows = read_rows(args.sheet)
    validate_rows(rows, blocks, ids)
    total, coding, science, remaining = status(rows, ids)
    print(f"{args.sheet}: {total}/{len(ids)} complete | coding {coding}/20 | science {science}/20")
    if args.status:
        print("Remaining:", ", ".join(remaining) if remaining else "none")
        return
    if not args.revise and not remaining:
        print("All 40 rows completed. No changes made.")
        return
    print("Your ratings must be independent. No heuristic labels are displayed.")
    print("Refer to data/audit/annotation_protocol.md for the definitions.")
    for index, tid in enumerate(ids, 1):
        if completed(rows[tid]) and not args.revise:
            continue
        domain, task_text = blocks[tid]
        print(f"\n{'=' * 72}\n{index}/40 {tid} ({domain})\n{'=' * 72}\n{task_text}\n")
        candidate = {}
        for field in LABELS:
            value = prompt_rating(field)
            if value is None:
                print("Stopped; any previously finished rows remain saved.")
                return
            candidate[field] = value
        rows[tid].update(candidate)
        save_rows(args.sheet, rows, ids)
        print(f"Saved {tid}. Progress: {status(rows, ids)[0]}/40")
    print("All selected questions completed. Use audit_agreement.py after both annotators finish.")


if __name__ == "__main__":
    main()
