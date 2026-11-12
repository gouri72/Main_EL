#!/usr/bin/env python3
"""Write or check the P3 dataset freeze manifest, without changing dataset files.

Usage:
  python data/scripts/freeze_manifest.py
  python data/scripts/freeze_manifest.py --verify

Only the six data artifacts below are frozen. Annotation CSVs and experiment
outputs are intentionally excluded because their work is ongoing.
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = (
    "tasks.jsonl",
    "tasks_coding.jsonl",
    "tasks_science.jsonl",
    "solutions.jsonl",
    "hints.jsonl",
    "build_config.yaml",
)
MANIFEST = ROOT / "stats" / "p3_freeze_manifest.json"


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def count_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return sum(bool(line.strip()) for line in f)


def snapshot(data_root=ROOT):
    """Return content hashes and line counts; never mutate input files."""
    entries = {}
    for rel in FILES:
        path = data_root / rel
        if not path.is_file():
            raise FileNotFoundError(f"required frozen artifact missing: {path}")
        entry = {"sha256": digest(path), "bytes": path.stat().st_size}
        if path.suffix == ".jsonl":
            entry["records"] = count_jsonl(path)
        entries[rel] = entry
    return {"format_version": 1, "purpose": "P3 frozen task/hint dataset",
            "files": entries}


def verify(expected, current):
    issues = []
    for rel in FILES:
        before = expected.get("files", {}).get(rel)
        after = current["files"][rel]
        if before is None:
            issues.append(f"{rel}: absent from manifest")
        elif before != after:
            issues.append(f"{rel}: changed (expected {before['sha256']}, got {after['sha256']})")
    if set(expected.get("files", {})) != set(FILES):
        issues.append("manifest file list differs from the required artifact list")
    return issues


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="only check hashes; do not overwrite manifest")
    parser.add_argument("--output", type=Path, default=MANIFEST)
    args = parser.parse_args()
    current = snapshot()
    if args.verify:
        if not args.output.exists():
            parser.error(f"manifest missing: {args.output}; first run without --verify")
        expected = json.loads(args.output.read_text(encoding="utf-8"))
        issues = verify(expected, current)
        if issues:
            for issue in issues:
                print("MISMATCH:", issue)
            raise SystemExit(1)
        print("P3 FREEZE VERIFIED: all six files match recorded hashes")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")
    print("P3 manifest created:", args.output)
    for rel, entry in current["files"].items():
        print(f"{rel}: {entry['sha256'][:12]} | {entry.get('records', '-')} records")


if __name__ == "__main__":
    main()
