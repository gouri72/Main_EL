#!/usr/bin/env python3
"""Create two separate, identically blinded rating sheets from an existing sample."""
import argparse
import csv
import random
from pathlib import Path


DIMENSIONS = ("relevance", "correctness", "level_fit", "usefulness")


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sheet", required=True, help="existing blind_annotation_sheet.csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=20261124)
    parser.add_argument("--force", action="store_true", help="replace output sheets if they exist")
    args = parser.parse_args()
    source = read_csv(args.sheet)
    if not source:
        raise SystemExit("The source blind sheet is empty; no rater sheets were created.")
    required = {"rating_id", "output_A", "output_B", "task_input", "operation", "level"}
    if not required.issubset(source[0]):
        raise SystemExit(f"Source sheet is missing columns: {sorted(required - set(source[0]))}")
    ids = [row["rating_id"] for row in source]
    if len(ids) != len(set(ids)):
        raise SystemExit("Duplicate rating_id values in source sheet; refusing ambiguous sheets.")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = [out / "blind_rater_1.csv", out / "blind_rater_2.csv"]
    if not args.force and any(path.exists() for path in paths):
        raise SystemExit("A rater sheet already exists. Choose a new output folder or pass --force before ratings begin.")

    base_fields = list(source[0])
    rating_fields = [f"{side}_{dim}_1to5" for dim in DIMENSIONS for side in ("A", "B")]
    for old in ("A_usefulness_1to5", "B_usefulness_1to5"):
        if old not in base_fields:
            base_fields.append(old)
    fields = base_fields + [field for field in rating_fields if field not in base_fields]
    if "preference_A_B_tie" not in fields:
        fields.append("preference_A_B_tie")
    if "notes" not in fields:
        fields.append("notes")

    for index, path in enumerate(paths, 1):
        rows = []
        for row in source:
            item = {field: row.get(field, "") for field in fields}
            for field in rating_fields + ["preference_A_B_tie", "notes"]:
                item[field] = ""
            rows.append(item)
        random.Random(args.seed + index).shuffle(rows)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    print(f"Created two independent blind sheets ({len(source)} pairs each) in {out}")
    print("Share only blind_rater_1.csv or blind_rater_2.csv with each rater; keep annotation_key.csv private until both are locked.")


if __name__ == "__main__":
    main()
