#!/usr/bin/env python3
"""Prepare blinded semantic error-review forms from actual paired evaluator output."""
import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path


CATEGORIES = ("both_pass", "base_fails_lora_passes", "lora_fails_base_passes", "both_fail")


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).open(encoding="utf-8") if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True, help="paired_error_cases.csv from the completed run")
    parser.add_argument("--test-file", required=True, help="prepared test.jsonl used by that run")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--per-category", type=int, default=5, help="seeded sample size per automatic pass/fail category")
    parser.add_argument("--seed", type=int, default=20261124)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.per_category < 1:
        parser.error("--per-category must be positive")
    cases = read_csv(args.cases)
    prompts = {row["example_id"]: row for row in read_jsonl(args.test_file)}
    by_category = defaultdict(list)
    for case in cases:
        category = case["paired_case"]
        if category not in CATEGORIES:
            raise SystemExit(f"Unexpected pair category: {category}")
        if case["example_id"] not in prompts:
            raise SystemExit(f"No test prompt found for example_id={case['example_id']}")
        by_category[category].append(case)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    outputs = [out / "error_review_rater_1.csv", out / "error_review_rater_2.csv", out / "error_review_key.csv"]
    if not args.force and any(path.exists() for path in outputs):
        raise SystemExit("An error-review output already exists; choose a new folder or use --force before annotation.")
    rng = random.Random(args.seed)
    selected = []
    for category in CATEGORIES:
        group = list(by_category[category])
        rng.shuffle(group)
        selected.extend(group[:args.per_category])
    rng.shuffle(selected)
    sheets, key_rows = [[], []], []
    for case in selected:
        prompt = prompts[case["example_id"]]
        flip = rng.choice([False, True])
        a_model, b_model = ("lora", "base") if flip else ("base", "lora")
        case_id = f"c{len(key_rows)+1:03d}"
        key_rows.append({"case_id": case_id, "example_id": case["example_id"], "group_id": case["group_id"],
                         "automatic_category": case["paired_case"], "A_model": a_model, "B_model": b_model,
                         "A_validator_reasons": case[f"{a_model}_failure_categories"],
                         "B_validator_reasons": case[f"{b_model}_failure_categories"]})
        for rater_index, sheet in enumerate(sheets):
            row = {"case_id": case_id, "domain": case["domain"], "operation": case["op"], "level": case["level"],
                   "task_context": prompt["user"], "output_A": case[f"{a_model}_output"],
                   "output_B": case[f"{b_model}_output"], "A_error_types": "", "A_explanation": "",
                   "A_confidence_1to3": "", "B_error_types": "", "B_explanation": "", "B_confidence_1to3": "",
                   "pair_preference_A_B_tie": "", "notes": ""}
            sheet.append(row)
    fields = list(sheets[0][0]) if selected else ["case_id", "domain", "operation", "level", "task_context", "output_A", "output_B",
        "A_error_types", "A_explanation", "A_confidence_1to3", "B_error_types", "B_explanation", "B_confidence_1to3", "pair_preference_A_B_tie", "notes"]
    for index, path in enumerate(outputs[:2]):
        random.Random(args.seed + index + 1).shuffle(sheets[index])
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(sheets[index])
    with outputs[2].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(key_rows[0]) if key_rows else ["case_id"])
        writer.writeheader(); writer.writerows(key_rows)
    counts = {category: min(args.per_category, len(by_category[category])) for category in CATEGORIES}
    print(json.dumps({"review_pairs": len(selected), "selected_per_automatic_category": counts,
                      "sheets": [str(path) for path in outputs[:2]],
                      "private_key": str(outputs[2]),
                      "review_guide": "Tag each output with answer_leakage, incorrect_guidance, operation_mismatch, assistance_level_inappropriate, no_identified_error, or other; explain the judgment. Do not share the key until both reviewers finish."}, indent=2))


if __name__ == "__main__":
    main()
