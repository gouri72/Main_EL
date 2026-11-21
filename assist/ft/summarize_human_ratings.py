#!/usr/bin/env python3
"""Summarize completed blind ratings; refuses blank or invalid human score fields."""
import argparse
import csv
import json
import random
import statistics
from collections import defaultdict, Counter
from pathlib import Path


def rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sheet", required=True, help="completed blind_annotation_sheet.csv copy")
    ap.add_argument("--key", required=True, help="matching annotation_key.csv")
    ap.add_argument("--output", default="human_rating_summary.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bootstrap-replicates", type=int, default=5000)
    args = ap.parse_args()
    sheet = rows(args.sheet); key = {r["rating_id"]: r for r in rows(args.key)}
    dimensions = ("relevance", "correctness", "level_fit")
    paired = []
    for row in sheet:
        mapping = key.get(row["rating_id"])
        if mapping is None: raise SystemExit(f"No blind key entry for rating_id={row['rating_id']}")
        scores = {}
        for dim in dimensions:
            scores["A_" + dim] = int(row["A_" + dim + "_1to5"])
            scores["B_" + dim] = int(row["B_" + dim + "_1to5"])
            if not (1 <= scores["A_" + dim] <= 5 and 1 <= scores["B_" + dim] <= 5):
                raise SystemExit(f"Rating values must be 1..5 for {row['rating_id']}")
        pref = row["preference_A_B_tie"].strip().upper()
        if pref not in {"A", "B", "TIE"}:
            raise SystemExit(f"Preference must be A, B, or tie for {row['rating_id']}")
        scores.update({"rating_id": row["rating_id"], "group_id": mapping["group_id"],
                       "domain": row["domain"], "level": str(row["level"]),
                       "A_is": mapping["A_is"], "B_is": mapping["B_is"],
                       "preference": pref})
        paired.append(scores)
    if not paired: raise SystemExit("No completed ratings found")

    rng = random.Random(args.seed)
    groups = defaultdict(list)
    for r in paired: groups[r["group_id"]].append(r)

    def mean_ci(differences):
        observed = statistics.mean(differences)
        clusters = defaultdict(list)
        for r, diff in zip(paired, differences): clusters[r["group_id"]].append(diff)
        vals = list(clusters.values()); boot = []
        for _ in range(args.bootstrap_replicates):
            sample = [rng.choice(vals) for _ in vals]
            boot.append(statistics.mean(x for group in sample for x in group))
        boot.sort()
        return observed, [boot[int(.025 * len(boot))], boot[min(len(boot)-1, int(.975 * len(boot)))]]

    summary = {"n_rated_pairs": len(paired), "n_task_groups": len(groups), "scale": "1=poor, 3=mixed, 5=strong",
               "ratings_are_human": True, "dimensions": {}}
    for dim in dimensions:
        base_scores, lora_scores, diffs = [], [], []
        for r in paired:
            a, b = r["A_" + dim], r["B_" + dim]
            base = a if r["A_is"] == "base" else b
            lora = a if r["A_is"] == "lora" else b
            base_scores.append(base); lora_scores.append(lora); diffs.append(lora - base)
        delta, ci = mean_ci(diffs)
        summary["dimensions"][dim] = {"base_mean": statistics.mean(base_scores),
            "lora_mean": statistics.mean(lora_scores), "paired_delta_lora_minus_base": delta,
            "task_group_bootstrap_95_ci": ci}
    pref_counts = Counter()
    for r in paired:
        if r["preference"] == "TIE": pref_counts["tie"] += 1
        else: pref_counts["lora" if r["A_is" if r["preference"] == "A" else "B_is"] == "lora" else "base"] += 1
    summary["preference_counts"] = dict(pref_counts)
    summary["limitation"] = "Ratings represent this rater/sample only. They are not evidence of learning gains or population-level human effects."
    Path(args.output).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
