#!/usr/bin/env python3
"""Summarize completed blind ratings and, when available, inter-rater agreement."""
import argparse
import csv
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def read_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def get_score(row, column, rating_id):
    raw = (row.get(column) or "").strip()
    if not raw:
        raise SystemExit(f"Human ratings are incomplete: rating_id={rating_id} has blank `{column}`.")
    try:
        value = int(raw)
    except ValueError as exc:
        raise SystemExit(f"Invalid score `{raw}` in {column} for {rating_id}; enter an integer from 1 to 5.") from exc
    if not 1 <= value <= 5:
        raise SystemExit(f"Rating values must be 1..5 in {column} for {rating_id}.")
    return value


def bootstrap_delta(records, replicates, seed):
    groups = defaultdict(list)
    for record in records:
        groups[record["group_id"]].append(record["delta"])
    observed = statistics.mean(value for group in groups.values() for value in group)
    values = list(groups.values())
    rng = random.Random(seed)
    estimates = []
    for _ in range(replicates):
        sample = [rng.choice(values) for _ in values]
        estimates.append(statistics.mean(value for group in sample for value in group))
    estimates.sort()
    return observed, [estimates[int(.025 * replicates)], estimates[min(replicates - 1, int(.975 * replicates))]]


def kappa(left, right, quadratic=False):
    if not left or len(left) != len(right):
        return {"agreement": None, "kappa": None}
    categories = sorted(set(left) | set(right))
    n = len(left)
    observed_agreement = sum(a == b for a, b in zip(left, right)) / n
    if len(categories) == 1:
        return {"agreement": observed_agreement, "kappa": None, "note": "No rating variation; kappa is undefined."}
    left_marginal = Counter(left)
    right_marginal = Counter(right)
    observed_disagreement = expected_disagreement = 0.0
    scale = max(categories) - min(categories) if quadratic else 1
    for a in categories:
        for b in categories:
            distance = ((a - b) / scale) ** 2 if quadratic else float(a != b)
            observed_disagreement += (sum(x == a and y == b for x, y in zip(left, right)) / n) * distance
            expected_disagreement += (left_marginal[a] / n) * (right_marginal[b] / n) * distance
    value = None if expected_disagreement == 0 else 1 - observed_disagreement / expected_disagreement
    return {"agreement": observed_agreement, "kappa": value,
            "weighting": "quadratic" if quadratic else "unweighted"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sheet", help="single completed blind_annotation_sheet.csv (legacy 3-dimension mode)")
    group.add_argument("--rater1", help="completed blind_rater_1.csv")
    parser.add_argument("--rater2", help="completed blind_rater_2.csv; required with --rater1")
    parser.add_argument("--key", required=True, help="private annotation_key.csv; do not share with raters")
    parser.add_argument("--output", default="human_rating_summary.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    args = parser.parse_args()
    if args.bootstrap_replicates < 1:
        parser.error("--bootstrap-replicates must be positive")
    if bool(args.rater1) != bool(args.rater2):
        parser.error("provide both --rater1 and --rater2")
    key_rows = read_rows(args.key)
    key = {row["rating_id"]: row for row in key_rows}
    if len(key) != len(key_rows):
        raise SystemExit("Duplicate rating_id values in the annotation key.")
    dimensions = ("relevance", "correctness", "level_fit") if args.sheet else (
        "relevance", "correctness", "level_fit", "usefulness")
    first = read_rows(args.sheet or args.rater1)
    second = read_rows(args.rater2) if args.rater2 else None
    first_by_id = {row["rating_id"]: row for row in first}
    second_by_id = {row["rating_id"]: row for row in second} if second is not None else None
    if len(first_by_id) != len(first) or (second is not None and len(second_by_id) != len(second)):
        raise SystemExit("Duplicate rating_id in a rater sheet.")
    if second_by_id is not None and set(first_by_id) != set(second_by_id):
        raise SystemExit("Rater sheets must contain the same rating_id set for paired agreement.")
    if set(first_by_id) != set(key):
        raise SystemExit("Rater sheet and annotation key have different rating_id sets.")

    prepared = []
    for rating_id in sorted(first_by_id):
        mapping = key[rating_id]
        r1 = first_by_id[rating_id]
        r2 = second_by_id[rating_id] if second_by_id is not None else None
        if r2 and (r1.get("output_A") != r2.get("output_A") or r1.get("output_B") != r2.get("output_B")):
            raise SystemExit(f"Rater sheets do not show identical A/B outputs for {rating_id}.")
        scores = {}
        for rater, row in (("r1", r1), ("r2", r2)):
            if row is None:
                continue
            for dim in dimensions:
                for side in ("A", "B"):
                    scores[f"{rater}_{side}_{dim}"] = get_score(row, f"{side}_{dim}_1to5", rating_id)
            pref = (row.get("preference_A_B_tie") or "").strip().upper()
            if pref not in {"A", "B", "TIE"}:
                raise SystemExit(f"Preference must be A, B, or tie for {rating_id} ({rater}).")
            scores[f"{rater}_preference"] = pref
        scores.update({"rating_id": rating_id, "group_id": mapping["group_id"],
                       "A_is": mapping["A_is"], "B_is": mapping["B_is"]})
        prepared.append(scores)

    summary = {"n_rated_pairs": len(prepared), "n_task_groups": len({row["group_id"] for row in prepared}),
               "rating_source": "scores entered in the supplied annotation forms; rater identity is not independently authenticated",
               "n_raters": 2 if second_by_id is not None else 1,
               "scale": "1=poor, 3=mixed, 5=strong",
               "dimensions": {},
               "limitation": "Ratings describe these raters and sampled outputs; they do not measure learner outcomes or population effects."}
    for dim in dimensions:
        base_scores, lora_scores, deltas, agreement_left, agreement_right = [], [], [], [], []
        for row in prepared:
            rater_deltas = []
            for rater in (("r1", "r2") if second_by_id is not None else ("r1",)):
                a, b = row[f"{rater}_A_{dim}"], row[f"{rater}_B_{dim}"]
                base = a if row["A_is"] == "base" else b
                lora = a if row["A_is"] == "lora" else b
                base_scores.append(base); lora_scores.append(lora); rater_deltas.append(lora - base)
            deltas.append({"group_id": row["group_id"], "delta": statistics.mean(rater_deltas)})
            if second_by_id is not None:
                agreement_left.extend([row[f"r1_A_{dim}"], row[f"r1_B_{dim}"]])
                agreement_right.extend([row[f"r2_A_{dim}"], row[f"r2_B_{dim}"]])
        delta, ci = bootstrap_delta(deltas, args.bootstrap_replicates, args.seed)
        result = {"base_mean": statistics.mean(base_scores), "lora_mean": statistics.mean(lora_scores),
                  "paired_delta_lora_minus_base": delta, "task_group_bootstrap_95_ci": ci}
        if second_by_id is not None:
            result["inter_rater"] = kappa(agreement_left, agreement_right, quadratic=True)
        summary["dimensions"][dim] = result

    def preference_label(row, rater):
        pref = row[f"{rater}_preference"]
        if pref == "TIE": return "tie"
        side = pref
        return row[f"{side}_is"]

    pref_counts = Counter()
    pref_agreement_a, pref_agreement_b = [], []
    for row in prepared:
        for rater in (("r1", "r2") if second_by_id is not None else ("r1",)):
            pref_counts[preference_label(row, rater)] += 1
        if second_by_id is not None:
            pref_agreement_a.append(preference_label(row, "r1"))
            pref_agreement_b.append(preference_label(row, "r2"))
    summary["preference_counts"] = dict(pref_counts)
    if second_by_id is not None:
        summary["preference_inter_rater"] = kappa(pref_agreement_a, pref_agreement_b)
    Path(args.output).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
