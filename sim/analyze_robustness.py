import csv
import math
from collections import defaultdict
from pathlib import Path

INPUT = Path("results/person2_robustness.csv")
OUTPUT = Path("results/person2_robustness_summary.csv")

METRICS = [
    "practice_success_rate",
    "mean_time",
    "frustration_count",
    "abandonment_count",
    "copying_count",
    "transfer_success",
]


def mean(values):
    return sum(values) / len(values)


def ci95(values):
    if len(values) < 2:
        return 0.0
    se = math.sqrt(sum((x - mean(values)) ** 2 for x in values) / (len(values) - 1)) / math.sqrt(len(values))
    return 1.96 * se


def main():
    groups = defaultdict(list)

    with INPUT.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["family"], row["profile"], int(row["level"]))
            groups[key].append(row)

    summary = []

    for (family, profile, level), rows in sorted(groups.items()):
        record = {
            "family": family,
            "profile": profile,
            "level": level,
            "n": len(rows),
        }

        for metric in METRICS:
            values = [float(r[metric]) for r in rows]
            record[f"{metric}_mean"] = mean(values)
            record[f"{metric}_ci95"] = ci95(values)

        summary.append(record)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)

    print(f"Summary groups: {len(summary)}")
    print(f"Saved: {OUTPUT}")

    print("\nTransfer performance by family and assistance:")
    for family in ("A", "B", "C"):
        for level in (0, 2, 4):
            selected = [
                r for r in summary
                if r["family"] == family and r["level"] == level
            ]
            transfer = mean([r["transfer_success_mean"] for r in selected])
            print(f"Family {family} | Level {level} | transfer={transfer:.3f}")


if __name__ == "__main__":
    main()
