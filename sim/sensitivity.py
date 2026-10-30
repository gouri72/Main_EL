
import csv
from pathlib import Path

from sim.base import Learner, PROFILES

FAMILIES = ("A", "B", "C")
LEVELS = range(5)
TRIALS = 100
TASK = {"difficulty": 2}


def run_sweep():
    rows = []

    for family in FAMILIES:
        for profile in PROFILES:
            for level in LEVELS:
                successes = 0
                total_time = 0.0
                transfer_successes = 0
                copied_count = 0

                for seed in range(TRIALS):
                    learner = Learner(family=family)
                    learner.reset(profile, seed=seed)

                    result = learner.attempt(
                        task=TASK,
                        op="execute",
                        level=level,
                        hint=None,
                    )
                    transfer = learner.transfer(TASK)

                    successes += int(result["success"])
                    total_time += result["time"]
                    transfer_successes += int(transfer["success"])
                    copied_count += int(result["copied"])

                rows.append({
                    "family": family,
                    "profile": profile,
                    "level": level,
                    "task_success_rate": round(successes / TRIALS, 4),
                    "mean_time": round(total_time / TRIALS, 2),
                    "transfer_success_rate": round(
                        transfer_successes / TRIALS, 4
                    ),
                    "copied_rate": round(copied_count / TRIALS, 4),
                    "trials": TRIALS,
                })

    output = Path("results/sensitivity.csv")
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"PASS: sensitivity sweep completed ({len(rows)} conditions)")
    print(f"Saved results to: {output}")

    print("\nAverage outcomes by family and assistance level:")
    for family in FAMILIES:
        for level in LEVELS:
            group = [
                row for row in rows
                if row["family"] == family and row["level"] == level
            ]
            avg_success = sum(
                row["task_success_rate"] for row in group
            ) / len(group)
            avg_transfer = sum(
                row["transfer_success_rate"] for row in group
            ) / len(group)
            avg_time = sum(
                row["mean_time"] for row in group
            ) / len(group)

            print(
                f"Family {family}, level {level}: "
                f"task={avg_success:.3f}, "
                f"transfer={avg_transfer:.3f}, "
                f"time={avg_time:.1f}"
            )


if __name__ == "__main__":
    run_sweep()
