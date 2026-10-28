
import csv
from pathlib import Path

from sim.base import Learner, PROFILES

FAMILIES = ("A", "B", "C")
TRIALS = 100

# Assistance decreases as the learner progresses.
SCHEDULES = {
    "fading": [4, 4, 3, 3, 2, 2, 1, 0],
    "fixed_full": [4] * 8,
    "fixed_moderate": [2] * 8,
}


def run_session(family, profile, levels, seed):
    learner = Learner(family=family)
    learner.reset(profile, seed=seed)

    successes = 0
    total_time = 0.0
    copied = 0

    for index, level in enumerate(levels):
        task = {"id": f"session_task_{index + 1}", "difficulty": 2}
        result = learner.attempt(
            task=task,
            op="execute",
            level=level,
            hint=None,
        )
        successes += int(result["success"])
        total_time += result["time"]
        copied += int(result["copied"])

    transfer = learner.transfer({"difficulty": 2})

    return {
        "session_success_rate": successes / len(levels),
        "mean_time_per_task": total_time / len(levels),
        "copied_rate": copied / len(levels),
        "final_transfer_success": int(transfer["success"]),
    }


def main():
    rows = []

    for family in FAMILIES:
        for profile in PROFILES:
            for schedule_name, levels in SCHEDULES.items():
                outcomes = [
                    run_session(family, profile, levels, seed)
                    for seed in range(TRIALS)
                ]

                count = len(outcomes)
                row = {
                    "family": family,
                    "profile": profile,
                    "schedule": schedule_name,
                    "levels": "-".join(map(str, levels)),
                    "session_success_rate": round(
                        sum(x["session_success_rate"] for x in outcomes)
                        / count, 4
                    ),
                    "mean_time_per_task": round(
                        sum(x["mean_time_per_task"] for x in outcomes)
                        / count, 2
                    ),
                    "copied_rate": round(
                        sum(x["copied_rate"] for x in outcomes) / count, 4
                    ),
                    "final_transfer_rate": round(
                        sum(x["final_transfer_success"] for x in outcomes)
                        / count, 4
                    ),
                    "trials": TRIALS,
                }
                rows.append(row)

    output = Path("results/fading_analysis.csv")
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"PASS: {len(rows)} family-profile-schedule conditions")
    print(f"Saved results to: {output}")

    print("\nOverall averages by schedule:")
    for name in SCHEDULES:
        group = [row for row in rows if row["schedule"] == name]
        print(
            f"{name}: "
            f"session success="
            f"{sum(r['session_success_rate'] for r in group) / len(group):.3f}, "
            f"transfer="
            f"{sum(r['final_transfer_rate'] for r in group) / len(group):.3f}, "
            f"time="
            f"{sum(r['mean_time_per_task'] for r in group) / len(group):.1f}, "
            f"copied="
            f"{sum(r['copied_rate'] for r in group) / len(group):.3f}"
        )


if __name__ == "__main__":
    main()
