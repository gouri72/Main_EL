import csv
from pathlib import Path

from sim.base import Learner


FAMILIES = ("A", "B", "C")
PROFILES = ("novice", "advanced", "hint-dependent", "frustration-prone")
LEVELS = (0, 2, 4)
SEEDS = range(100)
ATTEMPTS = 8
TASK = {"difficulty": 2}
OP = "execute"

OUTPUT = Path("results/person2_robustness.csv")


def run_trajectory(family, profile, level, seed):
    learner = Learner(family=family)
    learner.reset(profile, seed=seed)

    successes = 0
    total_time = 0.0
    frustration_count = 0
    abandonment_count = 0
    copying_count = 0

    for _ in range(ATTEMPTS):
        result = learner.attempt(
            task=TASK,
            op=OP,
            level=level,
            hint=None,
        )

        successes += int(result["success"])
        total_time += result["time"]
        frustration_count += int(result["frustrated"])
        abandonment_count += int(result["abandoned"])
        copying_count += int(result["copied"])

    transfer_result = learner.transfer(TASK)

    return {
        "family": family,
        "profile": profile,
        "level": level,
        "seed": seed,
        "attempts": ATTEMPTS,
        "practice_success_rate": successes / ATTEMPTS,
        "mean_time": total_time / ATTEMPTS,
        "frustration_count": frustration_count,
        "abandonment_count": abandonment_count,
        "copying_count": copying_count,
        "transfer_success": int(transfer_result["success"]),
    }


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    total_conditions = len(FAMILIES) * len(PROFILES) * len(LEVELS) * len(SEEDS)
    completed = 0

    for family in FAMILIES:
        for profile in PROFILES:
            for level in LEVELS:
                for seed in SEEDS:
                    rows.append(
                        run_trajectory(family, profile, level, seed)
                    )
                    completed += 1

        print(f"Completed family {family}: {completed}/{total_conditions}")

    with OUTPUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} trajectories to {OUTPUT}")


if __name__ == "__main__":
    main()
