
import csv
from pathlib import Path

from sim.base import Learner, PROFILES

FAMILIES = ("A", "B", "C")
LEVELS = range(5)
TRIALS = 100
TASK = {"difficulty": 2}


def evaluate(family, level, trials=TRIALS):
    task_success = 0
    transfer_success = 0
    total_time = 0.0

    for profile in PROFILES:
        for seed in range(trials):
            learner = Learner(family=family)
            learner.reset(profile, seed=seed)

            result = learner.attempt(
                task=TASK,
                op="execute",
                level=level,
                hint=None,
            )
            transfer = learner.transfer(TASK)

            task_success += int(result["success"])
            transfer_success += int(transfer["success"])
            total_time += result["time"]

    count = len(PROFILES) * trials
    task_rate = task_success / count
    transfer_rate = transfer_success / count
    mean_time = total_time / count

    # Transparent baseline utility: reward task success and transfer,
    # while applying a small penalty to time.
    utility = task_rate + transfer_rate - 0.005 * mean_time

    return {
        "task_success_rate": task_rate,
        "transfer_success_rate": transfer_rate,
        "mean_time": mean_time,
        "utility": utility,
    }


def main():
    rows = []

    for heldout in FAMILIES:
        training = [f for f in FAMILIES if f != heldout]

        # Select the assistance level using training families only.
        train_scores = {}
        for level in LEVELS:
            scores = [
                evaluate(family, level)
                for family in training
            ]
            train_scores[level] = sum(
                score["utility"] for score in scores
            ) / len(scores)

        selected_level = max(train_scores, key=train_scores.get)

        # Evaluate the selected level on the unseen family.
        test = evaluate(heldout, selected_level)

        row = {
            "heldout_family": heldout,
            "training_families": "+".join(training),
            "selected_level": selected_level,
            **{
                key: round(value, 4)
                for key, value in test.items()
            },
        }
        rows.append(row)

        print(
            f"Held out {heldout} | trained on {training} | "
            f"selected level {selected_level} | "
            f"task={test['task_success_rate']:.3f} | "
            f"transfer={test['transfer_success_rate']:.3f} | "
            f"time={test['mean_time']:.1f}"
        )

    output = Path("results/heldout_eval.csv")
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nPASS: all {len(rows)} held-out-family evaluations completed")
    print(f"Saved results to: {output}")


if __name__ == "__main__":
    main()
