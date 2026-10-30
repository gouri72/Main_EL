
from sim.base import Learner


def run_trial(practice_count, seed):
    learner = Learner(family="C")
    learner.reset("novice", seed=seed)

    task = {"difficulty": 2}

    # Repeated practice at a moderate assistance level.
    results = []
    for _ in range(practice_count):
        result = learner.attempt(
            task=task,
            op="execute",
            level=2,
            hint=None,
        )
        results.append(result["success"])

    # Test performance without assistance.
    final_result = learner.attempt(
        task=task,
        op="execute",
        level=0,
        hint=None,
    )

    return {
        "practice_success": (
            sum(results) / len(results) if results else 0.0
        ),
        "final_success": int(final_result["success"]),
    }


def main():
    trials = 300
    practice_counts = (0, 5, 15)

    for count in practice_counts:
        outcomes = [
            run_trial(count, seed)
            for seed in range(trials)
        ]

        final_rate = sum(
            item["final_success"] for item in outcomes
        ) / trials

        print(
            f"Practice attempts={count:2d} | "
            f"unaided final success={final_rate:.3f}"
        )

    print("\nCheck the trend rather than assuming it must be monotonic.")
    print("This test uses simulated learners, not real students.")


if __name__ == "__main__":
    main()
