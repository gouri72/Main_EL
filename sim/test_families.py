from sim.base import Learner


TRIALS = 500
PROFILE = "advanced"
TASK = {"difficulty": 1}
ASSISTED_ATTEMPTS = 8
ASSISTANCE_LEVEL = 4


def measure_transfer(family, assisted, trials=TRIALS):
    successes = 0

    for seed in range(trials):
        learner = Learner(family=family)
        learner.reset(PROFILE, seed=seed)

        if assisted:
            for _ in range(ASSISTED_ATTEMPTS):
                learner.attempt(
                    task=TASK,
                    op="execute",
                    level=ASSISTANCE_LEVEL,
                    hint=None,
                )

        result = learner.transfer(TASK)
        successes += int(result["success"])

    return successes / trials


def main():
    results = {}

    for family in ("A", "B"):
        for assisted in (False, True):
            rate = measure_transfer(family, assisted)
            condition = "heavy_assistance" if assisted else "no_assistance"
            results[(family, condition)] = rate

            print(
                f"Family {family} | {condition:16s} | "
                f"transfer={rate:.3f}"
            )

    a_assisted = results[("A", "heavy_assistance")]
    b_assisted = results[("B", "heavy_assistance")]
    a_baseline = results[("A", "no_assistance")]
    b_baseline = results[("B", "no_assistance")]

    print("\nMatched comparisons:")
    print(f"A heavy vs B heavy: {a_assisted:.3f} vs {b_assisted:.3f}")
    print(f"A baseline vs B baseline: {a_baseline:.3f} vs {b_baseline:.3f}")

    assert a_assisted < b_assisted, (
        "Expected Family A transfer to be lower than Family B "
        "under heavy assistance"
    )

    assert abs(a_baseline - b_baseline) < 0.10, (
        "Family A and B baseline transfer should be similar "
        "without assistance"
    )

    assert a_assisted < a_baseline, (
        "Expected Family A heavy assistance to reduce transfer "
        "relative to its own baseline"
    )

    print("\nPASS: matched Family A/B mechanism checks")


if __name__ == "__main__":
    main()
