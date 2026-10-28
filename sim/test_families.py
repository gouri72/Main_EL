
from sim.base import Learner

def measure_transfer(family, assisted):
    successes = 0
    trials = 200

    for seed in range(trials):
        learner = Learner(family=family)
        learner.reset("advanced", seed=seed)

        if assisted:
            for _ in range(8):
                learner.attempt(
                    task={"difficulty": 1},
                    op="execute",
                    level=4,
                    hint=None,
                )

        result = learner.transfer({"difficulty": 1})
        successes += int(result["success"])

    return successes / trials


def main():
    a_assisted = measure_transfer("A", assisted=True)
    b_assisted = measure_transfer("B", assisted=True)
    a_unassisted = measure_transfer("A", assisted=False)

    print(f"Family A, heavy assistance: {a_assisted:.3f}")
    print(f"Family B, heavy assistance: {b_assisted:.3f}")
    print(f"Family A, no assistance:    {a_unassisted:.3f}")

    assert a_assisted < b_assisted, (
        "Expected heavy assistance to reduce transfer more in Family A"
    )
    assert abs(a_unassisted - b_assisted) < 0.15, (
        "Baseline transfer rates differ more than expected"
    )

    print("PASS: Family A and B transfer behavior differs as intended")


if __name__ == "__main__":
    main()
