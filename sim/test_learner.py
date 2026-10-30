
from sim.base import Learner, PROFILES

def main():
    families = ("A", "B", "C")
    task = {"difficulty": 2}
    tested = 0

    for family in families:
        for profile in PROFILES:
            learner = Learner(family=family)
            learner.reset(profile, seed=42)

            result = learner.attempt(
                task=task,
                op="execute",
                level=2,
                hint=None,
            )

            required = {
                "success",
                "time",
                "frustrated",
                "abandoned",
                "copied",
            }
            assert required.issubset(result), (
                f"Missing result fields: {family}, {profile}"
            )
            assert isinstance(result["success"], bool)
            assert result["time"] > 0

            transfer = learner.transfer(task)
            assert isinstance(transfer, dict)
            assert isinstance(transfer["success"], bool)

            tested += 1

    assert tested == 24
    print("PASS: all 3 simulator families")
    print("PASS: all 8 learner profiles")
    print("PASS: attempt and transfer interfaces")
    print(f"PASS: {tested} family-profile combinations")

if __name__ == "__main__":
    main()
