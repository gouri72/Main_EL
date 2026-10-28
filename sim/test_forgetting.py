
from sim.base import Learner


class AlwaysFailRNG:
    """Deterministic RNG for testing forgetting without successful practice."""

    def random(self):
        return 0.99

    def uniform(self, low, high):
        return 0.0


def main():
    learner = Learner(family="C")
    learner.reset("advanced", seed=42)

    # Internal test setup only; mastery is never returned to the policy.
    learner._mastery = 0.80
    learner.rng = AlwaysFailRNG()

    task = {"difficulty": 3}

    # First attempt: no forgetting is applied before attempt 1.
    first = learner.attempt(task, op="execute", level=0, hint=None)
    assert first["success"] is False
    assert abs(learner._mastery - 0.80) < 1e-9

    # Second attempt: forgetting should apply before the attempt.
    second = learner.attempt(task, op="execute", level=0, hint=None)
    assert second["success"] is False
    expected = 0.80 * 0.985
    assert abs(learner._mastery - expected) < 1e-9

    # More unsuccessful attempts should continue to decay mastery.
    for _ in range(8):
        result = learner.attempt(
            task, op="execute", level=0, hint=None
        )
        assert result["success"] is False

    assert learner._mastery < expected

    print("PASS: forgetting starts after the first attempt")
    print("PASS: mastery decays by 1.5% per subsequent attempt")
    print("PASS: unsuccessful practice does not add learning")
    print("Note: mastery checks are internal simulator tests only.")


if __name__ == "__main__":
    main()
