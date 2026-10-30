
# sim/base.py
"""Simulated learner for HumanFirst-Route.

Families:
    A: Assistance can affect later unaided transfer.
    B: Full assistance does not directly damage transfer.
    C: Practice and forgetting change learner knowledge.

The RL policy must never receive hidden mastery or dependence.
"""

import random


PROFILES = {
    "novice": {"mastery": 0.25, "dependence": 0.25},
    "intermediate": {"mastery": 0.50, "dependence": 0.30},
    "advanced": {"mastery": 0.80, "dependence": 0.10},
    "hint-dependent": {"mastery": 0.45, "dependence": 0.85},
    "help-avoidant": {"mastery": 0.50, "dependence": 0.10},
    "frustration-prone": {"mastery": 0.40, "dependence": 0.35},
    "answer-copying": {"mastery": 0.45, "dependence": 0.90},
    "strong-independent": {"mastery": 0.85, "dependence": 0.05},
}

VALID_OPERATIONS = {
    "understand_plan",
    "execute",
    "debug_verify",
    "explain",
}

OPERATION_LEVEL_CAPS = {
    "understand_plan": 2,
    "execute": 4,
    "debug_verify": 3,
    "explain": 2,
}


class Learner:
    """A configurable learner that exposes observable outcomes only."""

    def __init__(self, family="A"):
        if family not in {"A", "B", "C"}:
            raise ValueError("family must be A, B, or C")

        self.family = family
        self.reset("intermediate", seed=42)

    def reset(self, profile, seed) -> None:
        """Start a new learner session with a reproducible random seed."""
        if profile not in PROFILES:
            raise ValueError(
                f"Unknown profile: {profile!r}. "
                f"Valid profiles: {list(PROFILES)}"
            )

        self.rng = random.Random(seed)
        self.profile = profile

        # Private simulator variables: never return these to the RL policy.
        self._mastery = PROFILES[profile]["mastery"]
        self._dependence = PROFILES[profile]["dependence"]
        self._transfer_damage = 0.0
        self._attempts = 0
        self._last_attempt_time = None

    def attempt(self, task, op, level, hint) -> dict:
        """Simulate one task attempt and return observable outcomes."""
        if not isinstance(level, int) or isinstance(level, bool):
            raise ValueError("level must be an integer from 0 to 4")
        if not 0 <= level <= 4:
            raise ValueError("level must be between 0 and 4")
        if op not in VALID_OPERATIONS:
            raise ValueError(f"Unknown operation: {op!r}")

        # Enforce the project's operation-specific assistance caps.
        if level > OPERATION_LEVEL_CAPS[op]:
            raise ValueError(
                f"Level {level} exceeds the cap "
                f"{OPERATION_LEVEL_CAPS[op]} for {op}"
            )

        self._attempts += 1

        # Family C: approximate forgetting between task attempts.
        if self.family == "C" and self._attempts > 1:
            self._mastery *= 0.985

        difficulty = task.get("difficulty", 2)
        if difficulty not in (1, 2, 3):
            raise ValueError("task difficulty must be 1, 2, or 3")

        # A small, configurable behavioral model:
        # assistance can help, but dependence moderates its benefit.
        assistance_effect = (
            0.13 * level * (1.0 - 0.35 * self._dependence)
        )
        difficulty_penalty = 0.10 * (difficulty - 1)

        probability = (
            self._mastery + assistance_effect - difficulty_penalty
        )
        probability = max(0.03, min(0.97, probability))
        success = self.rng.random() < probability

        # Assistance generally reduces completion time.
        time_taken = (
            55.0
            - 7.0 * level
            + (12.0 * self._dependence if level == 0 else 0.0)
            + self.rng.uniform(-4.0, 4.0)
        )
        time_taken = max(2.0, time_taken)

        # Family A: assistance may reduce later transfer.
        if self.family == "A" and level >= 3:
            self._transfer_damage += 0.025 * (level - 2)
            if level == 4:
                self._transfer_damage += 0.025 * self._dependence

        # Successful practice can improve mastery.
        if success:
            if self.family == "C":
                self._mastery += 0.045 * (1.0 - self._mastery)
            elif level <= 2:
                self._mastery += 0.025 * (1.0 - self._mastery)

        self._mastery = max(0.01, min(0.99, self._mastery))

        # Frustration-prone learners are more likely to struggle
        # when an unaided or lightly assisted attempt fails.
        frustrated = (
            self.profile == "frustration-prone"
            and not success
            and level <= 1
        )

        abandoned = frustrated and self.rng.random() < 0.15

        self._last_attempt_time = time_taken

        return {
            "success": bool(success),
            "time": round(time_taken, 2),
            "frustrated": bool(frustrated),
            "abandoned": bool(abandoned),
            "copied": bool(level == 4 and success),
        }

    def transfer(self, task) -> dict:
        """Evaluate an unaided transfer task without giving a hint."""
        probability = self._mastery

        if self.family == "A":
            probability -= self._transfer_damage

        # Family B has no direct transfer-damage penalty.
        # Family C uses its current mastery after practice/forgetting.
        probability = max(0.03, min(0.97, probability))

        return {"success": self.rng.random() < probability}
