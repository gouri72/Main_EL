"""Person 2: probability boundaries and input validation."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest
from sim.base import Learner


TASK = {"task_id": "boundary_test", "difficulty": 1}


def make_learner():
    learner = Learner("B")
    learner.reset("intermediate", seed=42)
    return learner


def test_lower_probability_boundary():
    learner = make_learner()
    learner._mastery = -1.0
    learner.rng.random = lambda: 0.029

    result = learner.attempt(TASK, "execute", 0, None)
    assert result["success"]


def test_lower_boundary_rejects_above_three_percent():
    learner = make_learner()
    learner._mastery = -1.0
    learner.rng.random = lambda: 0.031

    result = learner.attempt(TASK, "execute", 0, None)
    assert not result["success"]


def test_upper_probability_boundary():
    learner = make_learner()
    learner._mastery = 2.0
    learner.rng.random = lambda: 0.969

    result = learner.attempt(TASK, "execute", 0, None)
    assert result["success"]


def test_upper_boundary_rejects_above_97_percent():
    learner = make_learner()
    learner._mastery = 2.0
    learner.rng.random = lambda: 0.971

    result = learner.attempt(TASK, "execute", 0, None)
    assert not result["success"]


@pytest.mark.parametrize("level", [-1, 5, True, 1.5, "2"])
def test_invalid_assistance_levels(level):
    learner = make_learner()
    with pytest.raises(ValueError):
        learner.attempt(TASK, "execute", level, None)


def test_invalid_operation_rejected():
    learner = make_learner()
    with pytest.raises(ValueError):
        learner.attempt(TASK, "invalid_operation", 0, None)


@pytest.mark.parametrize(
    "operation,level",
    [
        ("understand_plan", 3),
        ("debug_verify", 4),
        ("explain", 3),
    ],
)
def test_operation_caps_rejected(operation, level):
    learner = make_learner()
    with pytest.raises(ValueError):
        learner.attempt(TASK, operation, level, None)


@pytest.mark.parametrize("difficulty", [0, 4, -1, "2"])
def test_invalid_difficulty_rejected(difficulty):
    learner = make_learner()
    task = {"task_id": "invalid", "difficulty": difficulty}
    with pytest.raises(ValueError):
        learner.attempt(task, "execute", 0, None)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

def test_exact_lower_probability_boundary_fails():
    learner = make_learner()
    learner._mastery = -1.0
    learner.rng.random = lambda: 0.03

    result = learner.attempt(TASK, "execute", 0, None)
    assert not result["success"]


def test_exact_upper_probability_boundary_fails():
    learner = make_learner()
    learner._mastery = 2.0
    learner.rng.random = lambda: 0.97

    result = learner.attempt(TASK, "execute", 0, None)
    assert not result["success"]
