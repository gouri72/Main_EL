"""Person 2: reproducibility and reset tests."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rl.env import HumanFirstEnv, FAKE_TASKS


def test_same_seed_produces_same_episode():
    env1 = HumanFirstEnv(FAKE_TASKS, profile="intermediate", family="B", seed=123)
    env2 = HumanFirstEnv(FAKE_TASKS, profile="intermediate", family="B", seed=123)

    obs1 = env1.reset()
    obs2 = env2.reset()

    assert obs1 == obs2

    result1 = env1.step(0)
    result2 = env2.step(0)

    assert result1 == result2


def test_reset_restores_episode_counters():
    env = HumanFirstEnv(FAKE_TASKS, seed=123)
    env.reset()

    env.step(0)

    assert env.current_task_idx > 0
    assert env.attempt_count > 0

    env.reset()

    assert env.current_task_idx == 0
    assert env.attempt_count == 0
    assert env.learner_attempt_count == 0
    assert env.help_seeking_count == 0
    assert env.frustration_count == 0
    assert env.time_total == 0.0
    assert env.copy_count == 0
    assert env.abandoned_count == 0


def test_reset_clears_estimator_history():
    env = HumanFirstEnv(FAKE_TASKS, seed=123)
    env.reset()

    env.estimator.update("coding", "execute", True, 0)

    assert env.estimator.domain_stats["coding"]["attempts"] > 0

    env.reset()

    assert env.estimator.domain_stats["coding"]["attempts"] == 0
    assert env.estimator.operation_stats[("coding", "execute")]["attempts"] == 0


def test_reset_reproduces_same_first_step():
    env = HumanFirstEnv(FAKE_TASKS, profile="novice", family="A", seed=777)

    env.reset()
    first_result = env.step(0)

    env.reset()
    second_result = env.step(0)

    assert first_result == second_result


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
