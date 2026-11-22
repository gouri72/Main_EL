"""Person 2: observation privacy and state encoding tests."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rl.env import HumanFirstEnv, FAKE_TASKS
from rl.configured_env import ConfiguredEnv


def test_observation_has_eight_features():
    env = HumanFirstEnv(FAKE_TASKS, seed=42)
    obs = env.reset()

    assert len(obs) == 8
    assert isinstance(obs, list)


def test_observation_does_not_directly_expose_private_learner_state():
    env = HumanFirstEnv(FAKE_TASKS, seed=42)
    env.reset()

    obs_before = env._get_obs()

    # Change private simulator values without changing observed history.
    env.learner._mastery = 0.99
    env.learner._dependence = 0.99
    env.learner._transfer_damage = 0.99

    obs_after = env._get_obs()

    assert obs_before == obs_after


def test_observation_contains_expected_feature_structure():
    env = HumanFirstEnv(FAKE_TASKS, seed=42)
    obs = env.reset()

    assert len(obs) == 8
    assert obs[0] == 0  # coding domain
    assert obs[1] == 0  # understand_plan operation
    assert obs[4] == 1  # task difficulty


def test_configured_env_blinds_operation_features():
    env = ConfiguredEnv(
        FAKE_TASKS,
        reward={"ablate": ["operation_state"]},
        seed=42,
    )
    obs = env.reset()

    assert obs[1] == 0
    assert obs[3] == 0


def test_configured_env_drops_selected_features():
    env = ConfiguredEnv(FAKE_TASKS, drop=[2, 4], seed=42)
    obs = env.reset()

    assert obs[2] == 0
    assert obs[4] == 0
    assert len(obs) == 8


def test_configured_env_observation_has_eight_features():
    env = ConfiguredEnv(FAKE_TASKS, seed=42)
    obs = env.reset()

    assert len(obs) == 8


def test_estimator_uses_observed_history_only():
    env = HumanFirstEnv(FAKE_TASKS, seed=42)
    env.reset()

    estimates_before = env.estimator.estimate("coding", "understand_plan")

    env.learner._mastery = 0.99
    env.learner._dependence = 0.99
    env.learner._transfer_damage = 0.99

    estimates_after = env.estimator.estimate("coding", "understand_plan")

    assert estimates_before == estimates_after


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))

def test_configured_env_drops_all_features():
    env = ConfiguredEnv(FAKE_TASKS, drop=list(range(8)), seed=42)
    obs = env.reset()

    assert len(obs) == 8
    assert obs == [0] * 8
