"""Person 2: HumanFirstEnv route and assistance validation tests."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rl.env import HumanFirstEnv
from sim.base import OPERATION_LEVEL_CAPS
from data.loader import load_tasks, session_pool


TASK = session_pool(load_tasks())[0]


def make_task(operation="understand_plan"):
    task = dict(TASK)
    task["operations"] = [{"op": operation}]
    return task


def scripted_result(success, level=0):
    return {
        "success": success,
        "time": 1.0 + level,
        "frustrated": False,
        "abandoned": False,
        "copied": False,
    }


def make_env(operation="understand_plan"):
    env = HumanFirstEnv(
        tasks=[make_task(operation)],
        profile="advanced",
        family="B",
        seed=42,
    )
    env.reset()
    return env


def test_human_first_independent_success(monkeypatch):
    env = make_env()
    calls = []

    def fake_attempt(task, operation, level, hint=None):
        calls.append(level)
        return scripted_result(True, level)

    monkeypatch.setattr(env.learner, "attempt", fake_attempt)

    _, _, done, info = env.step(4)

    assert calls == [0]
    assert done
    assert info["route_name"] == "human_first"
    assert info["independent_success"]
    assert not info["assistance_used"]
    assert info["actual_level"] == 0
    assert info["attempts_made"] == 1


def test_human_first_failure_then_assistance(monkeypatch):
    env = make_env()
    calls = []

    def fake_attempt(task, operation, level, hint=None):
        calls.append(level)
        return scripted_result(len(calls) == 2, level)

    monkeypatch.setattr(env.learner, "attempt", fake_attempt)

    _, _, done, info = env.step(4)

    assert calls == [0, 2]  # understand_plan cap is 2
    assert done
    assert info["route_name"] == "human_first"
    assert not info["independent_success"]
    assert info["assistance_used"]
    assert info["requested_level"] == 4
    assert info["actual_level"] == 2
    assert info["attempts_made"] == 2


def test_ai_first_assistance(monkeypatch):
    env = make_env()
    calls = []

    def fake_attempt(task, operation, level, hint=None):
        calls.append(level)
        return scripted_result(True, level)

    monkeypatch.setattr(env.learner, "attempt", fake_attempt)

    _, _, done, info = env.step(9)

    assert calls == [2]  # action 9 = AI-first level 4, capped at 2
    assert done
    assert info["route_name"] == "ai_first"
    assert info["requested_level"] == 4
    assert info["actual_level"] == 2
    assert info["assistance_used"]
    assert info["attempts_made"] == 1


def test_operation_action_caps():
    for operation, cap in OPERATION_LEVEL_CAPS.items():
        env = make_env(operation)
        valid = env.get_valid_actions()

        for action in range(10):
            requested_level = action % 5
            assert (action in valid) == (requested_level <= cap)


def test_episode_ends_and_rejects_extra_step(monkeypatch):
    env = make_env()

    def fake_attempt(task, operation, level, hint=None):
        return scripted_result(True, level)

    monkeypatch.setattr(env.learner, "attempt", fake_attempt)

    _, _, done, _ = env.step(0)
    assert done

    try:
        env.step(0)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Stepping after episode termination should fail")


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
