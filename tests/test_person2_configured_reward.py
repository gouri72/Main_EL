import pytest
from rl.configured_env import ConfiguredEnv, OP_PENALTY
from rl.env import HumanFirstEnv


def make_env(reward=None, transfer_repeats=3):
    task = {"id": "reward-test", "domain": "coding", "difficulty": 1,
            "operations": [{"op": "execute"}]}
    return ConfiguredEnv(tasks=[task], reward=reward, transfer_repeats=transfer_repeats,
                        profile="advanced", family="B", seed=42)


def parent_step(info, done=True):
    def fake(self, action_id):
        return [0] * 8, 0.0, done, dict(info)
    return fake


def test_progress_reward(monkeypatch):
    env = make_env()
    monkeypatch.setattr(HumanFirstEnv, "step", parent_step({
        "immediate_success": True, "actual_level": 0,
        "operation": "execute", "pre_transfer": None, "transfer_success": None
    }))
    _, reward, _, info = env.step(5)
    assert reward == pytest.approx(1.0)
    assert info["reward"] == pytest.approx(reward)


def test_operation_takeover_penalty_uses_actual_level(monkeypatch):
    env = make_env()
    monkeypatch.setattr(HumanFirstEnv, "step", parent_step({
        "immediate_success": True, "actual_level": 2,
        "operation": "understand_plan", "pre_transfer": None, "transfer_success": None
    }))
    _, reward, _, _ = env.step(7)
    assert reward == pytest.approx(1.0 - 2 * OP_PENALTY["understand_plan"])


def test_full_assistance_penalty(monkeypatch):
    env = make_env()
    monkeypatch.setattr(HumanFirstEnv, "step", parent_step({
        "immediate_success": True, "actual_level": 4,
        "operation": "execute", "pre_transfer": None, "transfer_success": None
    }))
    _, reward, _, _ = env.step(9)
    assert reward == pytest.approx(1.0 - 4 * OP_PENALTY["execute"] - 0.5)


def test_transfer_reward_is_post_minus_pre(monkeypatch):
    env = make_env({"w_transfer": 2.0})
    monkeypatch.setattr(HumanFirstEnv, "step", parent_step({
        "immediate_success": True, "actual_level": 0,
        "operation": "execute", "pre_transfer": 0.2, "transfer_success": 0.6
    }))
    _, reward, done, _ = env.step(5)
    assert done
    assert reward == pytest.approx(1.0 + 2.0 * (0.6 - 0.2))


def test_reward_ablation_disables_all_reward_terms(monkeypatch):
    env = make_env({"ablate": ["progress", "transfer", "full_assist_penalty", "takeover_penalty"]})
    monkeypatch.setattr(HumanFirstEnv, "step", parent_step({
        "immediate_success": True, "actual_level": 4,
        "operation": "execute", "pre_transfer": 0.1, "transfer_success": 0.9
    }))
    _, reward, _, _ = env.step(9)
    assert reward == pytest.approx(0.0)


def test_uniform_operation_penalty_ablation(monkeypatch):
    env = make_env({"ablate": ["operation_penalty"]})
    monkeypatch.setattr(HumanFirstEnv, "step", parent_step({
        "immediate_success": False, "actual_level": 2,
        "operation": "understand_plan", "pre_transfer": None, "transfer_success": None
    }))
    _, reward, _, _ = env.step(7)
    assert reward == pytest.approx(-2 * 0.1)


def test_transfer_average_and_rng_state_preservation(monkeypatch):
    env = make_env(transfer_repeats=3)
    env.transfer_tasks = [{"id": "t1"}, {"id": "t2"}]
    state_before = env.learner.rng.getstate()
    results = iter([1, 0, 1, 0, 0, 1])

    def fake_transfer(task):
        env.learner.rng.random()
        return {"success": next(results)}

    monkeypatch.setattr(env.learner, "transfer", fake_transfer)
    score = env._evaluate_transfer()
    assert score == pytest.approx(3 / 6)
    assert env.learner.rng.getstate() == state_before


def test_no_transfer_tasks_returns_none():
    env = make_env()
    env.transfer_tasks = []
    assert env._evaluate_transfer() is None

@pytest.mark.parametrize("repeats", [0, -1])
def test_transfer_repeats_must_be_positive(repeats):
    task = {"id": "invalid-repeats", "domain": "coding", "difficulty": 1,
            "operations": [{"op": "execute"}]}
    with pytest.raises(ValueError, match="transfer_repeats must be a positive integer"):
        ConfiguredEnv(tasks=[task], transfer_repeats=repeats,
                     profile="advanced", family="B", seed=42)
