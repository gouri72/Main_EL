import pytest
from rl.configured_env import ConfiguredEnv, OP_PENALTY
from rl.env import HumanFirstEnv
from sim.base import Learner, OPERATION_LEVEL_CAPS


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



@pytest.mark.parametrize("invalid_cap", [-1, 5, 2.5, True, "2"])
def test_operation_caps_validation(invalid_cap):
    caps = dict(OPERATION_LEVEL_CAPS)
    caps["understand_plan"] = invalid_cap

    with pytest.raises(ValueError):
        Learner(operation_caps=caps)


@pytest.mark.parametrize("valid_cap", [0, 4])
def test_operation_caps_accept_boundaries(valid_cap):
    caps = dict(OPERATION_LEVEL_CAPS)
    caps["understand_plan"] = valid_cap

    learner = Learner(operation_caps=caps)
    assert learner.operation_caps["understand_plan"] == valid_cap

@pytest.mark.parametrize("invalid_drop", [(8,), (-1,), ("1",), (1.5,), (True,), (1, 1)])
def test_drop_indices_must_be_valid(invalid_drop):
    task = {"id": "drop-test", "domain": "coding", "difficulty": 1,
            "operations": [{"op": "execute"}]}
    with pytest.raises(ValueError, match="drop"):
        ConfiguredEnv(tasks=[task], drop=invalid_drop,
                     profile="advanced", family="B", seed=42)


@pytest.mark.parametrize("valid_drop", [(), (0,), (7,), (0, 7)])
def test_drop_indices_accept_valid_values(valid_drop):
    task = {"id": "drop-test", "domain": "coding", "difficulty": 1,
            "operations": [{"op": "execute"}]}
    env = ConfiguredEnv(tasks=[task], drop=valid_drop,
                        profile="advanced", family="B", seed=42)
    obs = env._get_obs()
    assert len(obs) == 8
    for index in valid_drop:
        assert obs[index] == 0


@pytest.mark.parametrize("weight_key", [
    "w_progress", "w_transfer",
    "w_full_assist_penalty", "w_takeover_penalty"
])
@pytest.mark.parametrize("invalid_value", [
    -1, float("nan"), float("inf"), float("-inf"), True, "1", None
])
def test_reward_weights_reject_invalid_values(weight_key, invalid_value):
    with pytest.raises(ValueError, match="finite non-negative"):
        make_env({weight_key: invalid_value})


@pytest.mark.parametrize("valid_value", [0, 0.5, 1, 2, 100])
def test_reward_weights_accept_valid_values(valid_value):
    make_env({
        "w_progress": valid_value,
        "w_transfer": valid_value,
        "w_full_assist_penalty": valid_value,
        "w_takeover_penalty": valid_value,
    })


def test_reward_rejects_unknown_keys():
    with pytest.raises(ValueError, match="unknown reward key"):
        make_env({"w_typo": 1.0})

@pytest.mark.parametrize("repeats", [1.5, 2.9, True, "5", None])
def test_transfer_repeats_reject_invalid_types(repeats):
    task = {"id": "invalid-repeats", "domain": "coding", "difficulty": 1,
            "operations": [{"op": "execute"}]}
    with pytest.raises(ValueError, match="transfer_repeats must be a positive integer"):
        ConfiguredEnv(tasks=[task], transfer_repeats=repeats,
                     profile="advanced", family="B", seed=42)

@pytest.mark.parametrize("invalid_ablate", ["transfer", 1, True, {"transfer": True}, [1], [True]])
def test_ablate_rejects_invalid_types(invalid_ablate):
    with pytest.raises(ValueError, match="ablate must be a list or tuple of strings"):
        make_env({"ablate": invalid_ablate})

@pytest.mark.parametrize("invalid_reward", ["invalid", 1, True, [], ()])
def test_reward_rejects_non_dict_input(invalid_reward):
    task = {"id": "invalid-reward", "domain": "coding", "difficulty": 1,
            "operations": [{"op": "execute"}]}
    with pytest.raises(ValueError, match="reward must be a dictionary"):
        ConfiguredEnv(tasks=[task], reward=invalid_reward,
                     profile="advanced", family="B", seed=42)
