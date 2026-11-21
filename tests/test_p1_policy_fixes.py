
import numpy as np
import pytest

from rl.policy import (
    TabularQPolicy,
    LinearQPolicy,
    LinUCBPolicy,
    get_features,
)


STATE = [0, 0, 1, 1, 2, 0, 0, 0]
NEXT = [0, 1, 1, 1, 2, 1, 1, 0]

VALID = [0, 1, 2, 5, 6, 7]


@pytest.mark.parametrize(
    "policy_class",
    [
        TabularQPolicy,
        LinearQPolicy,
        LinUCBPolicy,
    ]
)
def test_valid_actions(policy_class):
    policy = policy_class(seed=42)

    for training in [True, False]:
        for _ in range(50):
            action = policy.get_action(
                STATE,
                valid_actions=VALID,
                is_training=training
            )

            assert action in VALID


@pytest.mark.parametrize(
    "policy_class",
    [
        TabularQPolicy,
        LinearQPolicy,
        LinUCBPolicy,
    ]
)
def test_policy_update(policy_class):
    policy = policy_class(seed=42)

    action = policy.get_action(
        STATE,
        valid_actions=VALID
    )

    policy.update(
        STATE,
        action,
        reward=1.0,
        next_state=NEXT,
        done=False,
        next_valid_actions=VALID
    )


def test_normalized_features():
    x = get_features(STATE)

    assert len(x) == 9
    assert np.all(np.isfinite(x))
    assert np.all(x[:8] >= 0)
    assert np.all(x[:8] <= 1)


def test_invalid_actions_rejected():
    policy = LinearQPolicy()

    with pytest.raises(ValueError):
        policy.get_action(
            STATE,
            valid_actions=[]
        )
