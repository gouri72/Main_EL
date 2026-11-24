"""P1 update compatibility must not swallow internal policy exceptions."""
import pytest
from run_experiment import update_policy

ARGS = ([1], 0, 1.0, [2], False, [0, 1])


def test_six_argument_policy_receives_next_valid_actions():
    class Policy:
        def update(self, state, action, reward, nxt, done, valid):
            self.valid = valid
    policy = Policy()
    update_policy(policy, *ARGS)
    assert policy.valid == [0, 1]


def test_five_argument_legacy_policy_supported():
    class Policy:
        def update(self, state, action, reward, nxt, done):
            self.calls = 1
    policy = Policy()
    update_policy(policy, *ARGS)
    assert policy.calls == 1


def test_internal_type_error_is_not_retried_or_hidden():
    class Policy:
        calls = 0
        def update(self, state, action, reward, nxt, done, valid):
            self.calls += 1
            raise TypeError("internal policy bug")
    policy = Policy()
    with pytest.raises(TypeError, match="internal policy bug"):
        update_policy(policy, *ARGS)
    assert policy.calls == 1


def test_unsupported_signature_fails_without_updating():
    class Policy:
        calls = 0
        def update(self, state, action):
            self.calls += 1
    policy = Policy()
    with pytest.raises(TypeError):
        update_policy(policy, *ARGS)
    assert policy.calls == 0
