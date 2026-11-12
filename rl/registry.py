"""Policy registry: the RL policies and all baselines of the master prompt, behind one name -> object factory.

State layout (see rl/configured_env.py): [domain, op, domain_bin, op_bin, difficulty, attempts_bin, help_bin, frustration_bin]
Action id = route*5 + level (route 0 = HUMAN_FIRST, 1 = AI_FIRST); the env caps the level per operation.

Names: tabular_q | linear_q | linucb | performance_only | domain_specific |
       always_full | always_none | fixed_human_first | fixed_ladder | rule_adaptive
`performance_only` = tabular Q-learning rewarded for immediate success only (no transfer/penalty terms).
"""
from rl.policy import TabularQPolicy, LinearQPolicy, LinUCBPolicy


class _Fixed:
    def __init__(self, action):
        self.action = int(action)

    def get_action(self, state, is_training=True):
        return self.action

    def update(self, *args, **kwargs):
        pass


class FixedLadder:
    """cue -> hint -> scaffold -> full. Human first; after every task on which help was needed the next task starts one rung higher. Never fades."""
    def get_action(self, state, is_training=True):
        return min(1 + state[6], 4)

    def update(self, *args, **kwargs):
        pass


class RuleAdaptive:
    """Human first; level from observable knowledge, difficulty and frustration (a hand-written fading rule)."""
    def get_action(self, state, is_training=True):
        dom_bin, diff, frustrated = state[2], state[4], state[7]
        if frustrated:
            level = 3
        elif dom_bin == 0:
            level = 2 if diff < 3 else 3
        elif dom_bin == 1:
            level = 1 if diff < 3 else 2
        else:
            level = 0 if diff < 3 else 1
        return level

    def update(self, *args, **kwargs):
        pass


class DomainSpecificQ:
    """One independent tabular Q-policy per domain (coding / science)."""
    def __init__(self, **kw):
        self.pols = {0: TabularQPolicy(**kw), 1: TabularQPolicy(**kw)}

    def get_action(self, state, is_training=True):
        return self.pols[state[0]].get_action(state, is_training)

    def update(self, state, action, reward, next_state, done):
        self.pols[state[0]].update(state, action, reward, next_state, done)


def make_policy(pc, seed):
    name = pc["name"]
    q = dict(alpha=pc.get("alpha", 0.1), gamma=pc.get("gamma", 0.9), epsilon=pc.get("epsilon", 0.1), seed=seed)
    table = {
        "tabular_q": lambda: TabularQPolicy(**q),
        "performance_only": lambda: TabularQPolicy(**q),
        "domain_specific": lambda: DomainSpecificQ(**q),
        "linear_q": lambda: LinearQPolicy(alpha=pc.get("alpha", 0.01), gamma=q["gamma"], epsilon=q["epsilon"], seed=seed),
        "linucb": lambda: LinUCBPolicy(alpha=pc.get("alpha", 1.0), seed=seed),
        "always_full": lambda: _Fixed(9),
        "always_none": lambda: _Fixed(0),
        "fixed_human_first": lambda: _Fixed(2),
        "fixed_ladder": FixedLadder,
        "rule_adaptive": RuleAdaptive,
    }
    if name not in table:
        raise NotImplementedError(f"Policy {name!r} is not registered; choose from {sorted(table)}")
    return table[name]()


def reward_for(cfg):
    """Reward settings from the YAML; performance_only forces an immediate-success-only reward."""
    r = dict(cfg.get("reward") or {})
    if cfg["policy"]["name"] == "performance_only":
        r.update(w_transfer=0.0, w_full_assist_penalty=0.0, w_takeover_penalty=0.0)
    return r