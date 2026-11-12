"""Policy registry: the RL policies and all baselines of the master prompt, behind one name -> object factory.

State layout (see rl/configured_env.py): [domain, op, domain_bin, op_bin, difficulty, attempts_bin, help_bin, frustration_bin]
Action id = route*5 + level (route 0 = HUMAN_FIRST, 1 = AI_FIRST); the env caps the level per operation.

Names: tabular_q | linear_q | linucb | performance_only | domain_specific |
       always_full | always_none | fixed_human_first | fixed_ladder | rule_adaptive
`performance_only` = tabular Q-learning rewarded for immediate success only (no transfer/penalty terms).
"""
from rl.policy import TabularQPolicy, LinearQPolicy, LinUCBPolicy
from rl.baselines import (
    AlwaysFullPolicy,
    AlwaysNonePolicy,
    FixedLadderPolicy,
    FixedHumanFirstPolicy,
    RuleBasedPolicy,
)


class DomainSpecificQ:
    """One independent tabular Q-policy per domain (coding / science)."""
    def __init__(self, **kw):
        self.pols = {0: TabularQPolicy(**kw), 1: TabularQPolicy(**kw)}

    def get_action(self, state, valid_actions=None, is_training=True):
        return self.pols[state[0]].get_action(state, valid_actions, is_training)

    def update(self, state, action, reward, next_state, done, next_valid_actions=None):
        self.pols[state[0]].update(state, action, reward, next_state, done, next_valid_actions)


def make_policy(pc, seed):
    name = pc["name"]
    q = dict(alpha=pc.get("alpha", 0.1), gamma=pc.get("gamma", 0.9), epsilon=pc.get("epsilon", 0.1), seed=seed)
    
    table = {
        "tabular_q": lambda: TabularQPolicy(**q),
        "performance_only": lambda: TabularQPolicy(**q),
        "domain_specific": lambda: DomainSpecificQ(**q),
        "linear_q": lambda: LinearQPolicy(alpha=pc.get("alpha", 0.01), gamma=q["gamma"], epsilon=q["epsilon"], seed=seed),
        "linucb": lambda: LinUCBPolicy(alpha=pc.get("alpha", 1.0), seed=seed),
        "always_full": AlwaysFullPolicy,
        "always_none": AlwaysNonePolicy,
        "fixed_human_first": FixedHumanFirstPolicy,
        "fixed_ladder": FixedLadderPolicy,
        "rule_adaptive": RuleBasedPolicy,
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