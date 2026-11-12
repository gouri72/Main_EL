"""HumanFirstEnv + the reward, state encoding and ablations that the experiment YAML asks for.

Wraps P2's rl/env.py without editing it (subclass). What it adds:
  * reward = w_progress*success - w_takeover*level*operation_penalty - w_full_assist*(level==4)
             + w_transfer*(post_transfer - pre_transfer)   [added on the last step of the session]
    -> the "reward for later unaided transfer" of the frozen design (env.py alone has no transfer term).
  * low-variance transfer estimate: transfer_repeats simulated attempts per transfer task (cfg key transfer_repeats, default 20)
  * discretized state (attempts 3 bins, help-seeking 4 bins, frustration 2 bins) as in the master prompt;
    optional state.drop = list of feature indices set to 0 (smaller state space)
  * ablations (reward.ablate list in the YAML):
      leave-one-out reward terms : progress | transfer | full_assist_penalty | takeover_penalty
      operation-blind pieces     : operation_state (policy cannot see the operation)
                                   operation_penalty (same penalty for every operation)
                                   operation_caps (all caps = 4; changes the shared dict IN PLACE, one config per process)
State layout: [domain, op, domain_bin, op_bin, difficulty, attempts_bin, help_bin, frustration_bin]
"""
from rl.env import HumanFirstEnv
from sim.base import OPERATION_LEVEL_CAPS

OP_PENALTY = {"understand_plan": 0.2, "explain": 0.2, "debug_verify": 0.05, "execute": 0.05}
ALIASES = {"takeover_penalty": "takeover", "full_assist_penalty": "full_assist"}
TERMS = {"progress", "transfer", "full_assist", "takeover"}
SWITCHES = {"operation_state", "operation_penalty", "operation_caps"}


def bin_attempts(n):
    return 0 if n <= 2 else 1 if n <= 5 else 2


def bin_help(n):
    return 0 if n == 0 else 1 if n <= 2 else 2 if n <= 4 else 3


def bin_frustration(n):
    return 0 if n == 0 else 1


class ConfiguredEnv(HumanFirstEnv):
    def __init__(self, tasks, reward=None, discretize=True, transfer_repeats=20, drop=(), **kw):
        super().__init__(tasks, **kw)
        self.transfer_repeats = int(transfer_repeats)
        self.drop = tuple(drop)
        r = {"w_progress": 1.0, "w_transfer": 2.0, "w_full_assist_penalty": 0.5, "w_takeover_penalty": 1.0, "ablate": []}
        r.update(reward or {})
        ablate = {ALIASES.get(a, a) for a in (r.get("ablate") or [])}
        unknown = ablate - TERMS - SWITCHES
        if unknown:
            raise ValueError(f"unknown ablation(s) {sorted(unknown)}; allowed: {sorted(TERMS | SWITCHES)}")
        self.w = {
            "progress": 0.0 if "progress" in ablate else float(r["w_progress"]),
            "transfer": 0.0 if "transfer" in ablate else float(r["w_transfer"]),
            "full_assist": 0.0 if "full_assist" in ablate else float(r["w_full_assist_penalty"]),
            "takeover": 0.0 if "takeover" in ablate else float(r["w_takeover_penalty"]),
        }
        self.blind_state = "operation_state" in ablate
        self.uniform_penalty = "operation_penalty" in ablate
        self.discretize = discretize
        if "operation_caps" in ablate:
            for op in OPERATION_LEVEL_CAPS:
                OPERATION_LEVEL_CAPS[op] = 4

    def _evaluate_transfer(self):
        """Unaided transfer success averaged over `transfer_repeats` simulated attempts per transfer task.
        One draw per task (env.py default) makes post-pre so noisy (SD ~0.2) that a transfer reward cannot be learned."""
        if not self.transfer_tasks:
            return None
        rng_state = self.learner.rng.getstate()
        try:
            hits = sum(
                self.learner.transfer(t)["success"]
                for t in self.transfer_tasks for _ in range(self.transfer_repeats)
            )
        finally:
            self.learner.rng.setstate(rng_state)
        return hits / (len(self.transfer_tasks) * self.transfer_repeats)

    def step(self, action_id):
        obs, _, done, info = super().step(action_id)
        success = 1.0 if info["immediate_success"] else 0.0
        level = int(info["actual_level"])
        pen = 0.1 if self.uniform_penalty else OP_PENALTY[info["operation"]]
        reward = self.w["progress"] * success - self.w["takeover"] * level * pen
        if level == 4:
            reward -= self.w["full_assist"]
        if done and info.get("pre_transfer") is not None and info.get("transfer_success") is not None:
            reward += self.w["transfer"] * (info["transfer_success"] - info["pre_transfer"])
        info["reward"] = reward
        return obs, reward, done, info

    def _get_obs(self):
        domain, op, dbin, obin, diff, att, hlp, frs = super()._get_obs()
        if self.blind_state:
            op, obin = 0, 0
        if self.discretize:
            att, hlp, frs = bin_attempts(att), bin_help(hlp), bin_frustration(frs)
        obs = [domain, op, dbin, obin, diff, att, hlp, frs]
        for i in self.drop:
            obs[i] = 0
        return obs