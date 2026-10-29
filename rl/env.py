# rl/env.py
from sim.base import Learner
from data.loader import load_tasks, session_pool, transfer_of
import numpy as np

class HumanFirstEnv:
    def __init__(self):
        self.learner = Learner()
        self.all_tasks = load_tasks()  # Load P3's real 600 tasks
        self.session_tasks = []
        self.transfer_tasks = []
        self.current_step = 0
        self.stats = {}

    def reset(self, profile="dummy_profile", seed=None):
        if seed is None:
            seed = np.random.randint(0, 10000)
            
        self.learner.reset(profile, seed)
        self.current_step = 0
        
        # 1. Get the full pool of eligible tasks from P3's loader
        pool = session_pool(self.all_tasks)
        
        # 2. Randomly draw exactly 8 tasks using the episode's seed
        rng = np.random.default_rng(seed)
        self.session_tasks = rng.choice(pool, size=8, replace=False).tolist()
        
        # 3. Get the transfer variants passing the full task dictionary
        self.transfer_tasks = [transfer_of(self.all_tasks, t) for t in self.session_tasks]
        
        # P4 Integration: Calculate pre_transfer on the _T variants before the session
        pre_successes = [self.learner.transfer(t).get("success", False) for t in self.transfer_tasks]
        self.stats = {
            "pre_transfer": sum(pre_successes) / len(pre_successes) if pre_successes else 0.0,
            "takeovers": 0,
            "total_time": 0.0,
            "frustrated_count": 0,
            "abandoned_count": 0,
            "copied_count": 0
        }
        return self._get_obs()

    def step(self, action_id, ablate_operation_penalty=False):
        route = action_id // 5  # 0: HUMAN_FIRST, 1: AI_FIRST
        level = action_id % 5   # 0 to 4

        task = self.session_tasks[self.current_step]
        op_str = task["operations"][0]["op"]
        
        level_cap = 2 if op_str in ["understand_plan", "explain"] else 4
        actual_level = min(level, level_cap)

        # Pass route to learner (P2's simulator will process this)
        result = self.learner.attempt(task, op_str, actual_level, hint="dummy_hint")
        # Track P4's required metrics
        if actual_level >= 3:  # Scaffold or full assistance counts as a takeover
            self.stats["takeovers"] += 1
        self.stats["total_time"] += result.get("time", 0.0)
        if result.get("frustrated", False): self.stats["frustrated_count"] += 1
        if result.get("abandoned", False): self.stats["abandoned_count"] += 1
        if result.get("copied", False): self.stats["copied_count"] += 1

        reward = 1.0 if result.get("success", False) else 0.0
        if not ablate_operation_penalty:
            op_penalty_weight = 0.2 if op_str in ["understand_plan", "explain"] else 0.05
            reward -= (actual_level * op_penalty_weight)
        else:
            reward -= (actual_level * 0.1)

        self.current_step += 1
        done = self.current_step >= len(self.session_tasks)

        info = {
            "immediate_success": result.get("success", False),
            "time": result.get("time", 0.0),
            "frustrated": result.get("frustrated", False),
            "abandoned": result.get("abandoned", False),
            "copied": result.get("copied", False),
            "route": route,
            "level": actual_level
        }

        if done:
            # Post-transfer evaluation on the _T variants
            post_successes = [self.learner.transfer(t).get("success", False) for t in self.transfer_tasks]
            info["transfer_success"] = sum(post_successes) / len(post_successes) if post_successes else 0.0
            info["pre_transfer"] = self.stats["pre_transfer"]
            info["takeover_rate"] = self.stats["takeovers"] / len(self.session_tasks)
            
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        if self.current_step >= len(self.session_tasks):
            return [0, 0, 0, 0, 0, 0, 0, 0]
            
        task = self.session_tasks[self.current_step]
        domain = 0 if task.get("domain") == "coding" else 1
        op_map = {"understand_plan": 0, "execute": 1, "debug_verify": 2, "explain": 3}
        op_str = task["operations"][0]["op"]
        op = op_map.get(op_str, 0)
        
        # Pull dynamic state from P2's learner if available, fallback to 0/1 stubs
        knowledge_domain = getattr(self.learner, "knowledge_domain", 1)
        knowledge_op = getattr(self.learner, "knowledge_op", 1)
        diff = task.get("difficulty", 1)
        attempts = getattr(self.learner, "attempts", 0)
        help_seeking = getattr(self.learner, "help_seeking", 0)
        frustration = getattr(self.learner, "frustration", 0)
        
        return [domain, op, knowledge_domain, knowledge_op, diff, attempts, help_seeking, frustration]