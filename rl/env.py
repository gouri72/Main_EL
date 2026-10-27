# rl/env.py
from sim.base import Learner

class HumanFirstEnv:
    def __init__(self, tasks):
        self.learner = Learner()
        self.tasks = tasks
        self.current_task_idx = 0

    def reset(self):
        self.learner.reset("dummy_profile", 42)
        self.current_task_idx = 0
        return self._get_obs()

    def step(self, action_id, ablate_operation_penalty=False):
        route = action_id // 5  # 0: HUMAN_FIRST, 1: AI_FIRST
        level = action_id % 5   # 0 to 4

        task = self.tasks[self.current_task_idx]
        op_str = task["operations"][0]["op"]
        
        # Enforce operation-specific caps (Rule from Section 1 & 4)
        # Deep reasoning capped at lower assistance, execution allows higher
        level_cap = 2 if op_str in ["understand_plan", "explain"] else 4
        actual_level = min(level, level_cap)

        result = self.learner.attempt(task, op_str, actual_level, hint="dummy_hint")

        # --- REWARD FUNCTION ---
        reward = 1.0 if result["success"] else 0.0

        if not ablate_operation_penalty:
            # Novelty 2: Operation-depth takeover penalty
            # Heavier penalty (0.2) for doing the reasoning for them, lighter (0.05) for execution help
            op_penalty_weight = 0.2 if op_str in ["understand_plan", "explain"] else 0.05
            reward -= (actual_level * op_penalty_weight)
        else:
            # Operation-blind baseline (for ablation studies later)
            reward -= (actual_level * 0.1)

        self.current_task_idx += 1
        done = self.current_task_idx >= len(self.tasks)

        info = {"immediate_success": result["success"]}
        if done:
            transfer_res = self.learner.transfer(task)
            info["transfer_success"] = transfer_res["success"]

        return self._get_obs(), reward, done, info

    def _get_obs(self):
        # Handle the terminal state at the end of an episode
        if self.current_task_idx >= len(self.tasks):
            return [0, 0, 0, 0, 0, 0, 0, 0]
            
        task = self.tasks[self.current_task_idx]
        
        # 1. Domain (0: coding, 1: science)
        domain = 0 if task["domain"] == "coding" else 1
        
        # 2. Operation (0: understand_plan, 1: execute, 2: debug_verify, 3: explain)
        op_map = {"understand_plan": 0, "execute": 1, "debug_verify": 2, "explain": 3}
        op_str = task["operations"][0]["op"]
        op = op_map.get(op_str, 0)
        
        # 3 & 4. Knowledge bins (Waiting on P2's estimator, stubbed as 1)
        knowledge_domain = 1
        knowledge_op = 1
        
        # 5. Difficulty (1, 2, or 3)
        diff = task.get("difficulty", 1)
        
        # 6, 7, 8. Attempts, Help-seeking, Frustration 
        # (These will dynamically update when P2 finishes the simulator. Stubbed for now.)
        attempts = 0
        help_seeking = 0
        frustration = 0
        
        return [domain, op, knowledge_domain, knowledge_op, diff, attempts, help_seeking, frustration]

# 10 Fake Tasks for Day-1 testing
FAKE_TASKS = [
    {"task_id": f"t{i}", "domain": "coding", "difficulty": 1, "operations": [{"op": "understand_plan"}]}
    for i in range(10)
]