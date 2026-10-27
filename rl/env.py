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

    def step(self, action_id):
        route = action_id // 5  # 0: HUMAN_FIRST, 1: AI_FIRST
        level = action_id % 5   # 0 to 4

        task = self.tasks[self.current_task_idx]
        op = task["operations"][0]["op"]
        
        # Call the dummy learner
        result = self.learner.attempt(task, op, level, hint="dummy_hint")

        # Basic stub reward: reward immediate success, penalize high assistance
        reward = 1.0 if result["success"] else 0.0
        reward -= (level * 0.1)

        self.current_task_idx += 1
        done = self.current_task_idx >= len(self.tasks)

        info = {"immediate_success": result["success"]}
        if done:
            transfer_res = self.learner.transfer(task)
            info["transfer_success"] = transfer_res["success"]

        return self._get_obs(), reward, done, info

    def _get_obs(self):
        # Stub state: [domain, op, knowledge_domain, knowledge_op, diff, attempts, help, frustration]
        return [0, 0, 1, 1, 1, 0, 0, 0]

# 10 Fake Tasks for Day-1 testing
FAKE_TASKS = [
    {"task_id": f"t{i}", "domain": "coding", "difficulty": 1, "operations": [{"op": "understand_plan"}]}
    for i in range(10)
]