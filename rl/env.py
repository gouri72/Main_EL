
"""Human-first RL environment with observable knowledge estimation."""

import numbers

from sim.base import Learner, OPERATION_LEVEL_CAPS
from sim.knowledge import KnowledgeEstimator


class HumanFirstEnv:
    """Environment for learning assistance policies.

    Action encoding:
        0-4: HUMAN_FIRST with assistance levels 0-4
        5-9: AI_FIRST with assistance levels 0-4
    """

    def __init__(
        self,
        tasks,
        profile="intermediate",
        family="A",
        seed=42,
    ):
        if not tasks:
            raise ValueError("tasks must contain at least one task")

        self.tasks = tasks
        self.profile = profile
        self.family = family
        self.seed = seed

        self.learner = Learner(family=family)
        self.estimator = KnowledgeEstimator()

        self.current_task_idx = 0

        # Task-level and interaction-level counters
        self.attempt_count = 0
        self.learner_attempt_count = 0
        self.help_seeking_count = 0
        self.frustration_count = 0

        self.time_total = 0.0
        self.copy_count = 0
        self.abandoned_count = 0

        self.pre_transfer = None

    def reset(self):
        """Reset learner, estimator, and episode counters."""

        self.learner.reset(self.profile, self.seed)
        self.estimator.reset()

        self.current_task_idx = 0

        self.attempt_count = 0
        self.learner_attempt_count = 0
        self.help_seeking_count = 0
        self.frustration_count = 0

        self.time_total = 0.0
        self.copy_count = 0
        self.abandoned_count = 0

        self.pre_transfer = None

        return self._get_obs()

    def _get_current_operation(self, task, task_index=None):
        """Select an operation based on task index and available operations."""

        operations = task.get("operations", [])

        if not operations:
            raise ValueError(
                f"Task {task.get('task_id', '<unknown>')} has no operations"
            )

        if task_index is None:
            task_index = self.current_task_idx

        operation_index = task_index % len(operations)
        operation = operations[operation_index].get("op")

        if operation not in OPERATION_LEVEL_CAPS:
            raise ValueError(
                f"Unsupported task operation: {operation}"
            )

        return operation

    def step(self, action_id, ablate_operation_penalty=False):
        """Execute one action and return observation, reward, done, info."""

        if self.current_task_idx >= len(self.tasks):
            raise RuntimeError(
                "Episode finished. Call reset() before stepping again."
            )

        if (
            isinstance(action_id, bool)
            or not isinstance(action_id, numbers.Integral)
        ):
            raise ValueError("action_id must be an integer from 0 to 9")

        action_id = int(action_id)

        if not 0 <= action_id < 10:
            raise ValueError("action_id must be an integer from 0 to 9")

        # Decode action.
        route = action_id // 5
        requested_level = action_id % 5

        task = self.tasks[self.current_task_idx]
        domain = task["domain"]
        operation = self._get_current_operation(task)

        # Apply operation-specific assistance cap.
        level_cap = OPERATION_LEVEL_CAPS[operation]
        capped_level = min(requested_level, level_cap)

        results = []

        independent_attempt = route == 0
        independent_success = False
        assistance_used = False

        # --------------------------------------------------
        # HUMAN-FIRST: Attempt independently before help.
        # --------------------------------------------------
        if route == 0:

            first_result = self.learner.attempt(
                task,
                operation,
                0,
                hint=None,
            )

            results.append(first_result)

            independent_success = bool(first_result["success"])

            self.estimator.update(
                domain=domain,
                operation=operation,
                success=first_result["success"],
                level=0,
            )

            # Request assistance only if the independent attempt
            # failed, the learner did not abandon, and help is available.
            if (
                not first_result["success"]
                and not first_result["abandoned"]
                and capped_level > 0
            ):

                assistance_used = True

                assisted_result = self.learner.attempt(
                    task,
                    operation,
                    capped_level,
                    hint="dummy_hint",
                )

                results.append(assisted_result)

                self.estimator.update(
                    domain=domain,
                    operation=operation,
                    success=assisted_result["success"],
                    level=capped_level,
                )

        # --------------------------------------------------
        # AI-FIRST: Assistance is provided immediately.
        # --------------------------------------------------
        else:

            assistance_used = capped_level > 0

            assisted_result = self.learner.attempt(
                task,
                operation,
                capped_level,
                hint="dummy_hint",
            )

            results.append(assisted_result)

            self.estimator.update(
                domain=domain,
                operation=operation,
                success=assisted_result["success"],
                level=capped_level,
            )

        # Final outcome of the task.
        final_result = results[-1]

        success = bool(final_result["success"])

        initial_success = bool(results[0]["success"])

        attempts_made = len(results)

        total_time = round(
            sum(result["time"] for result in results),
            2,
        )

        frustrated = any(
            result["frustrated"] for result in results
        )

        abandoned = any(
            result["abandoned"] for result in results
        )

        copied = any(
            result["copied"] for result in results
        )

        # --------------------------------------------------
        # Update episode metrics.
        # --------------------------------------------------

        self.time_total += total_time

        self.copy_count += int(copied)

        self.abandoned_count += int(abandoned)

        # One environment step corresponds to one task.
        self.attempt_count += 1

        # Actual learner interactions may be greater than task count.
        self.learner_attempt_count += attempts_made

        if assistance_used:
            self.help_seeking_count += 1

        if frustrated:
            self.frustration_count += 1

        # Actual assistance level is zero when no help was delivered.
        actual_level = capped_level if assistance_used else 0

        # --------------------------------------------------
        # Reward: final success minus assistance cost.
        # --------------------------------------------------

        reward = 1.0 if success else 0.0

        if ablate_operation_penalty:
            penalty_weight = 0.1
        else:
            penalty_weight = (
                0.2
                if operation in ("understand_plan", "explain")
                else 0.05
            )

        if assistance_used:
            reward -= actual_level * penalty_weight

        # Advance to the next task.
        self.current_task_idx += 1

        done = self.current_task_idx >= len(self.tasks)

        info = {
            "task_id": task.get("task_id"),

            "immediate_success": success,
            "initial_success": initial_success,

            "independent_attempt": independent_attempt,
            "independent_success": independent_success,

            "assistance_used": assistance_used,
            "attempts_made": attempts_made,

            "route": route,
            "route_name": (
                "human_first" if route == 0 else "ai_first"
            ),

            "operation": operation,

            "requested_level": requested_level,
            "actual_level": actual_level,
            "level_cap": level_cap,

            "reward": reward,
            "time": total_time,

            "frustrated": frustrated,
            "abandoned": abandoned,
            "copied": copied,
        }

        if done:

            # Preliminary simulator transfer outcome only.
            transfer_result = self.learner.transfer(task)

            info["pre_transfer"] = self.pre_transfer

            info["transfer_success"] = transfer_result["success"]

            # Takeover rate = copied tasks / total tasks.
            info["takeover_rate"] = (
                self.copy_count / self.attempt_count
                if self.attempt_count > 0
                else 0.0
            )

            info["time_total"] = round(self.time_total, 2)

            info["abandoned_count"] = self.abandoned_count

            info["frustration_count"] = self.frustration_count

            info["help_seeking_count"] = self.help_seeking_count

            info["task_count"] = self.attempt_count

            info["learner_attempt_count"] = (
                self.learner_attempt_count
            )

        return self._get_obs(), reward, done, info

    def _get_obs(self):
        """Return the existing eight-feature observable state."""

        if self.current_task_idx >= len(self.tasks):
            return [0, 0, 0, 0, 0, 0, 0, 0]

        task = self.tasks[self.current_task_idx]
        domain = task["domain"]

        operation = self._get_current_operation(
            task,
            self.current_task_idx,
        )

        domain_map = {
            "coding": 0,
            "science": 1,
        }

        operation_map = {
            "understand_plan": 0,
            "execute": 1,
            "debug_verify": 2,
            "explain": 3,
        }

        if domain not in domain_map:
            raise ValueError(f"Unsupported task domain: {domain}")

        if operation not in operation_map:
            raise ValueError(f"Unsupported task operation: {operation}")

        estimates = self.estimator.estimate(domain, operation)

        domain_knowledge = estimates["domain_knowledge"]
        operation_knowledge = estimates["operation_knowledge"]

        domain_bin = (
            0 if domain_knowledge < 0.4
            else 1 if domain_knowledge < 0.7
            else 2
        )

        operation_bin = (
            0 if operation_knowledge < 0.4
            else 1 if operation_knowledge < 0.7
            else 2
        )

        difficulty = task.get("difficulty", 1)

        return [
            domain_map[domain],
            operation_map[operation],
            domain_bin,
            operation_bin,
            difficulty,
            self.attempt_count,
            self.help_seeking_count,
            self.frustration_count,
        ]


# Small task set for local environment testing.
FAKE_TASKS = [
    {
        "task_id": f"t{i}",
        "domain": "coding",
        "difficulty": 1,
        "operations": [{"op": "understand_plan"}],
    }
    for i in range(10)
]
