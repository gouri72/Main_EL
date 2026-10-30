
#!/usr/bin/env python3
"""Run a configured HumanFirst-Route experiment.

Outputs:
    results/<exp_id>/steps.csv
    results/<exp_id>/episodes.csv
    results/<exp_id>/config.yaml
"""

import argparse
import hashlib
import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

STEPS_COLS = [
    "exp_id",
    "config_hash",
    "policy",
    "sim_family",
    "profile",
    "seed",
    "episode",
    "task_idx",
    "action",
    "route",
    "requested_level",
    "level",
    "reward",
    "success",
    "initial_success",
    "independent_attempt",
    "independent_success",
    "assistance_used",
    "attempts_made",
    "domain",
    "op",
    "time",
    "frustrated",
    "abandoned",
    "copied",
]

EPS_COLS = [
    "exp_id",
    "config_hash",
    "policy",
    "sim_family",
    "profile",
    "seed",
    "episode",
    "total_reward",
    "pre_transfer",
    "post_transfer",
    "transfer_gain",
    "n_steps",
    "n_learner_attempts",
    "full_assist_rate",
    "mean_level",
    "first_attempt_rate",
    "independent_success_rate",
    "assist_spent",
    "takeover_rate",
]


def load_cfg(path):
    """Load YAML and generate a stable configuration hash."""
    with open(path, encoding="utf-8") as file:
        cfg = yaml.safe_load(file)

    body = json.dumps(cfg, sort_keys=True)
    cfg["_hash"] = hashlib.sha256(body.encode()).hexdigest()[:8]
    return cfg


def set_seed(seed):
    """Seed Python and NumPy random generators."""
    random.seed(seed)
    np.random.seed(seed)


def load_tasks(path):
    """Load tasks from a JSONL file."""
    with open(ROOT / path, encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def build(cfg, seed, tasks):
    """Build the environment, policy, session tasks, and transfer pairs."""
    from rl.env import HumanFirstEnv, FAKE_TASKS
    from rl.policy import TabularQPolicy

    family = cfg["simulator"]["family"]
    profile = cfg["simulator"]["profile"]

    all_by_id = {
        task["task_id"]: task
        for task in tasks
    }

    originals = [
        task
        for task in tasks
        if not task["task_id"].endswith("_T")
    ]

    session_count = cfg["data"].get("session_tasks", 8)
    session_tasks = originals[:session_count]

    if not session_tasks:
        session_tasks = FAKE_TASKS

    transfer_tasks = []

    for task in session_tasks:
        transfer_id = task.get("transfer_task_id")

        if transfer_id and transfer_id in all_by_id:
            transfer_tasks.append(all_by_id[transfer_id])

    env = HumanFirstEnv(
        session_tasks,
        profile=profile,
        family=family,
        seed=seed,
    )

    policy_cfg = cfg["policy"]
    policy_name = policy_cfg["name"]

    if policy_name == "tabular_q":
        policy = TabularQPolicy(
            alpha=policy_cfg.get("alpha", 0.1),
            gamma=policy_cfg.get("gamma", 0.9),
            epsilon=policy_cfg.get("epsilon", 0.1),
            seed=seed,
        )
    else:
        raise NotImplementedError(
            f"Policy {policy_name!r} is not registered in this runner."
        )

    return env, policy, session_tasks, transfer_tasks


def evaluate_transfer(learner, transfer_tasks, eval_seed=2026):
    """Estimate transfer success with reproducible random draws.

    IMPORTANT:
    Learner.transfer() currently uses simulated internal mastery.
    It does not execute or validate the paired code/science tasks.

    The learner's original RNG is restored after evaluation so that
    evaluation does not consume the learner's training random stream.
    """
    if not transfer_tasks:
        return None

    original_rng = learner.rng
    outcomes = []

    try:
        for index, task in enumerate(transfer_tasks):
            learner.rng = random.Random(eval_seed + index)

            result = learner.transfer(task)
            outcomes.append(int(result["success"]))
    finally:
        learner.rng = original_rng

    return round(sum(outcomes) / len(outcomes), 4)


def calculate_transfer_gain(pre_transfer, post_transfer):
    """Calculate normalized transfer gain when defined."""
    if (
        pre_transfer is None
        or post_transfer is None
        or pre_transfer >= 1.0
    ):
        return None

    return round(
        (post_transfer - pre_transfer) / (1.0 - pre_transfer),
        4,
    )


def write_csv(path, cfg, columns, rows):
    """Write CSV output with experiment metadata."""
    with open(path, "w", encoding="utf-8", newline="") as file:
        file.write(
            f"# config_hash={cfg['_hash']} exp_id={cfg['exp_id']}\n"
        )
        file.write(",".join(columns) + "\n")

        for row in rows:
            values = []

            for column in columns:
                value = row.get(column)

                if value is None:
                    values.append("")
                else:
                    values.append(str(value))

            file.write(",".join(values) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        help="Path to experiment YAML",
    )
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    tasks = load_tasks(cfg["data"]["tasks"])

    meta = {
        "exp_id": cfg["exp_id"],
        "config_hash": cfg["_hash"],
        "policy": cfg["policy"]["name"],
        "sim_family": cfg["simulator"]["family"],
        "profile": cfg["simulator"]["profile"],
    }

    steps = []
    episodes = []

    from sim.base import Learner

    for seed in cfg["seeds"]:
        set_seed(seed)

        env, policy, session_tasks, transfer_tasks = build(
            cfg,
            seed,
            tasks,
        )

        for episode in range(cfg["n_episodes"]):
            episode_seed = (
                int(seed) * int(cfg["n_episodes"]) + episode
            )
            env.seed = episode_seed

            state = env.reset()
            done = False
            total_reward = 0.0
            step_count = 0

            # Episode-level metrics.
            levels = []
            full_assist_count = 0
            first_attempt_count = 0
            independent_success_count = 0
            total_learner_attempts = 0

            # Use a separate learner for the pre-session measurement.
            baseline_learner = Learner(
                family=cfg["simulator"]["family"]
            )
            baseline_learner.reset(
                cfg["simulator"]["profile"],
                seed=episode_seed + 1_000_003,
            )

            # Use the same evaluation seeds before and after the session.
            evaluation_seed = episode_seed + 2_000_003

            pre_transfer = evaluate_transfer(
                baseline_learner,
                transfer_tasks,
                eval_seed=evaluation_seed,
            )

            final_info = {}

            while not done:
                action = int(
                    policy.get_action(state, is_training=True)
                )

                next_state, reward, done, info = env.step(action)

                policy.update(
                    state,
                    action,
                    reward,
                    next_state,
                    done,
                )

                route = action // 5
                requested_level = action % 5

                # Read actual behavior from the environment.
                actual_level = info["actual_level"]
                assistance_used = info["assistance_used"]
                attempts_made = info["attempts_made"]

                task = session_tasks[
                    min(step_count, len(session_tasks) - 1)
                ]

                # Log the operation actually selected by the environment.
                operation = info.get("operation")

                if operation is None:
                    operations = task.get("operations", [])
                    if operations:
                        operation = operations[0].get("op")

                # Save detailed task-level data.
                steps.append({
                    **meta,
                    "seed": seed,
                    "episode": episode,
                    "task_idx": step_count,
                    "action": action,
                    "route": route,
                    "requested_level": requested_level,
                    "level": actual_level,
                    "reward": round(float(reward), 4),
                    "success": info.get("immediate_success"),
                    "initial_success": info.get("initial_success"),
                    "independent_attempt": info.get(
                        "independent_attempt"
                    ),
                    "independent_success": info.get(
                        "independent_success"
                    ),
                    "assistance_used": assistance_used,
                    "attempts_made": attempts_made,
                    "domain": task.get("domain"),
                    "op": operation,
                    "time": info.get("time"),
                    "frustrated": info.get("frustrated"),
                    "abandoned": info.get("abandoned"),
                    "copied": info.get("copied"),
                })

                # Count tasks where an independent attempt was made.
                first_attempt_count += int(
                    info["independent_attempt"]
                )

                # Count successful independent attempts.
                independent_success_count += int(
                    info["independent_success"]
                )

                # Count actual learner interactions.
                total_learner_attempts += attempts_made

                # Record assistance actually delivered.
                # Zero means no assistance was delivered.
                levels.append(actual_level)

                # Count full assistance only when actually delivered.
                if assistance_used and actual_level == 4:
                    full_assist_count += 1

                total_reward += float(reward)
                step_count += 1
                state = next_state
                final_info = info

            # Measure the post-session transfer proxy.
            post_transfer = evaluate_transfer(
                env.learner,
                transfer_tasks,
                eval_seed=evaluation_seed,
            )

            transfer_gain = calculate_transfer_gain(
                pre_transfer,
                post_transfer,
            )

            # Calculate actual episode-level metrics.
            full_assist_rate = (
                full_assist_count / step_count
                if step_count > 0
                else 0.0
            )

            mean_level = (
                sum(levels) / len(levels)
                if levels
                else 0.0
            )

            # Percentage of all tasks that received a Human-First attempt.
            first_attempt_rate = (
                first_attempt_count / step_count
                if step_count > 0
                else 0.0
            )

            # Percentage of independent attempts that succeeded.
            independent_success_rate = (
                independent_success_count / first_attempt_count
                if first_attempt_count > 0
                else 0.0
            )

            assist_spent = sum(levels)

            episodes.append({
                **meta,
                "seed": seed,
                "episode": episode,
                "total_reward": round(total_reward, 4),
                "pre_transfer": pre_transfer,
                "post_transfer": post_transfer,
                "transfer_gain": transfer_gain,
                "n_steps": step_count,
                "n_learner_attempts": total_learner_attempts,
                "full_assist_rate": round(
                    full_assist_rate,
                    4,
                ),
                "mean_level": round(
                    mean_level,
                    4,
                ),
                "first_attempt_rate": round(
                    first_attempt_rate,
                    4,
                ),
                "independent_success_rate": round(
                    independent_success_rate,
                    4,
                ),
                "assist_spent": int(assist_spent),
                "takeover_rate": final_info.get("takeover_rate"),
            })

    out = ROOT / "results" / cfg["exp_id"]
    out.mkdir(parents=True, exist_ok=True)

    write_csv(
        out / "steps.csv",
        cfg,
        STEPS_COLS,
        steps,
    )

    write_csv(
        out / "episodes.csv",
        cfg,
        EPS_COLS,
        episodes,
    )

    shutil.copy(args.config, out / "config.yaml")

    print(
        f"Wrote {len(steps)} step rows and "
        f"{len(episodes)} episode rows to {out} "
        f"(config_hash={cfg['_hash']})"
    )

    if not transfer_tasks:
        print(
            "WARNING: No paired transfer tasks were found; "
            "transfer metrics will be empty."
        )


if __name__ == "__main__":
    main()
