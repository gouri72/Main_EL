
#!/usr/bin/env python3

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

from data.loader import load_tasks, session_pool

STEPS_COLS = [
    "exp_id", "config_hash", "policy", "sim_family", "profile",
    "seed", "episode", "task_idx", "action", "route", "level",
    "reward", "success", "domain", "op", "time", "frustrated",
    "abandoned", "copied"
]

EPS_COLS = [
    "exp_id", "config_hash", "policy", "sim_family", "profile",
    "seed", "episode", "total_reward", "pre_transfer",
    "post_transfer", "transfer_gain", "n_steps",
    "full_assist_rate", "mean_level", "first_attempt_rate",
    "assist_spent", "takeover_rate"
]


def load_cfg(path):
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    body = json.dumps(cfg, sort_keys=True)
    cfg["_hash"] = hashlib.sha256(body.encode()).hexdigest()[:8]
    return cfg


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def build(cfg, seed, all_tasks):
    """Build P2 environment and policy using the complete task dictionary."""

    from rl.env import HumanFirstEnv
    from rl.policy import TabularQPolicy

    pool = session_pool(all_tasks)

    n_tasks = cfg["data"].get("session_tasks", 8)

    if len(pool) < n_tasks:
        raise ValueError(
            f"Need {n_tasks} session tasks, but only {len(pool)} available"
        )

    env = HumanFirstEnv(
        tasks=pool[:n_tasks],
        profile=cfg["simulator"]["profile"],
        family=cfg["simulator"]["family"],
        seed=seed,
        all_tasks=all_tasks
    )

    pc = cfg["policy"]

    if pc["name"] == "tabular_q":
        pol = TabularQPolicy(
            alpha=pc.get("alpha", 0.1),
            gamma=pc.get("gamma", 0.9),
            epsilon=pc.get("epsilon", 0.1),
            seed=seed
        )
    else:
        raise NotImplementedError(
            f"Policy '{pc['name']}' is not registered"
        )

    return env, pol, pool


def write_csv(path, cfg, cols, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(
            f"# config_hash={cfg['_hash']} exp_id={cfg['exp_id']}\n"
        )
        f.write(",".join(cols) + "\n")

        for row in rows:
            f.write(
                ",".join(
                    "" if row.get(col) is None else str(row.get(col))
                    for col in cols
                ) + "\n"
            )


def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    args = ap.parse_args()

    cfg = load_cfg(args.config)

    # P3: Load all 600 tasks, including transfer variants.
    tasks = load_tasks(ROOT / cfg["data"]["tasks"])

    meta = {
        "exp_id": cfg["exp_id"],
        "config_hash": cfg["_hash"],
        "policy": cfg["policy"]["name"],
        "sim_family": cfg["simulator"]["family"],
        "profile": cfg["simulator"]["profile"]
    }

    steps = []
    eps = []

    for seed in cfg["seeds"]:

        set_seed(seed)

        # Build the policy once per seed.
        env, pol, pool = build(cfg, seed, tasks)

        n_tasks = cfg["data"].get("session_tasks", 8)

        for ep in range(cfg["n_episodes"]):

            # Reproducible but different task draw for every episode.
            episode_seed = int(
                np.random.SeedSequence(
                    [int(seed), int(ep)]
                ).generate_state(1)[0]
            )

            rng = np.random.default_rng(episode_seed)

            env_tasks = rng.choice(
                pool,
                size=n_tasks,
                replace=False
            ).tolist()

            # Update the existing P2 environment for this episode.
            env.tasks = env_tasks
            env.all_tasks = tasks
            env.seed = episode_seed

            state = env.reset()

            done = False
            total_reward = 0.0
            k = 0

            level_history = []
            full = 0
            first = 0
            info = {}

            while not done:

                action = int(
                    pol.get_action(state, is_training=True)
                )

                next_state, reward, done, info = env.step(action)

                pol.update(
                    state, action, reward, next_state, done
                )

                task = env_tasks[k]

                route = info.get("route", action // 5)
                actual_level = info.get(
                    "actual_level", action % 5
                )

                steps.append({
                    **meta,
                    "seed": seed,
                    "episode": ep,
                    "task_idx": k,
                    "action": action,
                    "route": route,
                    "level": actual_level,
                    "reward": round(reward, 4),
                    "success": info.get("immediate_success"),
                    "domain": task.get("domain"),

                    # Actual operation selected by P2 environment.
                    "op": info.get("operation"),

                    "time": info.get("time"),
                    "frustrated": info.get("frustrated"),
                    "abandoned": info.get("abandoned"),
                    "copied": info.get("copied")
                })

                level_history.append(actual_level)
                full += actual_level == 4
                first += route == 0

                total_reward += reward
                k += 1
                state = next_state

            # Transfer evaluation from P2 environment.
            pre = info.get("pre_transfer")
            post = info.get("transfer_success")

            # Preserve fractional transfer success rates.
            gain = None

            if pre is not None and post is not None and pre < 1:
                gain = round(
                    (float(post) - float(pre)) / (1 - float(pre)),
                    4
                )

            eps.append({
                **meta,
                "seed": seed,
                "episode": ep,
                "total_reward": round(total_reward, 4),
                "pre_transfer": pre,
                "post_transfer": post,
                "transfer_gain": gain,
                "n_steps": k,
                "full_assist_rate": round(full / k, 4),
                "mean_level": round(
                    float(np.mean(level_history)), 4
                ),
                "first_attempt_rate": round(first / k, 4),
                "assist_spent": int(sum(level_history)),
                "takeover_rate": info.get("takeover_rate")
            })

    out = ROOT / "results" / cfg["exp_id"]
    out.mkdir(parents=True, exist_ok=True)

    write_csv(out / "steps.csv", cfg, STEPS_COLS, steps)
    write_csv(out / "episodes.csv", cfg, EPS_COLS, eps)

    shutil.copy(args.config, out / "config.yaml")

    print(
        f"Wrote {len(steps)} step rows, "
        f"{len(eps)} episode rows -> {out}"
    )


if __name__ == "__main__":
    main()
