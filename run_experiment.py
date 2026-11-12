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

# Added "phase" to distinguish train vs eval
STEPS_COLS = [
    "exp_id", "config_hash", "policy", "sim_family", "profile",
    "seed", "phase", "episode", "task_idx", "action", "route", "level",
    "reward", "success", "domain", "op", "time", "frustrated",
    "abandoned", "copied"
]

EPS_COLS = [
    "exp_id", "config_hash", "policy", "sim_family", "profile",
    "seed", "phase", "episode", "total_reward", "pre_transfer",
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
    from rl.configured_env import ConfiguredEnv
    from rl.registry import make_policy, reward_for

    pool = session_pool(all_tasks)
    n_tasks = cfg["data"].get("session_tasks", 8)

    if len(pool) < n_tasks:
        raise ValueError(f"Need {n_tasks} session tasks, but only {len(pool)} available")

    env = ConfiguredEnv(
        pool[:n_tasks],
        reward=reward_for(cfg),
        discretize=cfg.get("state", {}).get("discretize", True),
        transfer_repeats=cfg.get("transfer_repeats", 20),
        drop=cfg.get("state", {}).get("drop", []),
        profile=cfg["simulator"]["profile"],
        family=cfg["simulator"]["family"],
        seed=seed,
        all_tasks=all_tasks,
    )
    return env, make_policy(cfg["policy"], seed), pool

def write_csv(path, cfg, cols, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"# config_hash={cfg['_hash']} exp_id={cfg['exp_id']}\n")
        f.write(",".join(cols) + "\n")
        for row in rows:
            f.write(
                ",".join("" if row.get(col) is None else str(row.get(col)) for col in cols) + "\n"
            )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
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
        env, pol, pool = build(cfg, seed, tasks)

        n_tasks = cfg["data"].get("session_tasks", 8)
        n_train_eps = cfg["n_episodes"]
        n_eval_eps = cfg.get("n_eval_episodes", 50) # Fallback to 50 if not in config

        def run_phase(phase_name, num_episodes, is_training):
            for ep in range(num_episodes):
                episode_seed = int(np.random.SeedSequence([int(seed), int(ep), int(is_training)]).generate_state(1)[0])
                rng = np.random.default_rng(episode_seed)
                env_tasks = rng.choice(pool, size=n_tasks, replace=False).tolist()

                env.tasks = env_tasks
                env.all_tasks = tasks
                env.seed = episode_seed
                state = env.reset()

                done = False
                total_reward = 0.0
                k = 0
                level_history = []
                full, first = 0, 0
                info = {}

                while not done:
                    # Backward compatibility if you haven't updated env and policy yet
                    valid_actions = getattr(env, "get_valid_actions", lambda: list(range(10)))()
                    
                    try:
                        action = int(pol.get_action(state, valid_actions, is_training=is_training))
                    except TypeError:
                        action = int(pol.get_action(state, is_training=is_training))

                    next_state, reward, done, info = env.step(action)
                    next_valid_actions = getattr(env, "get_valid_actions", lambda: list(range(10)))() if not done else []

                    if is_training:
                        try:
                            pol.update(state, action, reward, next_state, done, next_valid_actions)
                        except TypeError:
                            pol.update(state, action, reward, next_state, done)

                    task = env_tasks[k]
                    route = info.get("route", action // 5)
                    actual_level = info.get("actual_level", action % 5)

                    steps.append({
                        **meta, "seed": seed, "phase": phase_name, "episode": ep,
                        "task_idx": k, "action": action, "route": route, 
                        "level": actual_level, "reward": round(reward, 4),
                        "success": info.get("immediate_success"),
                        "domain": task.get("domain"), "op": info.get("operation"),
                        "time": info.get("time"), "frustrated": info.get("frustrated"),
                        "abandoned": info.get("abandoned"), "copied": info.get("copied")
                    })

                    level_history.append(actual_level)
                    full += actual_level == 4
                    first += route == 0
                    total_reward += reward
                    k += 1
                    state = next_state

                pre, post = info.get("pre_transfer"), info.get("transfer_success")
                gain = round((float(post) - float(pre)) / (1 - float(pre)), 4) if (pre is not None and post is not None and pre < 1) else None

                eps.append({
                    **meta, "seed": seed, "phase": phase_name, "episode": ep,
                    "total_reward": round(total_reward, 4), "pre_transfer": pre,
                    "post_transfer": post, "transfer_gain": gain, "n_steps": k,
                    "full_assist_rate": round(full / k, 4) if k > 0 else 0, 
                    "mean_level": round(float(np.mean(level_history)), 4) if level_history else 0,
                    "first_attempt_rate": round(first / k, 4) if k > 0 else 0, 
                    "assist_spent": int(sum(level_history)),
                    "takeover_rate": info.get("takeover_rate")
                })

        # Run training loop then frozen-policy evaluation loop
        run_phase("train", n_train_eps, is_training=True)
        run_phase("eval", n_eval_eps, is_training=False)

    out = ROOT / "results" / cfg["exp_id"]
    out.mkdir(parents=True, exist_ok=True)

    write_csv(out / "steps.csv", cfg, STEPS_COLS, steps)
    write_csv(out / "episodes.csv", cfg, EPS_COLS, eps)

    shutil.copy(args.config, out / "config.yaml")

    print(f"Wrote {len(steps)} step rows, {len(eps)} episode rows -> {out}")


if __name__ == "__main__":
    main()